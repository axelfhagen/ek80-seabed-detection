"""Deterministic snapshot of the Slurm training fleet for the autoresearch routine.

Gathers, in one shot, everything an automated agent needs to decide what to do
next without overloading the cluster:

  - active jobs (running + pending) and how many GPUs they hold
  - fairshare / priority health per account, and which account to prefer
  - recently ended jobs classified as completed / timeout / failed / oom / cancelled
  - checkpoint dirs that look incomplete (no eval) and are resume candidates
  - a submission budget: how many *new* jobs it is safe to launch right now

Prints a human-readable report and, at the end, a single machine-readable JSON
block (between BEGIN_JSON / END_JSON markers) so the agent can parse the verdict.

Usage:
    python fleet_status.py                 # default 12h lookback
    python fleet_status.py --hours 24
    python fleet_status.py --json           # only emit the JSON block
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

USER = "axelfh"
ACCOUNTS = ["share-ie-idi", "share-ie-imf"]
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

# Guardrails — tune these to taste. They gate NEW experiments only.
# Resuming timeouts and resubmitting crashed/failed runs is ALWAYS allowed and
# is never limited by these numbers.
#
# Delivery is only days away, so the posture is deliberately aggressive: keeping
# GPUs busy is worth more than conserving quota. The gate is now PER ACCOUNT —
# idi and imf are treated independently, so when one account is busy or has low
# fairshare we can still pile new work onto the other. We only refuse new work
# when BOTH accounts are full or critically starved.
PER_ACCOUNT_SATURATED = 8         # user's GPU jobs on ONE account before it's "full"
MAX_NEW_EXPERIMENTS = 6           # never launch more than this many new sweeps per tick
FAIRSHARE_HOLD = 0.05             # below this on BOTH accounts, priority is "low"
FAIRSHARE_HEALTHY = 0.30          # at/above this, priority is comfortable
SATURATED_AT = PER_ACCOUNT_SATURATED * len(ACCOUNTS)  # global cap, for display only


def _array_multiplier(job_id: str, state: str) -> int:
    """A pending array like 24540956_[1-9%5] will materialise into several
    concurrent jobs. Estimate that footprint so the budget isn't fooled by the
    single squeue line. Running array elements appear as separate lines already
    (job_id like 24540956_3), so they count as 1 each."""
    if "[" not in job_id:
        return 1
    spec = job_id[job_id.index("[") + 1: job_id.rindex("]")] if "]" in job_id else ""
    conc = None
    if "%" in spec:
        try:
            conc = int(spec.split("%")[1])
        except ValueError:
            conc = None
        spec = spec.split("%")[0]
    # Count remaining elements in the range list (e.g. "1-9" or "1,3,5").
    total = 0
    for part in spec.split(","):
        if "-" in part:
            try:
                a, b = part.split("-")
                total += int(b) - int(a) + 1
            except ValueError:
                total += 1
        elif part:
            total += 1
    total = total or 1
    return min(conc, total) if conc else total


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001 - report, don't crash the snapshot
        return f"__ERROR__ {e}"


def get_active_jobs() -> list[dict]:
    out = sh(["squeue", "-u", USER, "-h", "-o", "%i|%j|%T|%M|%L|%P|%b|%a"])
    jobs = []
    for line in out.splitlines():
        if not line.strip() or line.startswith("__ERROR__"):
            continue
        jid, name, state, elapsed, left, part, gres, acct = (line.split("|") + [""] * 8)[:8]
        gpus = 0
        if "gpu" in gres:
            tail = gres.rsplit(":", 1)[-1]
            gpus = int(tail) if tail.isdigit() else 1
        gpus *= _array_multiplier(jid, state)
        jobs.append({
            "job_id": jid, "name": name, "state": state, "elapsed": elapsed,
            "time_left": left, "partition": part, "gres": gres, "gpus": gpus,
            "account": acct.strip(),
        })
    return jobs


def get_fairshare() -> dict[str, dict]:
    out = sh(["sshare", "-U", "-u", USER, "-P",
              "-o", "Account,FairShare,EffectvUsage,NormShares"])
    fs: dict[str, dict] = {}
    for line in out.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        acct, fairshare, eff, norm = parts[:4]

        def _f(x):
            try:
                return float(x)
            except ValueError:
                return None

        fs[acct.strip()] = {
            "fairshare": _f(fairshare),
            "effective_usage": _f(eff),
            "norm_shares": _f(norm),
        }
    return fs


def get_recent_jobs(hours: int) -> list[dict]:
    out = sh(["sacct", "-X", "--starttime", f"now-{hours}hours", "-P",
              "-o", "JobID,JobName,State,Elapsed,Timelimit,ExitCode,End", "-u", USER])
    jobs = []
    for line in out.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 7:
            continue
        jid, name, state, elapsed, tlimit, exitcode, end = parts[:7]
        jobs.append({
            "job_id": jid, "name": name, "state": state.split(" ")[0],
            "state_full": state, "elapsed": elapsed, "timelimit": tlimit,
            "exit_code": exitcode, "end": end,
        })
    return jobs


def classify_recent(jobs: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "completed": [], "timeout": [], "failed": [], "oom": [],
        "cancelled": [], "running": [], "other": [],
    }
    for j in jobs:
        s = j["state"].upper()
        ec = j.get("exit_code", "0:0")
        if s == "COMPLETED":
            buckets["completed"].append(j)
        elif s == "TIMEOUT":
            buckets["timeout"].append(j)
        elif s == "RUNNING" or s == "PENDING":
            buckets["running"].append(j)
        elif s in ("OUT_OF_MEMORY", "OOM"):
            buckets["oom"].append(j)
        elif s == "FAILED" or (ec not in ("0:0", "0:15") and s not in ("CANCELLED",)):
            # 137 / 9 in exit code often means OOM-killed
            if ec.split(":")[-1] in ("137", "9"):
                buckets["oom"].append(j)
            else:
                buckets["failed"].append(j)
        elif s == "CANCELLED":
            buckets["cancelled"].append(j)
        else:
            buckets["other"].append(j)
    return buckets


ACTIVE_TOUCH_MINUTES = 25  # a checkpoint/log touched this recently => still training
LOG_DIR = Path(__file__).resolve().parent / "slurm" / "logs"


def _ckpt_key(name: str) -> str:
    """Token shared between a checkpoint dir and its Slurm log filename.
    exp_<grid>_<task>_<ts>  -> <grid>_<task>_<ts>  (unique per launch_sweep run)
    chunk_<grid>_<task>     -> <grid>_<task>       (matches t8h_* logs)"""
    for prefix in ("exp_", "chunk_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _latest_log_mtime(key: str, recent_logs: list[tuple[str, float]]) -> float | None:
    """Newest mtime among logs whose filename contains this dir's key.
    A still-running job appends an epoch line to its log every epoch (even on a
    plateau, when *_best.pth stops changing), so log mtime tracks liveness."""
    hits = [m for fname, m in recent_logs if key in fname]
    return max(hits) if hits else None


def scan_incomplete_checkpoints(hours: int) -> tuple[list[dict], list[dict]]:
    """Split checkpoint dirs that lack eval_metrics.json into:
      - in_progress: a job is almost certainly still writing it (its checkpoint
        OR its Slurm log was touched in the last ACTIVE_TOUCH_MINUTES), and
      - resume_candidates: no eval and gone quiet -> interrupted (timeout/crash).
    A dir with eval_metrics.json is considered finished and ignored.

    CRITICAL: a *running* job that has plateaued stops rewriting *_best.pth, so
    the checkpoint mtime alone would mis-flag it as interrupted. We therefore
    also consult the job's log mtime, which keeps advancing every epoch."""
    if not CKPT_DIR.is_dir():
        return [], []
    now = datetime.now().timestamp()
    cutoff = now - hours * 3600

    recent_logs: list[tuple[str, float]] = []
    if LOG_DIR.is_dir():
        for lf in LOG_DIR.glob("*.out"):
            try:
                m = lf.stat().st_mtime
            except OSError:
                continue
            if m >= cutoff:
                recent_logs.append((lf.name, m))

    in_progress, resume = [], []
    candidates = list(CKPT_DIR.glob("exp_*")) + list(CKPT_DIR.glob("chunk_*"))
    for d in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        ckpts = list(d.glob("*_best.pth"))
        if not ckpts:
            continue
        if (d / "eval_metrics.json").exists():
            continue
        pth_mtime = max(c.stat().st_mtime for c in ckpts)
        log_mtime = _latest_log_mtime(_ckpt_key(d.name), recent_logs)
        liveness = max([pth_mtime] + ([log_mtime] if log_mtime else []))
        if liveness < cutoff:
            continue  # too old to be part of the current campaign
        hist = d / "training_history.json"
        last_epoch = None
        if hist.exists():
            try:
                h = json.loads(hist.read_text())
                last_epoch = len(h.get("val_iou") or h.get("val_loss") or [])
            except Exception:  # noqa: BLE001
                pass
        mins = (now - liveness) / 60.0
        rec = {
            "dir": str(d), "name": d.name, "last_epoch": last_epoch,
            "minutes_since_touch": round(mins, 1),
            "ckpt_mtime": datetime.fromtimestamp(pth_mtime).isoformat(timespec="seconds"),
            "log_active": bool(log_mtime and (now - log_mtime) / 60.0 < ACTIVE_TOUCH_MINUTES),
        }
        (in_progress if mins < ACTIVE_TOUCH_MINUTES else resume).append(rec)
    return resume, in_progress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=12, help="Lookback window for ended jobs")
    p.add_argument("--json", action="store_true", help="Emit only the JSON summary")
    args = p.parse_args()

    active = get_active_jobs()
    fairshare = get_fairshare()
    recent = get_recent_jobs(args.hours)
    buckets = classify_recent(recent)
    incomplete, in_progress = scan_incomplete_checkpoints(args.hours)

    active_gpu = sum(j["gpus"] for j in active if j["state"] in ("RUNNING", "PENDING"))
    running = [j for j in active if j["state"] == "RUNNING"]
    pending = [j for j in active if j["state"] == "PENDING"]

    # Per-account view: how many of the user's GPU jobs sit on each account, and
    # that account's fairshare. This is the cross-account headroom check — if the
    # user's jobs are concentrated on idi while imf is free (or vice versa), we
    # can still launch new work on the freer account.
    def _acct_fs(a):
        return (fairshare.get(a) or {}).get("fairshare")

    account_active = {a: 0 for a in ACCOUNTS}
    for j in active:
        if j["state"] in ("RUNNING", "PENDING"):
            a = j.get("account") or ""
            if a in account_active:
                account_active[a] += j["gpus"]

    # An account has headroom when the user isn't already saturating it AND its
    # fairshare isn't critically low. Missing fairshare is treated as usable.
    account_headroom = {}
    for a in ACCOUNTS:
        fs = _acct_fs(a)
        not_low = fs is None or fs >= FAIRSHARE_HOLD
        not_full = account_active.get(a, 0) < PER_ACCOUNT_SATURATED
        account_headroom[a] = bool(not_low and not_full)

    submittable = [a for a in ACCOUNTS if account_headroom[a]]

    # Prefer a submittable account, picking the highest fairshare so load spreads
    # to the freer side automatically. Fall back to best overall for reporting.
    pool = submittable or ACCOUNTS
    best_acct, best_fs = None, -1.0
    for acct in pool:
        fs = _acct_fs(acct)
        if fs is not None and fs > best_fs:
            best_acct, best_fs = acct, fs

    # NEW-experiment gate (resumes/fixes are NOT gated by any of this).
    # We can submit as long as ANY account has headroom; we only refuse when both
    # are full or starved.
    can_submit_new = len(submittable) > 0
    queue_saturated = all(account_active.get(a, 0) >= PER_ACCOUNT_SATURATED for a in ACCOUNTS)
    priority_low = all(
        (_acct_fs(a) is not None and _acct_fs(a) < FAIRSHARE_HOLD) for a in ACCOUNTS
    )
    free_slots = sum(max(0, PER_ACCOUNT_SATURATED - account_active.get(a, 0)) for a in submittable)
    new_experiment_slots = min(MAX_NEW_EXPERIMENTS, free_slots) if can_submit_new else 0

    if priority_low:
        priority_note = f"LOW (fairshare {best_fs:.3f} < {FAIRSHARE_HOLD})"
    elif best_fs >= FAIRSHARE_HEALTHY:
        priority_note = f"HEALTHY (fairshare {best_fs:.3f})"
    else:
        priority_note = f"MODERATE (fairshare {best_fs:.3f})"

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lookback_hours": args.hours,
        "active_gpu_jobs": active_gpu,
        "running": len(running),
        "pending": len(pending),
        "saturated_at": SATURATED_AT,
        "queue_saturated": queue_saturated,
        "fairshare": fairshare,
        "account_active_gpu_jobs": account_active,
        "account_headroom": account_headroom,
        "per_account_saturated_at": PER_ACCOUNT_SATURATED,
        "submittable_accounts": submittable,
        "preferred_account": best_acct,
        "preferred_fairshare": best_fs,
        "priority_status": priority_note,
        "priority_low": priority_low,
        "resume_and_fix_always_allowed": True,
        "can_submit_new_experiments": can_submit_new,
        "new_experiment_slots": new_experiment_slots,
        "max_new_experiments": MAX_NEW_EXPERIMENTS,
        "recent_counts": {k: len(v) for k, v in buckets.items()},
        "timeout_jobs": buckets["timeout"],
        "failed_jobs": buckets["failed"],
        "oom_jobs": buckets["oom"],
        "resume_candidates": incomplete,
        "in_progress_checkpoints": in_progress,
    }

    if not args.json:
        print("=" * 72)
        print(f" FLEET STATUS  ({summary['generated_at']}, last {args.hours}h)")
        print("=" * 72)

        print(f"\nActive GPU jobs: {active_gpu}  (running {len(running)}, pending {len(pending)})"
              f"  [saturated >= {SATURATED_AT}]")
        for j in active:
            print(f"  {j['job_id']:<18} {j['state']:<8} {j['name']:<16} "
                  f"used {j['elapsed']:>9}  left {j['time_left']:>9}  {j['gres']}")

        print("\nPriority / fairshare / cross-account headroom:")
        for acct in ACCOUNTS:
            info = fairshare.get(acct, {})
            head = "HEADROOM" if account_headroom.get(acct) else "full/low"
            print(f"  {acct:<16} fairshare={info.get('fairshare')}  "
                  f"effective_usage={info.get('effective_usage')}  "
                  f"my_gpu_jobs={account_active.get(acct, 0)}/{PER_ACCOUNT_SATURATED}  [{head}]")
        print(f"  -> preferred account: {best_acct}  ({priority_note})")
        print(f"  -> submittable accounts: {submittable or 'none'}")

        print(f"\nRecently ended ({args.hours}h):")
        for k in ("completed", "timeout", "failed", "oom", "cancelled"):
            if buckets[k]:
                print(f"  {k.upper()}: {len(buckets[k])}")
                for j in buckets[k]:
                    print(f"    {j['job_id']:<14} {j['name']:<18} "
                          f"{j['state_full']:<22} {j['elapsed']} / {j['timelimit']}  exit={j['exit_code']}")

        print(f"\nIn-progress (checkpoint touched < {ACTIVE_TOUCH_MINUTES} min ago, leave alone):")
        if in_progress:
            for c in in_progress:
                print(f"  {c['name']:<42} last_epoch={c['last_epoch']}  "
                      f"touched {c['minutes_since_touch']} min ago")
        else:
            print("  none")

        print("\nResume candidates (checkpoint present, no eval, gone quiet -> interrupted):")
        if incomplete:
            for c in incomplete:
                print(f"  {c['name']:<42} last_epoch={c['last_epoch']}  "
                      f"quiet for {c['minutes_since_touch']} min")
        else:
            print("  none")

        print("\nSubmission budget:")
        print("  resume timeouts / resubmit crashed+failed: ALWAYS ALLOWED")
        print(f"  new experiments: can_submit={can_submit_new}  slots={new_experiment_slots}"
              f"  (max {MAX_NEW_EXPERIMENTS}/tick; submit to any account with headroom; "
              f"refuse only when BOTH full/starved -> "
              f"queue_saturated={queue_saturated}, priority_low={priority_low})")
        print()

    print("BEGIN_JSON")
    print(json.dumps(summary, indent=2))
    print("END_JSON")


if __name__ == "__main__":
    main()
