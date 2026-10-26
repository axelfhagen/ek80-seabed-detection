"""Plot echograms from the 3 failed June 2025 raw-to-zarr conversions."""
import sys
sys.path.insert(0, '/cluster/work/axelfh/Master/code')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from raw_sensor_integration.ping_parser import extract_pings, pings_to_power_image
from raw_sensor_integration.raw_parser import read_raw_file, find_datagram_types
from raw_sensor_integration.metadata import get_channel_settings

RAW_FILES = [
    "/cluster/work/axelfh/Data_raw/EK80/D202506/D20250624/D20250624-T050243.raw",
    "/cluster/work/axelfh/Data_raw/EK80/D202506/D20250625/D20250625-T015027.raw",
    "/cluster/work/axelfh/Data_raw/EK80/D202506/D20250626/D20250626-T042617.raw",
]

OUT_DIR = "/cluster/work/axelfh/Master/code/plots_june2025"

import os
os.makedirs(OUT_DIR, exist_ok=True)

for raw_path in RAW_FILES:
    fname = os.path.basename(raw_path).replace('.raw', '')
    print(f"\n{'='*60}")
    print(f"Processing {fname}")
    print(f"  File size: {os.path.getsize(raw_path) / 1e9:.2f} GB")

    data = read_raw_file(raw_path)
    dtypes = find_datagram_types(data)
    print(f"  Datagram types: {list(dtypes.keys())}")
    for dt, offsets in dtypes.items():
        print(f"    {dt}: {len(offsets)} datagrams")

    try:
        settings = get_channel_settings(raw_path)
        channels = list(settings.keys())
        print(f"  Channels: {channels}")
        for ch, s in settings.items():
            print(f"    {ch}: freq={s.get('frequency', '?')}Hz, "
                  f"sample_interval={s.get('sample_interval', '?')}, "
                  f"max_range={s.get('max_range', '?')}m")
    except Exception as e:
        print(f"  Could not extract settings: {e}")
        channels = []

    pings = extract_pings(raw_path, max_pings=1000)
    if not pings:
        print("  No pings found!")
        continue

    ping_channels = sorted(set(p.channel_id for p in pings))
    print(f"  Extracted {len(pings)} pings across channels: {ping_channels}")

    n_channels = len(ping_channels)
    fig, axes = plt.subplots(1, n_channels, figsize=(7 * n_channels, 8))
    if n_channels == 1:
        axes = [axes]

    for ax, ch in zip(axes, ping_channels):
        power_db = pings_to_power_image(pings, ch)
        if power_db.size == 0:
            ax.set_title(f"{ch}\n(no data)")
            continue

        ch_pings = [p for p in pings if ch in p.channel_id]
        n_samples, n_pings_ch = power_db.shape

        vmin, vmax = np.percentile(power_db[np.isfinite(power_db)], [2, 98])
        im = ax.imshow(power_db, aspect='auto', cmap='viridis',
                       vmin=vmin, vmax=vmax, origin='upper')
        plt.colorbar(im, ax=ax, label='Power (dB)')

        freq_label = ch.split()[-1] if ' ' in ch else ch
        ax.set_title(f"{freq_label}\n{n_pings_ch} pings, {n_samples} samples")
        ax.set_xlabel('Ping index')
        ax.set_ylabel('Sample index')

    fig.suptitle(f"{fname}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{fname}_echogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

print("\nDone!")
