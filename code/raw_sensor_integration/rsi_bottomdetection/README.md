# rsi_bottomdetection — vendored CRIMAC source

This directory is a verbatim vendored copy of the `bottomdetection/` package
from the **CRIMAC-classifiers-bottom** repository, kept here for reference
and reproducibility.

- Upstream repository: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom
- Upstream license: MIT (see `LICENSE` in this directory)
- Upstream copyright: Copyright (c) 2021 NORCE / Contributors to the CRIMAC project

The original per-file MIT copyright headers are preserved in every `.py`
file in this directory.

## Files

The package implements xarray-based bottom detection for CRIMAC zarr stores:

- `simple_bottom_detector.py` — threshold + back-step
- `edge_bottom_detector.py` — Gaussian-derivative edge detector
- `angle_bottom_detector.py` — angle (split-beam) based detection
- `combined_bottom_detector.py` — edge + angle combination
- `dp_offline_bottom_detector.py` — dynamic-programming temporal smoothing
- `work_files_bottom_detector.py` — LSSS work-file fallback
- `bottom_detection_main.py`, `bottom_annotation.py`, `bottom_candidate.py`,
  `bottom_comparison.py`, `bottom_utils.py`, `parameters.py`, `docker_main.py`
  — supporting infrastructure

## Relationship to the project

This vendored copy is reference only. The project itself does not import from
`rsi_bottomdetection/` at runtime. The bottom-detection methods actually used
for label generation live in [`../bottom_labelling/`](../bottom_labelling/),
which contains numpy ports of the simple, edge, and gradient detectors plus
a number of additional methods specific to this thesis.

When updating the vendored copy, sync from the upstream commit and preserve
both the per-file MIT headers and the `LICENSE` file in this directory.
