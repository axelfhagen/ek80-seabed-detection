# Seabed detection in Simrad EK80 echosounder data

Code and LaTeX sources for the Master's thesis *Deep Learning for Seabed Detection with
Conformal Prediction Bounds in Dual-Frequency EK80 Echosounder Data: A Repeated-Route Case Study
on the M/F Bastø Electric* (Industrial Mathematics, NTNU, June 2026). The thesis trains a U-Net
to segment the seabed in dual-frequency EK80 echograms recorded on the Bastø ferry route. It
compares the U-Net with classical bottom detectors and quantifies the uncertainty of the
predicted bottom line with conformal prediction.

**Thesis PDF:** [master-thesis.pdf](https://github.com/axelfhagen/ek80-seabed-detection/releases/download/v1.0.0/master-thesis.pdf)
(attached to the [v1.0.0 release](https://github.com/axelfhagen/ek80-seabed-detection/releases/tag/v1.0.0)).
To cite it, use the "Cite this repository" button or `CITATION.cff`.

## Main results

The labels are algorithmic: two classical detectors on the 38 kHz channel, averaged and smoothed
over time. They are not manually verified, so the agreement figures below measure how well the
U-Net reproduces these labels, not the true seabed.

- On a held-out Oct 2024 period (378 files on the same route, absent from training), the U-Net
  reached a mean IoU of 0.9972 (median 0.9980) and a mean depth deviation of 0.57 m. The same
  checkpoint kept a mean IoU of 0.990 and a mean depth deviation of 0.516 m on a later Feb–Jun
  2025 split.
- Larger encoders, ImageNet pretraining, seed ensembles and alternative losses gave little
  evidence of decisive improvement. Augmentations that mimic acquisition faults helped within an
  ablation subset.
- A split-conformal band calibrated on Jan 2025 covered the held-out Feb and Jun 2025 data at
  92.6% against a nominal 90%. A label-free band built from repeated crossings kept 90.2%
  coverage and cut the median half-width from 7.59 m to 3.59 m.
- On an external survey (NOAA Pacific hake, SH2306), both agreement and Bastø-calibrated coverage
  degraded. Recalibrating on 20 hake files restored about 90% coverage with wider bands, and
  fine-tuning on hake recovered both. The model adapts per domain rather than transferring as is.

## Data

No data is included. The thesis uses two datasets:

- **Bastø Electric** (training and within-route testing): raw EK80 recordings from the Frisk
  Oslofjord project, held on the University of Oslo Educloud Research platform (project ec291).
  They are not publicly downloadable. Access is arranged through dScience at UiO, with
  permission from Frisk Oslofjord. Appendix G of the thesis gives the request and download
  steps.
- **NOAA Pacific hake survey SH2306** (external transfer test): public EK80 water-column data,
  [doi:10.25921/X1V5-DX52](https://doi.org/10.25921/X1V5-DX52).

## Layout

```text
code/      Python packages, analysis scripts, notebooks and SLURM job scripts (see code/README.md)
tests/     pytest suite pinning the behaviour of the core packages
thesis/    LaTeX sources of the thesis text (see thesis/README.md)
```

The repository contains no data, model checkpoints or figures. The scripts expect survey data
and checkpoints under `Datasets/` and `code/checkpoints/` relative to the repository root, and
raw survey folders under `DATA_ROOT`. Both locations can be set through environment variables:

```sh
export REPO_ROOT=/path/to/this/repo      # default: the repository root
export DATA_ROOT=/path/to/raw/surveys    # default: /cluster/work/axelfh
```

## Setup

Python 3.11 or newer.

```sh
python -m venv .venv
.venv/bin/pip install -r code/requirements.txt pytest
.venv/bin/pytest
```

The tests use synthetic inputs only and run on CPU in a few seconds. Tests marked `xfail`
record suspected defects in the current code. They are kept so that a fix shows up as an
unexpected pass.

## Third-party code

`code/raw_sensor_integration/rsi_bottomdetection/` is vendored from
[CRIMAC-classifiers-bottom](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom)
(MIT, Copyright (c) 2021 NORCE). Its licence file is kept alongside it.

## Licence

The code is released under the MIT licence (see `LICENSE`). The vendored CRIMAC code keeps its
own MIT licence. The thesis text in `thesis/` is not covered by `LICENSE`. All rights to it are
reserved by the author.
