# Seabed detection in Simrad EK80 echosounder data

Code and LaTeX sources for a Master's thesis in Statistics and Applied Mathematics. The thesis
trains a U-Net to segment the seabed in multi-frequency EK80 echograms recorded on the Bastø
ferry route. It compares the U-Net with classical bottom detectors and quantifies the
uncertainty of the predicted bottom line with conformal prediction.

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
