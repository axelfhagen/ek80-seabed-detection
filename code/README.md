# Seabed Detection in EK80 Echosounder Data

Deep learning approach to seabed detection in Simrad EK80 echograms, built around a U-Net trained on multi-frequency acoustic data.

## Structure

```text
project_paths.py               # REPO_ROOT / DATA_ROOT defaults, overridable via environment variables

raw_sensor_integration/
    raw_parser.py              # binary .raw file reading, datagram offset mapping
    ping_parser.py             # RAW3 decoding into Ping objects (complex64 samples)
    metadata.py                # channel settings + pulse duration from XML0 config
    gps_parser.py              # NME0 NMEA sentences -> lat/lon DataFrame
    bottom_labelling/          # bottom detection library (per-ping detectors + ensemble + offline pipeline)
    bottom_detection.py        # backward-compat shim re-exporting from bottom_labelling/
    bottom_detection_offline.py  # backward-compat shim re-exporting from bottom_labelling/
    data_loader.py             # EchogramData container, load_raw(), load_zarr(), raw_to_zarr()
    preprocessing.py           # normalization, downsampling, zarr Sv loading, patch extraction
    visualization.py           # echogram plotting helpers
    zarr_dataset.py            # save_patches_npz(), ZarrBottomDataset (npz-based workflow)
    bathymetry_*.py, advanced_bathymetry_*.py  # bathymetry maps from detected bottom depths
    rsi_bottomdetection/       # vendored CRIMAC-classifiers-bottom (MIT, reference only)
    notebooks/                 # numbered exploration notebooks
    slurm/                     # SLURM job scripts

deep_learning/
    unet_model.py              # smp.Unet wrapper, DiceBCE loss, training/inference
    architectures/             # model variants (smp models, vanilla U-Net, regression U-Net, lightweight CNN)
    echo_dataset.py            # patch extraction (3-ch or 1-ch), PyTorch Dataset, caching
    bottom_extraction.py       # bottom line from predicted masks
    evaluate.py                # per-file metrics (IoU, Dice, bottom depth error)
    training/
        config.py              # PipelineConfig dataclass with all hyperparameters
        datasets.py            # dataset variant specifications (DATASET_SPECS)
        train.py               # CLI entry point for end-to-end training
        notebooks/             # training experiment notebooks
        slurm/                 # SLURM job scripts for training and evaluation

validation/                    # conformal prediction, ensembles, uncertainty and dataset-shift analyses
data_cleaning/                 # patch quality metrics, scoring, re-labelling
data_visualization/            # thesis figure and analysis scripts, shared thesis_style.py
experiments/                   # self-contained experiments (e.g. hake fine-tuning)
thesis_metrics_regen/          # scripts that recompute the headline thesis numbers
Data/                          # notebook for downloading NMDC sensor data through its API

convert_raw_to_patches.py      # batch script: raw dirs -> zarr -> cached .npz patches
generate_all_datasets.py       # generate dataset variants with different hyperparameters
                               # each variant gets a params.json with extraction + training config
generate_patches_chunk.py      # SLURM job-array chunk worker for patch generation
infer_test_raw.py              # U-Net inference + 3x3 plot grids over a raw directory
scan_empty_patches.py          # report empty masks / all-zero patches in the patch caches
```

Most packages have their own `README.md` with a per-module listing.

Figure scripts write into `Figures/<topic>/` relative to the repository root. `Figures/`
is not included in this repository. Datasets, zarr stores and model checkpoints are
expected under `Datasets/` (relative to the repository root) and `DATA_ROOT`; neither is
included.

## Data flow

Raw EK80 `.raw` files are either loaded directly (`load_raw()`, uncalibrated power in dB) or converted to calibrated Sv zarr stores via echopype (`raw_to_zarr()`). The unified `load_zarr()` handles three zarr layouts (CRIMAC, echopype, dataset_zarr) and returns an `EchogramData` object with the same interface regardless of source.

All echograms follow the convention `(n_samples, n_pings)` -- depth on axis 0, pings on axis 1.

## Bottom detection

Three algorithms are ported from [CRIMAC-classifiers-bottom](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom) (MIT licensed, Copyright (c) 2021 NORCE / Contributors to the CRIMAC project) to work on plain numpy arrays:

- **Simple**: threshold + back-step along the leading edge of the bottom echo. Adaptive threshold (`max - 20 dB`) for uncalibrated data, fixed `-31 dB` for calibrated Sv.
- **Edge**: Gaussian-derivative convolution to pinpoint the leading edge. Kernel width is derived from pulse duration and beam geometry. More accurate on flat-bottom geometry.
- **Gradient**: threshold-free maximum positive gradient. Complementary to the other two.

The numpy ports live in `raw_sensor_integration/bottom_labelling/detectors.py`; the original xarray sources are vendored verbatim in `raw_sensor_integration/rsi_bottomdetection/` with the upstream `LICENSE`. See `raw_sensor_integration/bottom_labelling/README.md` for the per-file provenance table.

These are combined per-ping via median voting with outlier rejection (`ensemble_bottom`), then cleaned with a sliding-median temporal filter (`post_process_bottom`). The full pipeline is wrapped in `detect_bottom_ensemble_pipeline()`.

## U-Net training

The model is a `segmentation_models_pytorch` U-Net with a ResNet34 encoder (ImageNet pretrained). The default input is 3 channels: normalised Sv at 38 kHz, normalised Sv at 200 kHz, and their difference. Patches are 256 pings wide, resized to 512x512, with the binary "below-bottom" mask as the target. Single-frequency (1-channel) and alternative normalisation modes (symmetric, percentile-based) are also supported.

`generate_all_datasets.py` produces multiple dataset variants with different extraction hyperparameters into `Datasets/patch_datasets/`. Each variant directory contains a `params.json` documenting the extraction settings and recommended training parameters (`target_size`, `in_channels`).

Training uses DiceBCE loss, AdamW with differential learning rates (encoder 1e-4, decoder 1e-3), and ReduceLROnPlateau. The train/val split is at the file level to avoid leakage from overlapping patches. Inference runs a sliding window with overlap averaging.

## Setup

```bash
pip install -r requirements.txt
```

Paths default to the cluster layout used for the thesis. Set `REPO_ROOT` and `DATA_ROOT`
to run elsewhere (`code/project_paths.py` for Python, the same variables in the SLURM
scripts).

Data pipeline notebooks run from `code/raw_sensor_integration/notebooks/` with `sys.path.insert(0, '../..')`. Training notebooks in `code/deep_learning/training/notebooks/` use `sys.path.insert(0, '../../..')`.
