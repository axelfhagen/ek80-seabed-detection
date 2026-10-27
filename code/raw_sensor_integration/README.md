# raw_sensor_integration

Parsing, loading and preprocessing of Simrad EK80 echosounder data (`.raw`
files and zarr stores), bottom detection for reference-label generation, and
exploratory bathymetry plots of the Moss-Horten ferry route.

Import as a package with `code/` on `sys.path`
(`from raw_sensor_integration import data_loader`).

## Library modules

| Module | Purpose |
| --- | --- |
| `raw_parser.py` | Read `.raw` files and locate datagrams by type |
| `metadata.py` | Configuration, channel settings and file summaries from `.raw` headers |
| `ping_parser.py` | Decode RAW3 ping datagrams (power, angles, complex samples) |
| `gps_parser.py` | Extract NMEA GPS fixes; haversine distance |
| `data_loader.py` | Unified loader for `.raw`, CRIMAC zarr, echopype zarr and dataset zarr; `raw_to_zarr()` conversion via echopype |
| `preprocessing.py` | Echogram normalisation, depth downsampling, Sv loading from zarr stores, patch extraction |
| `zarr_dataset.py` | npz patch tensors from CRIMAC zarr stores (older workflow; the main pipeline uses `deep_learning.echo_dataset`) |
| `visualization.py` | Echogram plotting helpers |
| `bottom_labelling/` | Bottom detectors and post-processing used to build reference labels; see `bottom_labelling/README.md` |
| `bottom_detection.py`, `bottom_detection_offline.py` | Backward-compatibility re-exports of `bottom_labelling` functions; imported by older code and tests |
| `rsi_bottomdetection/` | Vendored CRIMAC-classifiers-bottom (MIT, NORCE), kept verbatim as reference for the numpy ports |

## Scripts

Run from the repository root (`python code/raw_sensor_integration/<script>.py`).
Paths default to the cluster layout through `code/project_paths.py`
(`REPO_ROOT`, `DATA_ROOT` environment variables).

| Script | Output |
| --- | --- |
| `bathymetry_map.py` | Processes all echopype zarr stores into `Figures/bathymetry/` (scatter/gridded maps, depth profiles, cached `bathymetry_data.csv`); also imported by the two scripts below |
| `regenerate_bathymetry_depth300.py` | Re-plots the cached CSV with a 300 m depth cutoff into `Figures/bathymetry/depth_max_300/` |
| `bathymetry_aggregated.py` | Along-route median/IQR envelopes and density plots into `Figures/bathymetry/depth_max_300/aggregated/` |
| `advanced_bathymetry_plots.py` | Consistency map, speed-coloured tracks, depth envelope into `Figures/bathymetry/advanced/` |
| `advanced_bathymetry_v2.py` | Revised consistency map, hillshaded relief, 3D view, depth-vs-date, tidal signal into `Figures/bathymetry/advanced/v2/` |
| `advanced_bathymetry_v3.py` | Revised 3D view, depth-vs-date and tidal analysis into `Figures/bathymetry/advanced/v3/` |
| `outlier_rate_map.py` | Per-cell outlier rate and plane-fit residual maps into `Figures/bathymetry/advanced/v3/` |
| `plot_test_raw_gps.py` | GPS scatter map of the `test_raw_2` files into `Figures/test_raw_2_gps/` |
| `stitch_remaining.py` | Stitched echogram PNGs for low-ping extra datasets into `Datasets/extra_raw_echograms/stitched/` |

All bathymetry scripts after `bathymetry_map.py` read the cached
`Figures/bathymetry/bathymetry_data.csv`.

## slurm/

SLURM job scripts for the cluster: `run_bathymetry.sh`, `run_stitch_remaining.sh`,
`run_outlier_bottom_verify.sh` (runs `code/data_visualization/visualize_outlier_bottom_detection.py`),
`run_extract_test_calib_patches.sh` and `run_gen_jun2025_caches.sh` (patch-cache
warm-up for held-out surveys), `run_sv_distribution_comparison_v2.sh`
(runs `validation.dataset_shift.distribution_comparison`). Each sets
`REPO_ROOT` with a cluster default.

## Notebooks

`notebooks/` holds the numbered exploration notebooks (01-22) for the data
pipeline: raw format, pings, GPS quality, zarr parsing, CRIMAC and echopype
pipelines, dataset creation, bottom detection and label quality.
They are run from `notebooks/` and add `code/` to `sys.path` with
`sys.path.insert(0, '../..')`. `15_crimac_vs_echopype.py` is a script version
of the CRIMAC/echopype comparison (`--help` lists its path arguments).
`run_21_explore_extra_raw.sh` executes notebook 21 under SLURM.

`bottom_labelling/notebooks/` holds the bottom-detection method comparison
(21), detection verification (22), and a variant of the mask quality analysis
(19) that also merges `quality_score` from notebook 05.
