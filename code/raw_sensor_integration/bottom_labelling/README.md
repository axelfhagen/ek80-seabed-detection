# bottom_labelling — bottom detection for label generation

This submodule generates per-ping bottom traces from EK80 echograms. The
traces are converted to binary masks and used as training labels for the
U-Net segmentation model. There is no externally validated ground truth,
so we refer to these outputs as *reference labels*, not ground truth.

## Provenance and credit

The three per-ping detectors in [`detectors.py`](detectors.py) — `detect_bottom_simple`,
`detect_bottom_edge`, and `detect_bottom_gradient` — are numpy ports of the
corresponding xarray-based detectors from the open-source
[CRIMAC-classifiers-bottom](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifiers-bottom)
repository (MIT licensed, Copyright (c) 2021 NORCE / Contributors to the CRIMAC project).

The constants `SOUND_VELOCITY` and `_BEAM_WIDTH` in [`utilities.py`](utilities.py)
are taken from the upstream `parameters.py`. The back-step inner loop in
`detect_bottom_simple` is a direct port of CRIMAC's `_back_step_inner`.

The full upstream package is also vendored verbatim under
[`../rsi_bottomdetection/`](../rsi_bottomdetection/) with its original MIT
license file, as reference for the port.

All other detectors and post-processing in this submodule (DP smoothing,
phase coherence, matched filter, CFAR, wavelet, change-point, morphological,
temporal coherence, multi-frequency fusion, ensemble voting, snake refinement,
the offline pipeline) are original additions for this thesis.

## Layout

| File | Origin | Purpose |
| --- | --- | --- |
| `detectors.py` | Ported from CRIMAC | simple / edge / gradient per-ping detectors |
| `utilities.py` | Original (constants from CRIMAC) | range/heave/mask helpers |
| `ensemble.py` | Original | median voting, temporal post-processing, `detect_bottom_ensemble_pipeline` |
| `dp.py` | Original | pseudo-TVG + dynamic-programming smoothing + snake refinement |
| `phase.py` | Original | circular phase variance from complex samples |
| `matched_filter.py` | Original | synthetic pulse cross-correlation |
| `temporal_coherence.py` | Original | ping-to-ping amplitude stability |
| `morphological.py` | Original | image-processing-based bottom extraction |
| `cfar.py` | Original | constant false-alarm rate detector |
| `wavelet.py` | Original | continuous wavelet transform detector |
| `changepoint.py` | Original | statistical change-point detector |
| `multifreq.py` | Original | cross-frequency fusion |

`notebooks/` contains the method comparison (21), detection verification (22)
and a mask quality analysis variant (19). They run from that directory with
`sys.path.insert(0, '../../..')`. The other bottom-detection notebooks
(05, 12, 16, 18) are in `../notebooks/`.

## License

The ported code carries the upstream MIT license. See
[`../rsi_bottomdetection/LICENSE`](../rsi_bottomdetection/LICENSE) for the
full text. Original additions are released under the same terms.
