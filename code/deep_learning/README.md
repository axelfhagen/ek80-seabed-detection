# deep_learning

U-Net seabed segmentation for EK80 echosounder data: dataset construction,
models, training, inference and evaluation. Import as `deep_learning.*` with
`code/` on `sys.path`.

## Modules

- `echo_dataset.py` - patch extraction from zarr stores (two frequencies plus
  their difference), label masks, patch cache (`.npz` and the pre-resized
  `.npy` fast cache), PyTorch datasets and augmentation.
- `unet_model.py` - model creation and encoder freezing, losses (Dice/BCE,
  focal, Tversky, geometry-aware, regression), training and validation steps,
  metrics, sliding-window inference over full echograms.
- `evaluate.py` - per-file and aggregate metrics (IoU, Dice, precision, recall,
  bottom-depth error in metres), bottom-line extraction and evaluation plots.
- `bottom_extraction.py` - alternative bottom-line extraction from a
  probability map (contiguous run, gated, dynamic programming, Hampel filter,
  Otsu threshold) and a jumpiness measure.
- `architectures/` - model registry (`create_model`, `list_architectures`); see
  its README.
- `training/` - CLI training pipeline, configuration, diagnostics, evaluation
  entry points and Slurm scripts; see its README.

Train from the repository root with
`python code/deep_learning/training/train.py --help`.
