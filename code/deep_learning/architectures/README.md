# deep_learning.architectures

Model registry. `create_model(name, **kwargs)` builds a model by key and
ignores keyword arguments the architecture does not take, so one config dict
works for every model. `list_architectures()` returns the registered keys.

## Registered keys

- `smp_unet`, `smp_unet_scse`, `smp_fpn`, `smp_deeplabv3plus`, `smp_manet`,
  `smp_segformer` - segmentation_models_pytorch models with a pretrained
  encoder (`smp_models.py`).
- `unet_regression` - U-Net whose output is read as a per-ping depth
  distribution for bottom-line regression via soft-argmax
  (`regression_unet.py`).
- `vanilla_unet`, `vanilla_unet_small`, `vanilla_unet_large` - U-Net trained
  from scratch with configurable depth and filter count (`vanilla_unet.py`).
- `lightweight_cnn` - small encoder-decoder with depthwise-separable
  convolutions (`lightweight_cnn.py`).

## Modules

- `base.py` - `SegmentationModel` base class shared by all architectures.
- `__init__.py` - registry, `register` decorator, `create_model`,
  `list_architectures`.
