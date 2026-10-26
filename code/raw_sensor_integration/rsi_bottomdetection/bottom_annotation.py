# -*- coding: utf-8 -*-
"""
Conversion of bottom depths to annotation masks.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import pandas as pd
import xarray as xr
import dask.array as da
import numpy as np


def to_pandas(zarr_data: xr.Dataset, channel_index: int, bottom_depth: xr.DataArray) -> pd.DataFrame:
    bottom_depth = bottom_depth.dropna('ping_time')
    channel_id = zarr_data['channel_id'][channel_index].values

    df = bottom_depth.to_dataframe('mask_depth_upper')

    df.reset_index(level=0, inplace=True)

    df = df.assign(mask_depth_lower=9999,
                   priority=2,
                   acoustic_category=999,
                   proportion=1,
                   object_id='bottom',
                   channel_id=channel_id)

    return df


def _bottom_2d(x):
    """Create 2d bottom annotation"""
    bottom_mark = int(x[-1])
    x[:] = np.nan
    x[bottom_mark:] = 1
    return x


def to_xarray(zarr_data: xr.Dataset, channel_index: int, bottom_depth: xr.DataArray, attributes: dict) -> xr.Dataset:
    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]
    bottom_range = bottom_depth - heave_corrected_transducer_depth

    # Get indices of the bottom_range
    bottom_range_idx = da.searchsorted(da.from_array(zarr_data.range), bottom_range.data)

    # Append indices to the last range
    bottom_range_1 = zarr_data.sv.isel(frequency=0).data
    bottom_range_1[:, -1] = bottom_range_idx

    # Convert annotation to 2d array
    bottom_range_2 = da.apply_along_axis(_bottom_2d, 1, bottom_range_1, dtype='float32',
                                         shape=(bottom_range_1.shape[1],))

    # Create dataset
    ds = xr.Dataset(
        data_vars=dict(
            bottom_range=(['ping_time', 'range'], bottom_range_2),
        ),
        coords=dict(
            frequency=zarr_data['frequency'][0],
            range=zarr_data['range'],
            ping_time=zarr_data['ping_time'],
        ),
    )

    # Remove unused dims
    remove_list = list(filter(lambda s: s not in ['frequency', 'ping_time', 'range'], list(ds.coords)))
    ds = ds.drop_vars(remove_list)

    for key in attributes.keys():
        ds.attrs[key] = attributes[key]

    return ds


def enrich_dataset(ds: xr.Dataset, bottom_outputs: xr.Dataset, attributes: dict) -> xr.Dataset:
    """Attach detector-emitted variables to the default bottom annotation dataset."""
    enriched = ds.copy()
    for var_name in ['bottom_depth', 'bottom_index', 'bottom_confidence', 'bottom_low_confidence', 'bottom_is_dropout']:
        if var_name in bottom_outputs:
            enriched[var_name] = bottom_outputs[var_name]

    if 'bottom_range' in bottom_outputs:
        enriched['bottom_range'] = bottom_outputs['bottom_range']

    for key, value in attributes.items():
        enriched.attrs[key] = value

    for key, value in bottom_outputs.attrs.items():
        enriched.attrs[f'output_{key}'] = value

    return enriched
