# -*- coding: utf-8 -*-
"""
Compares two bottom detections.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr


def compare(zarr_file: str,
            a_bottom_parquet_file: str,
            b_bottom_parquet_file: str,
            out_parquet_file: str) -> None:
    """
    Compares two bottom detections for the same data file and writes the result to an output file.

    :param zarr_file: the sv data file
    :param a_bottom_parquet_file: the first bottom detection file
    :param b_bottom_parquet_file: the second bottom detection file
    :param out_parquet_file: the output file for the comparison
    """
    dataset = xr.open_zarr(zarr_file, chunks={'frequency': 'auto', 'ping_time': 'auto', 'range': -1})
    print()
    print(f'zarr_file: {zarr_file}')
    print(dataset)

    channel_index = 0

    channel_id = dataset['channel_id'][channel_index].values
    ping_time = dataset['ping_time'].values
    r = dataset['range']

    heave = dataset['heave']
    sv = dataset['sv'][channel_index]
    transducer_draft = dataset['transducer_draft']

    min_range = r[0]
    sample_distance = r[1] - r[0]
    heave_corrected_transducer_depth = heave + transducer_draft[channel_index]

    a_bottom_depth = read_bottom_depth(a_bottom_parquet_file, channel_id, ping_time)
    a_bottom_range = a_bottom_depth - heave_corrected_transducer_depth
    a_bottom_indexes = (a_bottom_range - min_range) / sample_distance

    b_bottom_depth = read_bottom_depth(b_bottom_parquet_file, channel_id, ping_time)
    b_bottom_range = b_bottom_depth - heave_corrected_transducer_depth
    b_bottom_indexes = (b_bottom_range - min_range) / sample_distance

    depth_diff = (a_bottom_depth - b_bottom_depth) \
        .rename('depth_diff')

    sv_sum_diff = xr.apply_ufunc(per_ping_sv_sum_diff,
                                 sv,
                                 a_bottom_indexes,
                                 b_bottom_indexes,
                                 input_core_dims=[['range'], [], []],
                                 kwargs={},
                                 vectorize=True,
                                 dask='parallelized',
                                 output_dtypes=[np.float64])
    imr_constant = 4.0 * np.pi * 1852.0 * 1852.0
    sa_diff = (imr_constant * sample_distance * sv_sum_diff) \
        .rename('sa_diff')

    a_has_bottom = a_bottom_depth.where(np.isnan(a_bottom_depth), 1.0).fillna(0.0)
    b_has_bottom = b_bottom_depth.where(np.isnan(b_bottom_depth), 1.0).fillna(0.0)
    has_bottom_diff = (a_has_bottom - b_has_bottom) \
        .rename('has_bottom_diff')

    df = depth_diff.to_dataframe()
    df.reset_index(level=0, inplace=True)
    df = df.assign(
        sa_diff=sa_diff,
        has_bottom_diff=has_bottom_diff,
    )

    table = pa.Table.from_pandas(df)
    with pq.ParquetWriter(out_parquet_file, table.schema) as writer:
        writer.write_table(table=table)

    print()
    print(f'out_parquet_file: {out_parquet_file}')
    print(table)


def read_bottom_depth(parquet_file: str, channel_id, ping_time) -> xr.DataArray:
    df = pd.read_parquet(parquet_file)
    df = df[df['channel_id'] == channel_id][df['object_id'] == 'bottom']
    df = df[['ping_time', 'mask_depth_upper']]
    df = df.set_index('ping_time')
    df = df.reindex(ping_time)
    return df['mask_depth_upper'][ping_time].to_xarray()


def per_ping_sv_sum_diff(sv: xr.DataArray, a_index: float, b_index: float) -> float:
    i_a = len(sv) if np.isnan(a_index) else int(a_index)
    i_b = len(sv) if np.isnan(b_index) else int(b_index)
    if i_a >= i_b:
        return sv[i_b:i_a].sum()
    else:
        return -sv[i_a:i_b].sum()


def print_comparison(comparison_parquet_file: str) -> None:
    df = pd.read_parquet(comparison_parquet_file)
    ping_time = df['ping_time']
    depth_diff = df['depth_diff']
    has_bottom_diff = df['has_bottom_diff']
    sa_diff = df['sa_diff']
    print()
    print(f'comparison_parquet_file: {comparison_parquet_file}')
    print(f'ping_time:')
    print(f'    min: {ping_time.min()}')
    print(f'    max: {ping_time.max()}')
    print(f'depth_diff:')
    print(f'    min: {depth_diff.min()}')
    print(f'    max: {depth_diff.max()}')
    print(f'    sum: {depth_diff.abs().sum()}')
    print(f'sa_diff:')
    print(f'    min: {sa_diff.min()}')
    print(f'    max: {sa_diff.max()}')
    print(f'    sum: {sa_diff.abs().sum()}')
    print(f'has_bottom_diff:')
    print(f'    count +1: {sum(map(lambda x: x > 0, has_bottom_diff))}')
    print(f'    count -1: {sum(map(lambda x: x < 0, has_bottom_diff))}')
