# -*- coding: utf-8 -*-
"""
Uses the lowest layer boundary in LSSS work files as the detected bottom.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import xarray as xr


def detect_bottom(zarr_data: xr.Dataset, work_dir: str) -> xr.DataArray:
    ping_time = zarr_data['ping_time']
    bottom_depth = xr.DataArray(name='bottom_depth', data=np.zeros(len(ping_time)), dims=['ping_time'],
                                coords={'ping_time': ping_time})
    raw_file_grouping = ping_time.groupby(zarr_data['raw_file'])
    for raw_file, indexes in raw_file_grouping.groups.items():
        work_file = f'{work_dir}/{Path(raw_file).stem}.work'
        if not Path(work_file).exists():
            continue
        depths = work_file_to_bottom_boundary_depths(work_file)
        index_start = indexes[0]
        bottom_depth[index_start:index_start + len(depths)] = depths

    bottom_depth = bottom_depth.where(bottom_depth > 0, np.nan)
    return bottom_depth


def work_file_to_bottom_boundary_depths(work_file: str) -> np.ndarray:
    tree = ET.parse(work_file)
    root = tree.getroot()

    number_of_pings = int(root.find('timeRange').get('numberOfPings'))
    max_depths = np.zeros(number_of_pings)

    connector_id_to_ping_offset = {}
    connectors = root.find('layerInterpretation').find('connectors').findall('connector')
    for connector in connectors:
        connector_id = connector.get('id')
        ping_offset = int(connector.find('connectorRep').get('pingOffset'))
        connector_id_to_ping_offset[connector_id] = ping_offset

    curve_boundaries = root.find('layerInterpretation').find('boundaries').findall('curveBoundary')
    for curve_boundary in curve_boundaries:
        start_connector_id = curve_boundary.get('startConnector')
        ping_offset = connector_id_to_ping_offset[start_connector_id]
        depths_as_strings = np.array(curve_boundary.find('curveRep').find('depths').text.split())
        depths = np.asarray(depths_as_strings, float)
        max_depths_slice = max_depths[ping_offset:ping_offset + len(depths)]
        max_depths_slice[:] = np.maximum.reduce([max_depths_slice, depths])

    return max_depths
