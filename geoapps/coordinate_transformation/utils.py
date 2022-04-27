#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os

import numpy as np
from geoh5py.groups import Group
from geoh5py.objects import Grid2D
from geoh5py.workspace import Workspace

from geoapps.utils import soft_import


def geotiff_2_grid(
    workspace: Workspace,
    file_name: str,
    grid: Grid2D = None,
    grid_name: str = None,
    parent: Group = None,
) -> Grid2D | None:
    """
    Load a geotiff from file.

    :param workspace: Workspace to load the data into.
    :param file_name: Input file name with path.
    :param grid: Existing Grid2D object to load the data into. A new object is created by default.
    :param grid_name: Name of the new Grid2D object. Defaults to the file name.
    :param parent: Group entity to store the new Grid2D object into.

     :return grid: Grid2D object with values stored.
    """
    gdal = soft_import("osgeo", objects=["gdal"], interrupt=True)

    tiff_object = gdal.Open(file_name)
    band = tiff_object.GetRasterBand(1)
    temp = band.ReadAsArray()

    file_name = os.path.basename(file_name).split(".")[0]
    if grid is None:
        if grid_name is None:
            grid_name = file_name

        grid = Grid2D.create(
            workspace,
            name=grid_name,
            origin=[
                tiff_object.GetGeoTransform()[0],
                tiff_object.GetGeoTransform()[3],
                0,
            ],
            u_count=temp.shape[1],
            v_count=temp.shape[0],
            u_cell_size=tiff_object.GetGeoTransform()[1],
            v_cell_size=tiff_object.GetGeoTransform()[5],
            parent=parent,
        )

    assert isinstance(grid, Grid2D), "Parent object must be a Grid2D"

    # Replace 0 to nan
    values = temp.ravel()
    if np.issubdtype(values.dtype, np.integer):
        values = values.astype("int32")
        print(values)
    else:
        values[values == 0] = np.nan

    grid.add_data({file_name: {"values": values}})
    del tiff_object
    return grid
