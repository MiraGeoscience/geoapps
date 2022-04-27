#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from geoh5py.data import FloatData
from geoh5py.objects import Grid2D
from scipy.interpolate import interp1d

from geoapps.shared_utils.utils import rotate_xy
from geoapps.utils import soft_import


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [float(val) for val in string.split(",") if len(val) > 0]


def export_grid_2_geotiff(
    data: FloatData, file_name: str, wkt_code: str = None, data_type: str = "float"
):
    """
    Write a geotiff from float data stored on a Grid2D object.

    :param data: FloatData object with Grid2D parent.
    :param file_name: Output file name *.tiff.
    :param wkt_code: Well-Known-Text string used to assign a projection.
    :param data_type:
        Type of data written to the geotiff.
        'float': Single band tiff with data values.
        'RGB': Three bands tiff with the colormap values.

    Original Source:

        Cameron Cooke: http://cgcooke.github.io/GDAL/

    Modified: 2020-04-28
    """

    gdal = soft_import("osgeo", objects=["gdal"], interrupt=True)

    grid2d = data.parent

    assert isinstance(grid2d, Grid2D), f"The parent object must be a Grid2D entity."

    values = data.values.copy()
    values[(values > 1e-38) * (values < 2e-38)] = -99999

    # TODO Re-sample the grid if rotated
    # if grid2d.rotation != 0.0:

    driver = gdal.GetDriverByName("GTiff")

    # Chose type
    if data_type == "RGB":
        encode_type = gdal.GDT_Byte
        num_bands = 3
        if data.entity_type.color_map is not None:
            cmap = data.entity_type.color_map._values
            red = interp1d(
                cmap["Value"], cmap["Red"], bounds_error=False, fill_value="extrapolate"
            )(values)
            blue = interp1d(
                cmap["Value"],
                cmap["Blue"],
                bounds_error=False,
                fill_value="extrapolate",
            )(values)
            green = interp1d(
                cmap["Value"],
                cmap["Green"],
                bounds_error=False,
                fill_value="extrapolate",
            )(values)
            array = [
                red.reshape(grid2d.shape, order="F").T,
                green.reshape(grid2d.shape, order="F").T,
                blue.reshape(grid2d.shape, order="F").T,
            ]

            np.savetxt(
                file_name[:-4] + "_RGB.txt",
                np.c_[cmap["Value"], cmap["Red"], cmap["Green"], cmap["Blue"]],
                fmt="%.5e %i %i %i",
            )
        else:
            print("A color_map is required for RGB export.")
            return
    else:
        encode_type = gdal.GDT_Float32
        num_bands = 1
        array = values.reshape(grid2d.shape, order="F").T

    dataset = driver.Create(
        file_name,
        grid2d.shape[0],
        grid2d.shape[1],
        num_bands,
        encode_type,
    )

    # Get rotation
    angle = -grid2d.rotation
    vec = rotate_xy(np.r_[np.c_[1, 0], np.c_[0, 1]], [0, 0], angle)

    dataset.SetGeoTransform(
        (
            grid2d.origin["x"],
            vec[0, 0] * grid2d.u_cell_size,
            vec[0, 1] * grid2d.v_cell_size,
            grid2d.origin["y"],
            vec[1, 0] * grid2d.u_cell_size,
            vec[1, 1] * grid2d.v_cell_size,
        )
    )

    try:
        dataset.SetProjection(wkt_code)
    except ValueError:
        print(
            f"A valid well-known-text (wkt) code is required. Provided {wkt_code} not understood"
        )

    if num_bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(0, num_bands):
            dataset.GetRasterBand(i + 1).WriteArray(array[i])

    dataset.FlushCache()  # Write to disk.
