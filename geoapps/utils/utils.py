#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import gc
import json
import os
import re
import warnings
from uuid import UUID

import dask
import dask.array as da
import geoh5py
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from geoh5py.data import FloatData, IntegerData
from geoh5py.groups import Group
from geoh5py.objects import (
    BlockModel,
    CurrentElectrode,
    Grid2D,
    Octree,
    PotentialElectrode,
    Surface,
)
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from shapely.geometry import LineString, mapping
from SimPEG.electromagnetics.static.resistivity import Survey
from skimage.measure import marching_cubes
from sklearn.neighbors import KernelDensity


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [string_2_numeric(val) for val in string.split(",") if len(val) > 0]


def string_2_numeric(text: str) -> int | float | str:
    """Converts numeric string representation to int or string if possible."""
    try:
        text_as_float = float(text)
        text_as_int = int(text_as_float)
        return text_as_int if text_as_int == text_as_float else text_as_float
    except ValueError:
        return text


def sorted_alphanumeric_list(alphanumerics: list[str]) -> list[str]:
    """
    Sorts a list of stringd containing alphanumeric characters in readable way.

    Sorting precedence is alphabetical for all string components followed by
    numeric component found in string from left to right.

    :param alphanumerics: list of alphanumeric strings.

    :return : naturally sorted list of alphanumeric strings.
    """

    def sort_precedence(text):
        numeric_regex = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
        non_numeric = re.split(numeric_regex, text)
        numeric = [string_2_numeric(k) for k in re.findall(numeric_regex, text)]
        order = non_numeric + numeric
        return order

    return sorted(alphanumerics, key=sort_precedence)


def sorted_children_dict(
    object: UUID | Entity, workspace: Workspace = None
) -> dict[str, UUID]:
    """
    Uses natural sorting algorithm to order the keys of a dictionary containing
    children name/uid key/value pairs.

    If valid uuid entered calls get_entity.  Will return None if no object found
    in workspace for provided object

    :param object: geoh5py object containing children IntegerData, FloatData
        entities

    :return : sorted name/uid dictionary of children entities of object.

    """

    if isinstance(object, UUID):
        object = workspace.get_entity(object)[0]
        if not object:
            return None

    children_dict = {}
    for c in object.children:
        if not isinstance(c, (IntegerData, FloatData)):
            continue
        else:
            children_dict[c.name] = c.uid

    children_order = sorted_alphanumeric_list(list(children_dict.keys()))

    return {k: children_dict[k] for k in children_order}


def get_locations(workspace: Workspace, entity: UUID | Entity):
    """
    Returns entity's centroids or vertices.

    If no location data is found on the provided entity, the method will
    attempt to call itself on it's parent.

    :param workspace: Geoh5py Workspace entity.
    :param entity: Object or uuid of entity containing centroid or
        vertex location data.

    :return: Array shape(*, 3) of x, y, z location data

    """
    locations = None

    if isinstance(entity, UUID):
        entity = workspace.get_entity(entity)[0]

    if hasattr(entity, "centroids"):
        locations = entity.centroids
    elif hasattr(entity, "vertices"):
        locations = entity.vertices
    elif getattr(entity, "parent", None) is not None and entity.parent is not None:
        locations = get_locations(workspace, entity.parent)

    return locations


def find_value(labels: list, keywords: list, default=None) -> list:
    """
    Find matching keywords within a list of labels.

    :param labels: List of labels or list of [key, value] that may contain the keywords.
    :param keywords: List of keywords to search for.
    :param default: Default value be returned if none of the keywords are found.

    :return matching_labels: List of labels containing any of the keywords.
    """
    value = None
    for entry in labels:
        for string in keywords:

            if isinstance(entry, list):
                name = entry[0]
            else:
                name = entry

            if isinstance(string, str) and (
                (string.lower() in name.lower()) or (name.lower() in string.lower())
            ):
                if isinstance(entry, list):
                    value = entry[1]
                else:
                    value = name

    if value is None:
        value = default
    return value


def get_surface_parts(surface: Surface) -> np.ndarray:
    """
    Find the connected cells from a surface.

    :param surface: Input surface with cells property.

    :return parts: shape(*, 3)
        Array of parts for each of the surface vertices.
    """
    cell_sorted = np.sort(surface.cells, axis=1)
    cell_sorted = cell_sorted[np.argsort(cell_sorted[:, 0]), :]

    parts = np.zeros(surface.vertices.shape[0], dtype="int")
    count = 1
    for ii in range(cell_sorted.shape[0] - 1):

        if (
            (cell_sorted[ii, 0] in cell_sorted[ii + 1 :, :])
            or (cell_sorted[ii, 1] in cell_sorted[ii + 1 :, :])
            or (cell_sorted[ii, 2] in cell_sorted[ii + 1 :, :])
        ):
            parts[cell_sorted[ii, :]] = count
        else:
            parts[cell_sorted[ii, :]] = count
            count += 1

    parts[cell_sorted[-1, :]] = count

    return parts


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

    try:
        import gdal
    except ModuleNotFoundError:
        warnings.warn(
            "Modules 'gdal' is missing from the environment. "
            "Consider installing with: 'conda install -c conda-forge gdal'"
        )

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
    try:
        import gdal
    except ModuleNotFoundError:
        warnings.warn(
            "Modules 'gdal' is missing from the environment. "
            "Consider installing with: 'conda install -c conda-forge gdal'"
        )
        return

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


def export_curve_2_shapefile(
    curve, attribute: geoh5py.data.Data = None, wkt_code: str = None, file_name=None
):
    """
    Export a Curve object to *.shp

    :param curve: Input Curve object to be exported.
    :param attribute: Data values exported on the Curve parts.
    :param wkt_code: Well-Known-Text string used to assign a projection.
    :param file_name: Specify the path and name of the *.shp. Defaults to the current directory and `curve.name`.
    """
    try:
        import fiona
    except ModuleNotFoundError:
        warnings.warn(
            "Modules 'fiona' is missing from the environment. "
            "Consider installing with: 'conda install -c conda-forge fiona gdal'"
        )
        return

    attribute_vals = None

    if attribute is not None and curve.get_data(attribute):
        attribute_vals = curve.get_data(attribute)[0].values

    polylines, values = [], []
    for lid in curve.unique_parts:

        ind_line = np.where(curve.parts == lid)[0]
        polylines += [curve.vertices[ind_line, :2]]

        if attribute_vals is not None:
            values += [attribute_vals[ind_line]]

    # Define a polygon feature geometry with one attribute
    schema = {"geometry": "LineString"}

    if values:
        attr_name = attribute.replace(":", "_")
        schema["properties"] = {attr_name: "float"}
    else:
        schema["properties"] = {"id": "int"}

    with fiona.open(
        file_name + ".shp",
        "w",
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=wkt_code,
    ) as c:

        # If there are multiple geometries, put the "for" loop here
        for ii, poly in enumerate(polylines):

            if len(poly) > 1:
                pline = LineString(list(tuple(map(tuple, poly))))

                res = {}
                res["properties"] = {}

                if attribute and values:
                    res["properties"][attr_name] = np.mean(values[ii])
                else:
                    res["properties"]["id"] = ii

                # geometry of of the original polygon shapefile
                res["geometry"] = mapping(pline)
                c.write(res)


def calculate_2D_trend(
    points: np.ndarray, values: np.ndarray, order: int = 0, method: str = "all"
):
    """
    detrend2D(points, values, order=0, method='all')

    Function to remove a trend from 2D scatter points with values

    Parameters:
    ----------

    points: array or floats, shape(*, 2)
        Coordinates of input points

    values: array of floats, shape(*,)
        Values to be de-trended

    order: Order of the polynomial to be used

    method: str
        Method to be used for the detrending
            "all": USe all points
            "perimeter": Only use points on the convex hull


    Returns
    -------

    trend: array of floats, shape(*,)
        Calculated trend

    coefficients: array of floats, shape(order+1)
        Coefficients for the polynomial describing the trend

        trend = c[0] + points[:, 0] * c[1] +  points[:, 1] * c[2]
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError(
            "Polynomial 'order' should be an integer > 0. "
            f"Value of {order} provided."
        )

    ind_nan = ~np.isnan(values)
    loc_xy = points[ind_nan, :]
    values = values[ind_nan]

    if method == "perimeter":
        hull = ConvexHull(loc_xy[:, :2])
        # Extract only those points that make the ConvexHull
        loc_xy = loc_xy[hull.vertices, :2]
        values = values[hull.vertices]
    elif not method == "all":
        raise ValueError(
            "'method' must be either 'all', or 'perimeter'. " f"Value {method} provided"
        )

    # Compute center of mass
    center_x = np.sum(loc_xy[:, 0] * np.abs(values)) / np.sum(np.abs(values))
    center_y = np.sum(loc_xy[:, 1] * np.abs(values)) / np.sum(np.abs(values))

    polynomial = []
    xx, yy = np.triu_indices(order + 1)
    for x, y in zip(xx, yy):
        polynomial.append(
            (loc_xy[:, 0] - center_x) ** float(x)
            * (loc_xy[:, 1] - center_y) ** float(y - x)
        )
    polynomial = np.vstack(polynomial).T

    if polynomial.shape[0] <= polynomial.shape[1]:
        raise ValueError(
            "The number of input values must be greater than the number of coefficients in the polynomial. "
            f"Provided {polynomial.shape[0]} values for a {order}th order polynomial with {polynomial.shape[1]} coefficients."
        )

    params, _, _, _ = np.linalg.lstsq(polynomial, values, rcond=None)
    data_trend = np.zeros(points.shape[0])
    for count, (x, y) in enumerate(zip(xx, yy)):
        data_trend += (
            params[count]
            * (points[:, 0] - center_x) ** float(x)
            * (points[:, 1] - center_y) ** float(y - x)
        )
    print(
        f"Removed {order}th order polynomial trend with mean: {np.mean(data_trend):.6g}"
    )
    return data_trend, params


def weighted_average(
    xyz_in: np.ndarray,
    xyz_out: np.ndarray,
    values: list,
    max_distance: float = np.inf,
    n: int = 8,
    return_indices: bool = False,
    threshold: float = 1e-1,
) -> list:
    """
    Perform a inverse distance weighted averaging on a list of values.

    :param xyz_in: shape(*, 3) Input coordinate locations.
    :param xyz_out: shape(*, 3) Output coordinate locations.
    :param values: Values to be averaged from the input to output locations.
    :param max_distance: Maximum averaging distance beyond which values do not contribute to the average.
    :param n: Number of nearest neighbours used in the weighted average.
    :param return_indices: If True, return the indices of the nearest neighbours from the input locations.
    :param threshold: Small value added to the radial distance to avoid zero division.
        The value can also be used to smooth the interpolation.

    :return avg_values: List of values averaged to the output coordinates
    """
    n = np.min([xyz_in.shape[0], n])
    assert isinstance(values, list), "Input 'values' must be a list of numpy.ndarrays"

    assert all(
        [vals.shape[0] == xyz_in.shape[0] for vals in values]
    ), "Input 'values' must have the same shape as input 'locations'"

    avg_values = []
    for value in values:
        sub = ~np.isnan(value)
        tree = cKDTree(xyz_in[sub, :])
        rad, ind = tree.query(xyz_out, n)
        ind = np.c_[ind]
        rad = np.c_[rad]
        rad[rad > max_distance] = np.nan

        values_interp = np.zeros(xyz_out.shape[0])
        weight = np.zeros(xyz_out.shape[0])

        for ii in range(n):
            v = value[sub][ind[:, ii]] / (rad[:, ii] + threshold)
            values_interp = np.nansum([values_interp, v], axis=0)
            w = 1.0 / (rad[:, ii] + threshold)
            weight = np.nansum([weight, w], axis=0)

        values_interp[weight > 0] = values_interp[weight > 0] / weight[weight > 0]
        values_interp[weight == 0] = np.nan
        avg_values += [values_interp]

    if return_indices:
        return avg_values, ind

    return avg_values


def window_xy(
    x: np.ndarray, y: np.ndarray, window: dict[str, float], mask: np.array = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Window x, y coordinates with window limits built from center and size.

    Notes
    -----
    This formulation is restricted to window outside of a north-south,
    east-west oriented box.  If the data you wish to window has an
    orientation other than this, then consider using the filter_xy
    function which includes an optional rotation parameter.

    :param x: Easting coordinates, as vector or meshgrid-like array.
    :param y: Northing coordinates, as vector or meshgrid-like array.
    :param window: Window parameters describing a domain of interest.
        Must contain the following keys and values:
        window = {
            "center": [X: float, Y: float],
            "size": [width: float, height: float]
        }
    :param mask: Optionally provide an existing mask and return the union
        of the two masks and it's effect on x and y.

    :return: mask: Boolean mask that was applied to x, and y.
    :return: x[mask]: Masked input array x.
    :return: y[mask]: Masked input array y.


    """

    if ("center" in window.keys()) & ("size" in window.keys()):
        x_lim = [
            window["center"][0] - window["size"][0] / 2,
            window["center"][0] + window["size"][0] / 2,
        ]
        y_lim = [
            window["center"][1] - window["size"][1] / 2,
            window["center"][1] + window["size"][1] / 2,
        ]
    else:
        msg = f"Missing window keys: 'center' and 'size'."
        raise KeyError(msg)

    window_mask = x >= x_lim[0]
    window_mask &= x <= x_lim[1]
    window_mask &= y >= y_lim[0]
    window_mask &= y <= y_lim[1]

    if mask is not None:
        window_mask &= mask

    return window_mask, x[window_mask], y[window_mask]


def downsample_xy(
    x: np.ndarray, y: np.ndarray, distance: float, mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Downsample locations to approximate a grid with defined spacing.

    :param x: Easting coordinates, as a 1-dimensional vector.
    :param y: Northing coordinates, as a 1-dimensional vector.
    :param distance: Desired coordinate spacing.
    :param mask: Optionally provide an existing mask and return the union
        of the two masks and it's effect on x and y.

    :return: mask: Boolean mask that was applied to x, and y.
    :return: x[mask]: Masked input array x.
    :return: y[mask]: Masked input array y.

    """

    downsample_mask = np.ones_like(x, dtype=bool)
    xy = np.c_[x.ravel(), y.ravel()]
    tree = cKDTree(xy)

    mask_ind = np.where(downsample_mask)[0]
    nstn = xy.shape[0]
    for ii in range(nstn):
        if downsample_mask[mask_ind[ii]]:
            ind = tree.query_ball_point(xy[ii, :2], distance)
            downsample_mask[mask_ind[ind]] = False
            downsample_mask[mask_ind[ii]] = True

    if mask is not None:
        downsample_mask &= mask

    xy = xy[downsample_mask]
    return downsample_mask, xy[:, 0], xy[:, 1]


def downsample_grid(
    xg: np.ndarray, yg: np.ndarray, distance: float, mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample grid locations to approximate spacing provided by 'distance'.

    Notes
    -----
    This implementation is more efficient than the 'downsample_xy' function
    for locations on a regular grid.

    :param xg: Meshgrid-like array of Easting coordinates.
    :param yg: Meshgrid-like array of Northing coordinates.
    :param distance: Desired coordinate spacing.
    :param mask: Optionally provide an existing mask and return the union
        of the two masks and it's effect on xg and yg.

    :return: mask: Boolean mask that was applied to xg, and yg.
    :return: xg[mask]: Masked input array xg.
    :return: yg[mask]: Masked input array yg.

    """

    u_diff = lambda u: np.unique(np.diff(u, axis=1))[0]
    v_diff = lambda v: np.unique(np.diff(v, axis=0))[0]

    du = np.linalg.norm(np.c_[u_diff(xg), u_diff(yg)])
    dv = np.linalg.norm(np.c_[v_diff(xg), v_diff(yg)])
    u_ds = np.max([int(np.rint(distance / du)), 1])
    v_ds = np.max([int(np.rint(distance / dv)), 1])

    downsample_mask = np.zeros_like(xg, dtype=bool)
    downsample_mask[::v_ds, ::u_ds] = True

    if mask is not None:
        downsample_mask &= mask

    return downsample_mask, xg[downsample_mask], yg[downsample_mask]


def filter_xy(
    x: np.array,
    y: np.array,
    distance: float = None,
    window: dict = None,
    angle: float = None,
    mask: np.ndarray = None,
) -> np.array:
    """
    Window and down-sample locations based on distance and window parameters.

    :param x: Easting coordinates, as vector or meshgrid-like array
    :param y: Northing coordinates, as vector or meshgrid-like array
    :param distance: Desired coordinate spacing.
    :param window: Window parameters describing a domain of interest.
        Must contain the following keys and values:
        window = {
            "center": [X: float, Y: float],
            "size": [width: float, height: float]
        }
        May also contain an "azimuth" in the case of rotated x and y.
    :param angle: Angle through which the locations must be rotated
        to take on a east-west, north-south orientation.  Supersedes
        the 'azimuth' key/value pair in the window dictionary if it
        exists.
    :param mask: Boolean mask to be combined with filter_xy masks via
        logical 'and' operation.

    :return mask: Boolean mask to be applied input arrays x and y.
    """

    if mask is None:
        mask = np.ones_like(x, dtype=bool)

    azim = None
    if angle is not None:
        azim = angle
    elif window is not None:
        if "azimuth" in window.keys():
            azim = window["azimuth"]

    is_rotated = False if (azim is None) | (azim == 0) else True
    if is_rotated:
        xy_locs = rotate_xy(np.c_[x.ravel(), y.ravel()], window["center"], azim)
        xr = xy_locs[:, 0].reshape(x.shape)
        yr = xy_locs[:, 1].reshape(y.shape)

    if window is not None:

        if is_rotated:
            mask, _, _ = window_xy(xr, yr, window, mask=mask)
        else:
            mask, _, _ = window_xy(x, y, window, mask=mask)

    if distance not in [None, 0]:

        is_grid = False
        if x.ndim > 1:

            if is_rotated:
                u_diff = np.unique(np.round(np.diff(xr, axis=1), 8))
                v_diff = np.unique(np.round(np.diff(yr, axis=0), 8))
            else:
                u_diff = np.unique(np.round(np.diff(x, axis=1), 8))
                v_diff = np.unique(np.round(np.diff(y, axis=0), 8))

            is_grid = (len(u_diff) == 1) & (len(v_diff) == 1)

        if is_grid:
            mask, _, _ = downsample_grid(x, y, distance, mask=mask)
        else:
            mask, _, _ = downsample_xy(x, y, distance, mask=mask)

    return mask


def rotate_xy(xyz: np.ndarray, center: list, angle: float):
    """
    Perform a counterclockwise rotation on the XY plane about a center point.

    :param xyz: shape(*, 3) Input coordinates
    :param center: len(2) Coordinates for the center of rotation.
    :param  angle: Angle of rotation in degree
    """
    R = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)],
    ]

    locs = xyz.copy()
    locs[:, 0] -= center[0]
    locs[:, 1] -= center[1]

    xy_rot = np.dot(R, locs[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], locs[:, 2:]]


def running_mean(
    values: np.array, width: int = 1, method: str = "centered"
) -> np.array:
    """
    Compute a running mean of an array over a defined width.

    :param values: Input values to compute the running mean over
    :param width: Number of neighboring values to be used
    :param method: Choice between 'forward', 'backward' and ['centered'] averaging.

    :return mean_values: Averaged array values of shape(values, )
    """
    # Averaging vector (1/N)
    weights = np.r_[np.zeros(width + 1), np.ones_like(values)]
    sum_weights = np.cumsum(weights)

    mean = np.zeros_like(values)

    # Forward averaging
    if method in ["centered", "forward"]:
        padd = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padd)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padd = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padd)
        mean += (
            (cumsum[(width + 1) :] - cumsum[: (-width - 1)])
            / (sum_weights[(width + 1) :] - sum_weights[: (-width - 1)])
        )[::-1]

    if method == "centered":
        mean /= 2.0

    return mean


def tensor_2_block_model(workspace, mesh, name=None, parent=None, data={}):
    """
    Function to convert a tensor mesh from :obj:`~discretize.TensorMesh` to
    :obj:`~geoh5py.objects.block_model.BlockModel`
    """

    block_model = BlockModel.create(
        workspace,
        origin=[mesh.x0[0], mesh.x0[1], mesh.x0[2]],
        u_cell_delimiters=(mesh.vectorNx - mesh.x0[0]),
        v_cell_delimiters=(mesh.vectorNy - mesh.x0[1]),
        z_cell_delimiters=(mesh.vectorNz - mesh.x0[2]),
        name=name,
        parent=parent,
    )

    for name, model in data.items():
        modelMat = mesh.reshape(model, "CC", "CC", "M")

        # Transpose the axes
        modelMatT = modelMat.transpose((2, 0, 1))
        modelMatTR = modelMatT.reshape((-1, 1), order="F")

        block_model.add_data({name: {"values": modelMatTR}})

    return block_model


def block_model_2_tensor(block_model, models=[]):
    """
    Function to convert a :obj:`~geoh5py.objects.block_model.BlockModel`
    to tensor mesh :obj:`~discretize.TensorMesh`
    """

    from discretize import TensorMesh

    tensor = TensorMesh(
        [
            np.abs(block_model.u_cells),
            np.abs(block_model.v_cells),
            np.abs(block_model.z_cells),
        ],
        x0="CC0",
    )
    tensor.x0 = [
        block_model.origin["x"] + block_model.u_cells[block_model.u_cells < 0].sum(),
        block_model.origin["y"] + block_model.v_cells[block_model.v_cells < 0].sum(),
        block_model.origin["z"] + block_model.z_cells[block_model.z_cells < 0].sum(),
    ]
    out = []

    for model in models:
        values = model.copy().reshape((tensor.nCz, tensor.nCx, tensor.nCy), order="F")

        if tensor.x0[2] != block_model.origin["z"]:
            values = values[::-1, :, :]
        values = np.transpose(values, (1, 2, 0))

        values = values.reshape((-1, 1), order="F")
        out += [values]

    return tensor, out


def treemesh_2_octree(workspace, treemesh, **kwargs):

    indArr, levels = treemesh._ubc_indArr
    ubc_order = treemesh._ubc_order

    indArr = indArr[ubc_order] - 1
    levels = levels[ubc_order]

    origin = treemesh.x0.copy()
    origin[2] += treemesh.h[2].size * treemesh.h[2][0]
    mesh_object = Octree.create(
        workspace,
        origin=origin,
        u_count=treemesh.h[0].size,
        v_count=treemesh.h[1].size,
        w_count=treemesh.h[2].size,
        u_cell_size=treemesh.h[0][0],
        v_cell_size=treemesh.h[1][0],
        w_cell_size=-treemesh.h[2][0],
        octree_cells=np.c_[indArr, levels],
        **kwargs,
    )

    return mesh_object


def octree_2_treemesh(mesh):
    """
    Convert a geoh5 octree mesh to discretize.TreeMesh
    Modified code from module discretize.TreeMesh.readUBC function.
    """

    from discretize import TreeMesh

    tswCorn = np.asarray(mesh.origin.tolist())

    smallCell = [mesh.u_cell_size, mesh.v_cell_size, mesh.w_cell_size]

    nCunderMesh = [mesh.u_count, mesh.v_count, mesh.w_count]

    cell_sizes = [np.ones(nr) * sz for nr, sz in zip(nCunderMesh, smallCell)]
    u_shift, v_shift, w_shift = (np.sum(h[h < 0]) for h in cell_sizes)
    h1, h2, h3 = (np.abs(h) for h in cell_sizes)
    x0 = tswCorn + np.array([u_shift, v_shift, w_shift])

    ls = np.log2(nCunderMesh).astype(int)
    if ls[0] == ls[1] and ls[1] == ls[2]:
        max_level = ls[0]
    else:
        max_level = min(ls) + 1

    treemesh = TreeMesh([h1, h2, h3], x0=x0)

    # Convert indArr to points in coordinates of underlying cpp tree
    # indArr is ix, iy, iz(top-down) need it in ix, iy, iz (bottom-up)
    cells = np.vstack(mesh.octree_cells.tolist())

    levels = cells[:, -1]
    indArr = cells[:, :-1]

    indArr = 2 * indArr + levels[:, None]  # get cell center index
    indArr[:, 2] = 2 * nCunderMesh[2] - indArr[:, 2]  # switch direction of iz
    levels = max_level - np.log2(levels)  # calculate level

    treemesh.__setstate__((indArr, levels))

    return treemesh


def object_2_dataframe(entity, fields=[], inplace=False, vertices=True, index=None):
    """
    Convert an object to a pandas dataframe
    """
    if getattr(entity, "vertices", None) is not None:
        locs = entity.vertices
    elif getattr(entity, "centroids", None) is not None:
        locs = entity.centroids

    if index is None:
        index = np.arange(locs.shape[0])

    data_dict = {}
    if vertices:
        data_dict["X"] = locs[index, 0]
        data_dict["Y"] = locs[index, 1]
        data_dict["Z"] = locs[index, 2]

    d_f = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
    for field in fields:
        for data in entity.workspace.get_entity(field):
            if (data in entity.children) and (data.values.shape[0] == locs.shape[0]):
                d_f[data.name] = data.values.copy()[index]
                if inplace:
                    data.values = None

    return d_f


def csv_2_zarr(input_csv, out_dir="zarr", rowchunks=100000, dask_chunks="64MB"):
    """
    Zarr conversion for large CSV files

    NOTE: Need testing
    """
    # Need to run this part only once
    if ~os.path.exists(out_dir):
        for ii, chunk in enumerate(pd.read_csv(input_csv, chunksize=rowchunks)):
            array = chunk.to_numpy()[1:, :]
            da_array = da.from_array(array, chunks=dask_chunks)
            da.to_zarr(da_array, url=out_dir + rf"\Tile{ii}")

    # Just read the header
    header = pd.read_csv(input_csv, nrows=1)

    # Stack all the blocks in one big zarr
    count = len([name for name in os.listdir(out_dir)])
    dask_arrays = []
    for ii in range(count):
        block = da.from_zarr(out_dir + f"/Tile{ii}")
        dask_arrays.append(block)

    return header, da.vstack(dask_arrays)


def data_2_zarr(h5file, entity_name, downsampling=1, fields=[], zarr_file="data.zarr"):
    """
    Convert an data entity and values to a dictionary of zarr's
    """

    workspace = Workspace(h5file)
    entity = workspace.get_entity(entity_name)[0]

    if getattr(entity, "vertices", None) is not None:
        n_data = entity.n_vertices
    elif getattr(entity, "centroids", None) is not None:
        n_data = entity.n_cells
    del workspace, entity

    vec_len = int(np.ceil(n_data / downsampling))

    def load(field):
        """
        Load one column from geoh5
        """
        workspace = Workspace(h5file)
        entity = workspace.get_entity(entity_name)[0]
        obj = entity.get_data(field)[0]
        values = obj.values[::downsampling]
        if isinstance(obj, FloatData) and values.shape[0] == vec_len:
            values[(values > 1e-38) * (values < 2e-38)] = -99999
        else:
            values = np.ones(vec_len) * -99999
        del workspace, obj, entity
        gc.collect()
        return values

    row = dask.delayed(load, pure=True)

    make_rows = [row(field) for field in fields]

    delayed_array = [
        da.from_delayed(
            make_row, dtype=np.float32, shape=(np.ceil(n_data / downsampling),)
        )
        for make_row in make_rows
    ]

    stack = da.vstack(delayed_array)

    if os.path.exists(zarr_file):

        data_mat = da.from_zarr(zarr_file)

        if np.all(
            np.r_[
                np.any(np.r_[data_mat.chunks[0]] == stack.chunks[0]),
                np.any(np.r_[data_mat.chunks[1]] == stack.chunks[1]),
                np.r_[data_mat.shape] == np.r_[stack.shape],
            ]
        ):
            # Check that loaded G matches supplied data and mesh
            print("Zarr file detected with same shape and chunksize ... re-loading")

            return data_mat
        else:

            print("Zarr file detected with wrong shape and chunksize ... over-writing")

    with ProgressBar():
        print("Saving G to zarr: " + zarr_file)
        data_mat = da.to_zarr(
            stack,
            zarr_file,
            compute=True,
            return_stored=True,
            overwrite=True,
        )

    return data_mat


def rotate_vertices(xyz, center, phi, theta):
    """
    Rotate scatter points in column format around a center location

    INPUT
    :param: xyz nDx3 matrix
    :param: center xyz location of rotation
    :param: theta angle rotation around z-axis
    :param: phi angle rotation around x-axis

    """
    xyz -= np.kron(np.ones((xyz.shape[0], 1)), np.r_[center])

    phi = -np.deg2rad(np.asarray(phi))
    theta = np.deg2rad((450.0 - np.asarray(theta)) % 360.0)

    Rx = np.asarray(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )

    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    R = Rz.dot(Rx)

    xyzRot = R.dot(xyz.T).T

    return xyzRot + np.kron(np.ones((xyz.shape[0], 1)), np.r_[center])


def rotate_azimuth_dip(azimuth, dip):
    """
    dipazm_2_xyz(dip,azimuth)

    Function converting degree angles for dip and azimuth from north to a
    3-components in cartesian coordinates.

    INPUT
    dip     : Value or vector of dip from horizontal in DEGREE
    azimuth   : Value or vector of azimuth from north in DEGREE

    OUTPUT
    M       : [n-by-3] Array of xyz components of a unit vector in cartesian

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    azimuth = np.asarray(azimuth)
    dip = np.asarray(dip)

    # Number of elements
    nC = azimuth.size

    M = np.zeros((nC, 3))

    # Modify azimuth from North to cartesian-X
    inc = -np.deg2rad(np.asarray(dip))
    dec = np.deg2rad((450.0 - np.asarray(azimuth)) % 360.0)

    M[:, 0] = np.cos(inc) * np.cos(dec)
    M[:, 1] = np.cos(inc) * np.sin(dec)
    M[:, 2] = np.sin(inc)

    return M


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [float(val) for val in string.split(",") if len(val) > 0]


class RectangularBlock:
    """
    Define a rotated rectangular block in 3D space

    :param
        - length, width, depth: width, length and height of prism
        - center : center of prism in horizontal plane
        - dip, azimuth : dip and azimuth of prism
    """

    def __init__(self, **kwargs):

        self._center = [0.0, 0.0, 0.0]
        self._length = 1.0
        self._width = 1.0
        self._depth = 1.0
        self._dip = 0.0
        self._azimuth = 0.0
        self._vertices = None

        self.triangles = np.vstack(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 4],
                [1, 4, 5],
                [1, 3, 5],
                [3, 5, 7],
                [2, 3, 6],
                [3, 6, 7],
                [0, 2, 4],
                [2, 4, 6],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )

        for attr, item in kwargs.items():
            try:
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def center(self):
        """Prism center"""
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self._vertices = None

    @property
    def length(self):
        """"""
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self._vertices = None

    @property
    def width(self):
        """"""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._vertices = None

    @property
    def depth(self):
        """"""
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value
        self._vertices = None

    @property
    def dip(self):
        """"""
        return self._dip

    @dip.setter
    def dip(self, value):
        self._dip = value
        self._vertices = None

    @property
    def azimuth(self):
        """"""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value
        self._vertices = None

    @property
    def vertices(self):
        """
        Prism eight corners in 3D space
        """

        if getattr(self, "_vertices", None) is None:
            x1, x2 = [
                -self.length / 2.0 + self.center[0],
                self.length / 2.0 + self.center[0],
            ]
            y1, y2 = [
                -self.width / 2.0 + self.center[1],
                self.width / 2.0 + self.center[1],
            ]
            z1, z2 = [
                -self.depth / 2.0 + self.center[2],
                self.depth / 2.0 + self.center[2],
            ]

            block_xyz = np.asarray(
                [
                    [x1, x2, x1, x2, x1, x2, x1, x2],
                    [y1, y1, y2, y2, y1, y1, y2, y2],
                    [z1, z1, z1, z1, z2, z2, z2, z2],
                ]
            )

            xyz = rotate_vertices(block_xyz.T, self.center, self.dip, self.azimuth)

            self._vertices = xyz

        return self._vertices


def hex_to_rgb(hex):
    """
    Convert hex color code to RGB
    """
    code = hex.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def symlog(values, threshold):
    """
    Convert values to log with linear threshold near zero
    """
    return np.sign(values) * np.log10(1 + np.abs(values) / threshold)


def inv_symlog(values, threshold):
    """
    Compute the inverse symlog mapping
    """
    return np.sign(values) * threshold * (-1.0 + 10.0 ** np.abs(values))


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]

    return (data * x_indicies**i_order * y_indices**j_order).sum()


def random_sampling(
    values, size, method="histogram", n_bins=100, bandwidth=0.2, rtol=1e-4
):
    """
    Perform a random sampling of the rows of the input array based on
    the distribution of the columns values.

    Parameters
    ----------

    values: numpy.array of float
        Input array of values N x M, where N >> M
    size: int
        Number of indices (rows) to be extracted from the original array

    Returns
    -------
    indices: numpy.array of int
        Indices of samples randomly selected from the PDF
    """
    if size == values.shape[0]:
        return np.where(np.all(~np.isnan(values), axis=1))[0]
    else:
        if method == "pdf":
            kde_skl = KernelDensity(bandwidth=bandwidth, rtol=rtol)
            kde_skl.fit(values)
            probabilities = np.exp(kde_skl.score_samples(values))
            probabilities /= probabilities.sum()
        else:
            probabilities = np.zeros(values.shape[0])
            for ind in range(values.shape[1]):
                vals = values[:, ind]
                nnan = ~np.isnan(vals)
                pop, bins = np.histogram(vals[nnan], n_bins)
                ind = np.digitize(vals[nnan], bins)
                ind[ind > n_bins] = n_bins
                probabilities[nnan] += 1.0 / (pop[ind - 1] + 1)

    probabilities[np.any(np.isnan(values), axis=1)] = 0
    probabilities /= probabilities.sum()

    np.random.seed = 0
    return np.random.choice(
        np.arange(values.shape[0]), replace=False, p=probabilities, size=size
    )


def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return [x_centroid, y_centroid], cov


def ij_2_ind(coordinates, shape):
    """
    Return the index of ij coordinates
    """
    return [ij[0] * shape[1] + ij[1] for ij in coordinates]


def ind_2_ij(indices, shape):
    """
    Return the index of ij coordinates
    """
    return [[int(np.floor(ind / shape[1])), ind % shape[1]] for ind in indices]


def get_neighbours(index, shape):
    """
    Get all neighbours of cell in a 2D grid
    """
    j, i = int(np.floor(index / shape[1])), index % shape[1]
    vec_i = np.r_[i - 1, i, i + 1]
    vec_j = np.r_[j - 1, j, j + 1]

    vec_i = vec_i[(vec_i >= 0) * (vec_i < shape[1])]
    vec_j = vec_j[(vec_j >= 0) * (vec_j < shape[0])]

    ii, jj = np.meshgrid(vec_i, vec_j)

    return ij_2_ind(np.c_[jj.ravel(), ii.ravel()].tolist(), shape)


def get_active_neighbors(index, shape, model, threshold, blob_indices):
    """
    Given an index, append to a list if active
    """
    out = []
    for ind in get_neighbours(index, shape):
        if (model[ind] > threshold) and (ind not in blob_indices):
            out.append(ind)
    return out


def get_blob_indices(index, shape, model, threshold, blob_indices=[]):
    """
    Function to return indices of cells inside a model value blob
    """
    out = get_active_neighbors(index, shape, model, threshold, blob_indices)

    for neigh in out:
        blob_indices += [neigh]
        blob_indices = get_blob_indices(
            neigh, shape, model, threshold, blob_indices=blob_indices
        )

    return blob_indices


def format_labels(x, y, axs, labels=None, aspect="equal", tick_format="%i", **kwargs):
    if labels is None:
        axs.set_ylabel("Northing (m)")
        axs.set_xlabel("Easting (m)")
    else:
        axs.set_xlabel(labels[0])
        axs.set_ylabel(labels[1])
    xticks = np.linspace(x.min(), x.max(), 5)
    yticks = np.linspace(y.min(), y.max(), 5)

    axs.set_yticks(yticks)
    axs.set_yticklabels(
        [tick_format % y for y in yticks.tolist()], rotation=90, va="center"
    )
    axs.set_xticks(xticks)
    axs.set_xticklabels([tick_format % x for x in xticks.tolist()], va="center")
    axs.autoscale(tight=True)
    axs.set_aspect(aspect)


def input_string_2_float(input_string):
    """
    Function to input interval and value as string to a list of floats.

    Parameter
    ---------
    input_string: str
        Input string value of type `val1:val2:ii` and/or a list of values `val3, val4`


    Return
    ------
    list of floats
        Corresponding list of values in float format

    """
    if input_string != "":
        vals = re.split(",", input_string)
        cntrs = []
        for val in vals:
            if ":" in val:
                param = np.asarray(re.split(":", val), dtype="float")
                if len(param) == 2:
                    cntrs += [np.arange(param[0], param[1] + 1)]
                else:
                    cntrs += [np.arange(param[0], param[1] + param[2], param[2])]
            else:
                cntrs += [float(val)]
        return np.unique(np.sort(np.hstack(cntrs)))

    return None


def iso_surface(
    entity,
    values,
    levels,
    resolution=100,
    max_distance=np.inf,
):
    """
    Generate 3D iso surface from an entity vertices or centroids and values.

    Parameters
    ----------
    entity: geoh5py.objects
        Any entity with 'vertices' or 'centroids' attribute.

    values: numpy.ndarray
        Array of values to create iso-surfaces from.

    levels: list of floats
        List of iso values

    max_distance: float, default=numpy.inf
        Maximum distance from input data to generate iso surface.
        Only used for input entities other than BlockModel.

    resolution: int, default=100
        Grid size used to generate the iso surface.
        Only used for input entities other than BlockModel.

    Returns
    -------
    surfaces: list of numpy.ndarrays
        List of surfaces (one per levels) defined by
        vertices and cell indices.
        [(vertices, cells)_level_1, ..., (vertices, cells)_level_n]
    """
    if getattr(entity, "vertices", None) is not None:
        locations = entity.vertices
    elif getattr(entity, "centroids", None) is not None:
        locations = entity.centroids
    else:
        print("Input 'entity' must have 'vertices' or 'centroids'.")
        return None

    if isinstance(entity, BlockModel):
        values = values.reshape(
            (entity.shape[2], entity.shape[0], entity.shape[1]), order="F"
        ).transpose((1, 2, 0))
        grid = [
            entity.u_cell_delimiters,
            entity.v_cell_delimiters,
            entity.z_cell_delimiters,
        ]

    else:
        grid = []
        for ii in range(3):
            grid += [
                np.arange(
                    locations[:, ii].min(),
                    locations[:, ii].max() + resolution,
                    resolution,
                )
            ]

        y, x, z = np.meshgrid(grid[1], grid[0], grid[2])
        values = weighted_average(
            locations,
            np.c_[x.flatten(), y.flatten(), z.flatten()],
            [values],
            threshold=1e-1,
            n=8,
            max_distance=max_distance,
        )
        values = values[0].reshape(x.shape)

    surfaces = []
    for level in levels:
        try:
            verts, faces, _, _ = marching_cubes(values, level=level)

            # Remove all vertices and cells with nan
            nan_verts = np.any(np.isnan(verts), axis=1)
            rem_cells = np.any(nan_verts[faces], axis=1)

            active = np.arange(nan_verts.shape[0])
            active[nan_verts] = nan_verts.shape[0]
            _, inv_map = np.unique(active, return_inverse=True)

            verts = verts[nan_verts == False, :]
            faces = faces[rem_cells == False, :]
            faces = inv_map[faces].astype("uint32")

            vertices = []
            for ii in range(3):
                F = interp1d(
                    np.arange(grid[ii].shape[0]), grid[ii], fill_value="extrapolate"
                )
                vertices += [F(verts[:, ii])]

            if isinstance(entity, BlockModel):
                vertices = rotate_xy(np.vstack(vertices).T, [0, 0, 0], entity.rotation)
                vertices[:, 0] += entity.origin["x"]
                vertices[:, 1] += entity.origin["y"]
                vertices[:, 2] += entity.origin["z"]

            else:
                vertices = np.vstack(vertices).T
        except RuntimeError:
            vertices, faces = [], []

        surfaces += [[vertices, faces]]

    return surfaces


def get_inversion_output(h5file: str | Workspace, inversion_group: str | UUID):
    """
    Recover inversion iterations from a ContainerGroup comments.
    """
    if isinstance(h5file, Workspace):
        workspace = h5file
    else:
        workspace = Workspace(h5file)

    out = {"time": [], "iteration": [], "phi_d": [], "phi_m": [], "beta": []}

    try:
        group = workspace.get_entity(inversion_group)[0]

        for comment in group.comments.values:
            if "Iteration" in comment["Author"]:
                out["iteration"] += [np.int(comment["Author"].split("_")[1])]
                out["time"] += [comment["Date"]]
                values = json.loads(comment["Text"])
                out["phi_d"] += [float(values["phi_d"])]
                out["phi_m"] += [float(values["phi_m"])]
                out["beta"] += [float(values["beta"])]

        if len(out["iteration"]) > 0:
            out["iteration"] = np.hstack(out["iteration"])
            ind = np.argsort(out["iteration"])
            out["iteration"] = out["iteration"][ind]
            out["phi_d"] = np.hstack(out["phi_d"])[ind]
            out["phi_m"] = np.hstack(out["phi_m"])[ind]
            out["time"] = np.hstack(out["time"])[ind]

            return out
    except IndexError:
        raise IndexError(
            f"BaseInversion group {inversion_group} could not be found in the target geoh5 {h5file}"
        )


def load_json_params(file: str):
    """
    Read input parameters from json
    """
    with open(file) as f:
        input_dict = json.load(f)

    params = {}
    for key, param in input_dict.items():
        if isinstance(param, dict):
            params[key] = param["value"]
        else:
            params[key] = param

    return params


def direct_current_from_simpeg(
    workspace: Workspace, survey: Survey, name: str = None, data: dict = None
):
    """
    Convert a inversion direct-current survey to geoh5 format.
    """
    u_src_poles, src_pole_id = np.unique(
        np.r_[survey.locations_a, survey.locations_b], axis=0, return_inverse=True
    )
    n_src = int(src_pole_id.shape[0] / 2.0)
    u_src_cells, src_id = np.unique(
        np.c_[src_pole_id[:n_src], src_pole_id[n_src:]], axis=0, return_inverse=True
    )
    u_rcv_poles, rcv_pole_id = np.unique(
        np.r_[survey.locations_m, survey.locations_n], axis=0, return_inverse=True
    )
    n_rcv = int(rcv_pole_id.shape[0] / 2.0)
    u_rcv_cells = np.c_[rcv_pole_id[:n_rcv], rcv_pole_id[n_rcv:]]
    currents = CurrentElectrode.create(
        workspace, name=name, vertices=u_src_poles, cells=u_src_cells.astype("uint32")
    )
    currents.add_default_ab_cell_id()

    potentials = PotentialElectrode.create(
        workspace, name=name, vertices=u_rcv_poles, cells=u_rcv_cells.astype("uint32")
    )
    potentials.current_electrodes = currents
    potentials.ab_cell_id = np.asarray(src_id + 1, dtype="int32")

    if data is not None:
        potentials.add_data({key: {"values": value} for key, value in data.items()})

    return currents, potentials


def active_from_xyz(
    mesh, xyz, grid_reference="cell_centers", method="linear", logical="all"
):
    """Returns an active cell index array below a surface

    **** ADAPTED FROM discretize.utils.mesh_utils.active_from_xyz ****


    """
    if method == "linear":
        tri2D = Delaunay(xyz[:, :2])
        z_interpolate = LinearNDInterpolator(tri2D, xyz[:, 2])
    else:
        z_interpolate = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])

    if grid_reference == "cell_centers":
        # this should work for all 4 mesh types...
        locations = mesh.gridCC

    elif grid_reference == "top_nodes":
        locations = np.vstack(
            [
                mesh.gridCC
                + (np.c_[-1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[-1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
            ]
        )
    elif grid_reference == "bottom_nodes":
        locations = np.vstack(
            [
                mesh.gridCC
                + (np.c_[-1, 1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[-1, -1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, 1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                mesh.gridCC
                + (np.c_[1, -1, -1][:, None] * mesh.h_gridded / 2.0).squeeze(),
            ]
        )

    # Interpolate z values on CC or N
    z_xyz = z_interpolate(locations[:, :-1]).squeeze()

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_xyz)
    if any(ind_nan):
        tree = cKDTree(xyz)
        _, ind = tree.query(locations[ind_nan, :])
        z_xyz[ind_nan] = xyz[ind, -1]

    # Create an active bool of all True
    active = getattr(np, logical)(
        (locations[:, -1] < z_xyz).reshape((mesh.nC, -1), order="F"), axis=1
    )

    return active.ravel()


colors = [
    "#000000",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
]
