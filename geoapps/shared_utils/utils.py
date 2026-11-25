# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from geoapps_utils.utils.transformations import rotate_xyz
from geoh5py.data import FloatData
from geoh5py.objects import Curve
from geoh5py.workspace import Workspace
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from geoapps.utils.string import string_to_numeric


def hex_to_rgb(hex_color):
    """
    Convert hex color code to RGB
    """
    code = hex_color.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def window_xy(
    x: np.ndarray,
    y: np.ndarray,
    window: dict[str, float],
    mask: np.array | None = None,
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

    if ("center" in window) & ("size" in window):
        x_lim = [
            window["center"][0] - window["size"][0] / 2,
            window["center"][0] + window["size"][0] / 2,
        ]
        y_lim = [
            window["center"][1] - window["size"][1] / 2,
            window["center"][1] + window["size"][1] / 2,
        ]
    else:
        msg = "Missing window keys: 'center' and 'size'."
        raise KeyError(msg)

    window_mask = x >= x_lim[0]
    window_mask &= x <= x_lim[1]
    window_mask &= y >= y_lim[0]
    window_mask &= y <= y_lim[1]

    if mask is not None:
        window_mask &= mask

    return window_mask, x[window_mask], y[window_mask]


def downsample_xy(
    x: np.ndarray, y: np.ndarray, distance: float, mask: np.ndarray | None = None
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
    for i in range(nstn):
        if downsample_mask[mask_ind[i]]:
            ind = tree.query_ball_point(xy[i, :2], distance)
            downsample_mask[mask_ind[ind]] = False
            downsample_mask[mask_ind[i]] = True

    if mask is not None:
        downsample_mask &= mask

    xy = xy[downsample_mask]
    return downsample_mask, xy[:, 0], xy[:, 1]


def downsample_grid(
    xg: np.ndarray, yg: np.ndarray, distance: float, mask: np.ndarray | None = None
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

    def u_diff(u):
        return np.unique(np.diff(u, axis=1))[0]

    def v_diff(v):
        return np.unique(np.diff(v, axis=0))[0]

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
    distance: float | None = None,
    window: dict | None = None,
    angle: float | None = None,
    mask: np.ndarray | None = None,
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
        if "azimuth" in window:
            azim = window["azimuth"]

    is_rotated = False if (azim is None) | (azim == 0) else True
    if is_rotated:
        xy_locs = rotate_xyz(np.c_[x.ravel(), y.ravel()], window["center"], azim)
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


def get_contours(
    interval_min: float,
    interval_max: float,
    interval_spacing: float,
    fixed_contours: str | list[float] | None,
) -> list[float]:
    """
    Function to input interval contour and fixed contour information and get contours as a list of floats.

    :params interval_min: Minimum value for contour list.
    :params interval_max: Maximum value for contour list.
    :params interval_spacing: Step size for contour list.
    :params fixed_contours: List of fixed contours.
    :return : Corresponding list of values in float format.
    """

    if (
        None not in [interval_min, interval_max, interval_spacing]
        and interval_spacing != 0
    ):
        interval_contours = np.arange(
            interval_min, interval_max + interval_spacing, interval_spacing
        ).tolist()
    else:
        interval_contours = []

    if fixed_contours != "" and fixed_contours is not None:
        if type(fixed_contours) is str:
            fixed_contours = re.split(",", fixed_contours.replace(" ", ""))
            fixed_contours = [float(c) for c in fixed_contours]
        elif type(fixed_contours) is float:
            fixed_contours = [fixed_contours]
    else:
        fixed_contours = []

    contours = np.unique(np.sort(interval_contours + fixed_contours))
    return contours


def get_inversion_output(h5file: str | Workspace, inversion_group: str | UUID):
    """
    Recover inversion iterations from a ContainerGroup comments.
    """
    if isinstance(h5file, Workspace):
        workspace = h5file
    else:
        workspace = Workspace(h5file)

    try:
        group = workspace.get_entity(inversion_group)[0]
    except IndexError as exc:
        raise IndexError(
            f"BaseInversion group {inversion_group} could not be found in the target geoh5 {h5file}"
        ) from exc

    outfile = group.get_entity("SimPEG.out")[0]
    out = [
        line for line in outfile.values.decode("utf-8").replace("\r", "").split("\n")
    ][:-1]
    cols = out.pop(0).split(" ")
    out = [[string_to_numeric(k) for k in line.split(" ")] for line in out]
    out = dict(zip(cols, list(map(list, zip(*out, strict=False))), strict=False))

    return out


def densify_curve(curve: Curve, increment: float) -> np.ndarray:
    """
    Refine a curve by adding points along the curve at a given increment.

    :param curve: Curve object to be refined.
    :param increment: Distance between points along the curve.

    :return: Array of shape (n, 3) of x, y, z locations.
    """
    locations = []
    for part in curve.unique_parts:
        logic = curve.parts == part
        cells = curve.cells[np.all(logic[curve.cells], axis=1)]
        vert_ind = np.r_[cells[:, 0], cells[-1, 1]]
        locs = curve.vertices[vert_ind, :]
        locations.append(resample_locations(locs, increment))

    return np.vstack(locations)


def resample_locations(locations: np.ndarray, increment: float) -> np.ndarray:
    """
    Resample locations along a sequence of positions at a given increment.

    :param locations: Array of shape (n, 3) of x, y, z locations.
    :param increment: Minimum distance between points along the curve.

    :return: Array of shape (n, 3) of x, y, z locations.
    """
    distance = np.cumsum(
        np.r_[0, np.linalg.norm(locations[1:, :] - locations[:-1, :], axis=1)]
    )
    new_distances = np.sort(
        np.unique(np.r_[distance, np.arange(0, distance[-1], increment)])
    )

    resampled = []
    for axis in locations.T:
        interpolator = interp1d(distance, axis, kind="linear")
        resampled.append(interpolator(new_distances))

    return np.c_[resampled].T


@dataclass
class WindowOptions:
    """
    Parameters to window options for 2D data.
    """

    center_x: float
    center_y: float
    width: float
    height: float
    azimuth: float = 0.0


@dataclass
class DrapeOptions:
    """
    Parameters to change the receiver locations to drape on a topography surface.
    """

    topography_object: str
    topography: str
    z_from_topo: bool
    receivers_offset_z: float
    receivers_radar_drape: FloatData | None = None
