#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#

from __future__ import annotations

import re
from uuid import UUID

import numpy as np
from discretize import TensorMesh
from geoh5py.objects import Curve, DrapeModel
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from geoapps.utils.string import string_to_numeric
from geoapps.utils.surveys import compute_alongline_distance


def hex_to_rgb(hex_color):
    """
    Convert hex color code to RGB
    """
    code = hex_color.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def get_locations(workspace: Workspace, entity: UUID | Entity):
    """
    Returns entity's centroids or vertices.

    If no location data is found on the provided entity, the method will
    attempt to call itself on its parent.

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

        for i in range(n):
            v = value[sub][ind[:, i]] / (rad[:, i] + threshold)
            values_interp = np.nansum([values_interp, v], axis=0)
            w = 1.0 / (rad[:, i] + threshold)
            weight = np.nansum([weight, w], axis=0)

        values_interp[weight > 0] = values_interp[weight > 0] / weight[weight > 0]
        values_interp[weight == 0] = np.nan
        avg_values += [values_interp]

    if return_indices:
        return avg_values, ind

    return avg_values


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


def rotate_xyz(xyz: np.ndarray, center: list, theta: float, phi: float = 0.0):
    """
    Perform a counterclockwise rotation of scatter points around the z-axis, then x-axis, about a center point.

    :param xyz: shape(*, 2) or shape(*, 3) Input coordinates.
    :param center: len(2) or len(3) Coordinates for the center of rotation.
    :param theta: Angle of rotation around z-axis in degrees.
    :param phi: Angle of rotation around x-axis in degrees.
    """
    return2d = False
    locs = xyz.copy()

    # If the input is 2-dimensional, add zeros in the z column.
    if len(center) == 2:
        center.append(0)
    if locs.shape[1] == 2:
        locs = np.concatenate((locs, np.zeros((locs.shape[0], 1))), axis=1)
        return2d = True

    locs = np.subtract(locs, center)
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)

    # Construct rotation matrix
    Rx = np.r_[
        np.c_[1, 0, 0],
        np.c_[0, np.cos(phi), -np.sin(phi)],
        np.c_[0, np.sin(phi), np.cos(phi)],
    ]
    Rz = np.r_[
        np.c_[np.cos(theta), -np.sin(theta), 0],
        np.c_[np.sin(theta), np.cos(theta), 0],
        np.c_[0, 0, 1],
    ]
    R = Rz.dot(Rx)

    xyz_rot = R.dot(locs.T).T
    xyz_out = xyz_rot + center

    if return2d:
        # Return 2-dimensional data if the input xyz was 2-dimensional.
        return xyz_out[:, :2]
    else:
        return xyz_out


def drape_2_tensor(drape_model: DrapeModel, return_sorting: bool = False) -> tuple:
    """
    Convert a geoh5 drape model to discretize.TensorMesh.

    :param: drape_model: geoh5py.DrapeModel object.
    :param: return_sorting: If True then return an index array that would
        re-sort a model in TensorMesh order to DrapeModel order.
    """
    prisms = drape_model.prisms
    layers = drape_model.layers
    z = np.append(np.unique(layers[:, 2]), prisms[:, 2].max())
    x = compute_alongline_distance(prisms[:, :2])
    dx = np.diff(x)
    end_core = [np.argmin(dx.round(1)), len(dx) - np.argmin(dx[::-1].round(1))]
    core = dx[end_core[0]]
    exp_fact = dx[0] / dx[1]
    cell_width = np.r_[
        core * exp_fact ** np.arange(end_core[0], 0, -1),
        core * np.ones(end_core[1] - end_core[0] + 1),
        core * exp_fact ** np.arange(1, len(dx) - end_core[1] + 1),
    ]
    h = [cell_width, np.diff(z)]
    origin = [-cell_width[: end_core[0]].sum(), layers[:, 2].min()]
    mesh = TensorMesh(h, origin)

    if return_sorting:
        sorting = np.arange(mesh.n_cells)
        sorting = sorting.reshape(mesh.shape_cells[1], mesh.shape_cells[0], order="C")
        sorting = sorting[::-1].T.flatten()
        return (mesh, sorting)
    else:
        return mesh


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
    out = [l for l in outfile.values.decode("utf-8").replace("\r", "").split("\n")][:-1]
    cols = out.pop(0).split(" ")
    out = [[string_to_numeric(k) for k in l.split(" ")] for l in out]
    out = dict(zip(cols, list(map(list, zip(*out)))))

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
