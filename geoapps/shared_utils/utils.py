#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import re
from uuid import UUID

import numpy as np
from discretize import TensorMesh, TreeMesh
from geoh5py.objects import DrapeModel
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
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


def cell_centers_to_faces(centers: np.ndarray) -> np.ndarray:
    """
    Compute faces from centers of cells containing a evenly spaced core region.

    :param: centers: Array of grid centers in one dimension.
    """

    h = np.diff(centers)
    icore = np.where(np.isclose(h, h.min()))[0]
    icore = np.append(icore, icore[-1] + 1)  # index of core hs to core centers
    faces = np.hstack([centers[icore] - h.min() / 2, centers[icore][-1] + h.min() / 2])

    # Don't assume symmetric padding loop over each extremity individually
    right_pad_inds = np.arange(icore[-1] + 1, len(centers))
    for i in right_pad_inds:
        faces = np.hstack([faces, 2 * centers[i] - faces[-1]])

    left_pad_inds = np.arange(icore[0])[::-1]
    for i in left_pad_inds:
        faces = np.hstack([2 * centers[i] - faces[0], faces])

    return faces


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
    x = cell_centers_to_faces(x)
    dx = np.diff(np.unique(x))
    h = [np.diff(x), np.diff(z)]
    origin = [-dx[: np.argmin(dx.round(6))].sum(), layers[:, 2].min()]
    mesh = TensorMesh(h, origin)

    if return_sorting:
        sorting = np.arange(mesh.n_cells)
        sorting = sorting.reshape(mesh.nCy, mesh.nCx, order="C")
        sorting = sorting[::-1].T.flatten()
        return (mesh, sorting)
    else:
        return mesh


def octree_2_treemesh(mesh):
    """
    Convert a geoh5 octree mesh to discretize.TreeMesh
    Modified code from module discretize.TreeMesh.readUBC function.
    """
    tsw_corner = np.asarray(mesh.origin.tolist())
    small_cell = [mesh.u_cell_size, mesh.v_cell_size, mesh.w_cell_size]
    n_cell_dim = [mesh.u_count, mesh.v_count, mesh.w_count]
    cell_sizes = [np.ones(nr) * sz for nr, sz in zip(n_cell_dim, small_cell)]
    u_shift, v_shift, w_shift = (np.sum(h[h < 0]) for h in cell_sizes)
    h1, h2, h3 = (np.abs(h) for h in cell_sizes)
    x0 = tsw_corner + np.array([u_shift, v_shift, w_shift])
    ls = np.log2(n_cell_dim).astype(int)

    if ls[0] == ls[1] and ls[1] == ls[2]:
        max_level = ls[0]
    else:
        max_level = min(ls) + 1

    treemesh = TreeMesh([h1, h2, h3], x0=x0)

    # Convert array_ind to points in coordinates of underlying cpp tree
    # array_ind is ix, iy, iz(top-down) need it in ix, iy, iz (bottom-up)
    cells = np.vstack(mesh.octree_cells.tolist())
    levels = cells[:, -1]
    array_ind = cells[:, :-1]
    array_ind = 2 * array_ind + levels[:, None]  # get cell center index
    array_ind[:, 2] = 2 * n_cell_dim[2] - array_ind[:, 2]  # switch direction of iz
    levels = max_level - np.log2(levels)  # calculate level

    treemesh.__setstate__((array_ind, levels))

    return treemesh


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

    # TODO use a get_entity call here once we update geoh5py entities with the method
    outfile = [c for c in group.children if c.name == "SimPEG.out"][0]
    out = [l for l in outfile.values.decode("utf-8").replace("\r", "").split("\n")][:-1]
    cols = out.pop(0).split(" ")
    out = [[string_to_numeric(k) for k in l.split(" ")] for l in out]
    out = dict(zip(cols, list(map(list, zip(*out)))))

    return out


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
