#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#

from __future__ import annotations

import uuid

import numpy as np
from geoh5py import Workspace
from geoh5py.data import Data, NumericData
from geoh5py.objects import Grid2D, ObjectBase, Points, PotentialElectrode
from geoh5py.shared.utils import is_uuid

from geoapps.inversion.utils import calculate_2D_trend
from geoapps.shared_utils.utils import filter_xy, get_locations


# TODO replace with implementation in geoh5py v0.9.0
def grid_to_points(grid2d: Grid2D) -> Points:
    """"""
    points = Points.create(
        grid2d.workspace, vertices=grid2d.centroids, name=grid2d.name
    )
    for child in grid2d.children:
        if isinstance(child, NumericData):
            child.copy(parent=points, association="VERTEX")

    return points


def window_data(
    data_object: ObjectBase,
    components: list[str],
    data_dict: dict,
    workspace: Workspace,
    window_azimuth: float,
    window_center_x: float,
    window_center_y: float,
    window_width: float,
    window_height: float,
    resolution: float,
) -> (ObjectBase, dict, np.ndarray):
    """
    Window, downsample, and rotate data_object. Update data_dict with new data uids.

    :param data_object: Data object to be windowed.
    :param components: List of active data components.
    :param data_dict: Dictionary of data components and uncertainties.
    :param workspace: Output workspace.
    :param window_azimuth: Azimuth of the window.
    :param window_center_x: X center of the window.
    :param window_center_y: Y center of the window.
    :param window_width: Width of the window.
    :param window_height: Height of the window.
    :param resolution: Resolution for downsampling.

    :return: Windowed data object.
    :return: Updated data dict.
    :return: Vertices or centroids of the windowed data object.
    """

    if not isinstance(data_object, ObjectBase):
        raise TypeError(
            f"'data_object' must be an {ObjectBase}, found '{type(data_object)}' instead."
        )

    if isinstance(data_object, Grid2D):
        data_object = grid_to_points(data_object)

    # Get locations
    locations = get_locations(workspace, data_object)

    # Get window
    window = {
        "azimuth": window_azimuth,
        "center_x": window_center_x,
        "center_y": window_center_y,
        "width": window_width,
        "height": window_height,
        "center": [
            window_center_x,
            window_center_y,
        ],
        "size": [window_width, window_height],
    }

    # Get mask
    mask = filter_xy(
        locations[:, 0],
        locations[:, 1],
        window=window,
        distance=resolution,
    )

    if isinstance(data_object, PotentialElectrode):
        vert_mask = np.zeros(data_object.n_vertices, dtype=bool)
        vert_mask[data_object.cells[mask, :].ravel()] = True
        mask = vert_mask

    new_data_object = data_object.copy(
        parent=data_object.workspace,
        copy_children=True,
        clear_cache=False,
        mask=mask,
        name=data_object.name + "_processed",
    )

    if (
        not isinstance(data_object, PotentialElectrode)
        and getattr(data_object, "parts", None) is not None
    ):
        new_data_object.parts = data_object.parts[mask]

    # Update data dict
    for comp in components:
        data_dict[comp + "_channel"]["values"] = new_data_object.get_entity(
            data_dict[comp + "_channel"]["name"]
        )[0].values
        if comp + "_uncertainty" in data_dict:
            data_dict[comp + "_uncertainty"]["values"] = new_data_object.get_entity(
                data_dict[comp + "_uncertainty"]["name"]
            )[0].values

    # Get new locations
    if hasattr(new_data_object, "centroids"):
        locations = new_data_object.centroids
    elif hasattr(new_data_object, "vertices"):
        locations = new_data_object.vertices

    return new_data_object, data_dict, locations


def detrend_data(
    detrend_type: str,
    detrend_order: int,
    components: list[str],
    data_dict: dict,
    locations: np.ndarray,
) -> dict:
    """
    Detrend data in data_dict.

    :param detrend_type: Method to be used for the detrending.
        "all": Use all points.
        "perimeter": Only use points on the convex hull .
    :param detrend_order: Order of the polynomial to be used.
    :param components: List of active data components.
    :param data_dict: Dictionary of data components and uncertainties.
    :param locations: Vertices of the data object.

    :return: Updated data_dict with detrended data values.
    """
    if detrend_type == "none" or detrend_type is None or detrend_order is None:
        return data_dict

    for comp in components:
        data = data_dict[comp + "_channel"]
        # Get data trend
        values = data["values"]
        data_trend, _ = calculate_2D_trend(
            locations,
            values,
            detrend_order,
            detrend_type,
        )
        # Update data values and add to object
        data["values"] -= data_trend
    return data_dict


def set_infinity_uncertainties(
    ignore_values: str,
    forward_only: bool,
    components: list[str],
    data_dict: dict,
) -> dict:
    """
    Use ignore_value ignore_type to set uncertainties to infinity.

    :param ignore_values: Values to be set to infinity.
    :param forward_only: Forward inversion only.
    :param components: List of active data components.
    :param data_dict: Dictionary of data components and uncertainties.

    :return: Updated data_dict with uncertainties set to infinity.
    """
    ignore_value, ignore_type = parse_ignore_values(ignore_values, forward_only)

    for comp in components:
        if comp + "_uncertainty" not in data_dict:
            continue
        data = data_dict[comp + "_channel"]["values"]
        uncertainty = data_dict[comp + "_uncertainty"]["values"]
        uncertainty[np.isnan(data)] = np.inf

        if ignore_value is None:
            continue
        elif ignore_type == "<":
            uncertainty[data <= ignore_value] = np.inf
        elif ignore_type == ">":
            uncertainty[data >= ignore_value] = np.inf
        elif ignore_type == "=":
            uncertainty[data == ignore_value] = np.inf
        else:
            msg = f"Unrecognized ignore type: {ignore_type}."
            raise (ValueError(msg))

        data_dict[comp + "_uncertainty"]["values"] = uncertainty

    return data_dict


def parse_ignore_values(ignore_values: str, forward_only: bool) -> tuple[float, str]:
    """
    Returns an ignore value and type ('<', '>', or '=') from params data.

    :param ignore_values: Values to be ignored.
    :param forward_only: Forward inversion only.

    :return: Float value to be ignored.
    :return: Ignore type ('<', '>', or '=').
    """
    if forward_only:
        return None, None

    if ignore_values is None:
        return None, None
    ignore_type = [k for k in ignore_values if k in ["<", ">"]]
    ignore_type = "=" if not ignore_type else ignore_type[0]
    if ignore_type in ["<", ">"]:
        ignore_value = float(ignore_values.split(ignore_type)[1])
    else:
        try:
            ignore_value = float(ignore_values)
        except ValueError:
            return None, None

    return ignore_value, ignore_type


def get_data_dict(workspace: Workspace, param_dict: dict) -> (list[str], dict):
    """
    Get dictionary of active components from param_dict.

    :param workspace: Workspace that the data belong to.
    :param param_dict: Dictionary of params to run the inversion.

    :return: List of active components.
    :return: Dictionary of active channels and uncertainties.
    """
    # Get components
    components = []
    data_dict = {}
    for key, value in param_dict.items():
        if key.endswith("_channel_bool") and value:
            comp = key.replace("_channel_bool", "")
            components.append(comp)

            # Add data to data dict
            data = param_dict[comp + "_channel"]
            data_dict[comp + "_channel"] = {
                "name": data.name,
            }

            # Add uncertainties to data dict
            if comp + "_uncertainty" in param_dict:
                unc = param_dict[comp + "_uncertainty"]
                if isinstance(unc, Data):
                    data_dict[comp + "_uncertainty"] = {
                        "name": param_dict[comp + "_uncertainty"].name,
                    }
                elif is_uuid(unc):
                    data_dict[comp + "_uncertainty"] = {
                        "name": workspace.get_entity(uuid.UUID(unc))[0].name,
                    }

    return components, data_dict


def preprocess_data(
    workspace: Workspace,
    param_dict: dict,
    data_object: ObjectBase,
    resolution: float | None,
    window_center_x: float,
    window_center_y: float,
    window_width: float,
    window_height: float,
    window_azimuth: float | None = None,
    ignore_values: str | None = None,
    detrend_type: str | None = None,
    detrend_order: int | None = None,
    components: list | None = None,
    data_dict: dict | None = None,
) -> dict:
    """
    Window, detrend, and ignore values in data_object. Update data_dict with new data uids.

    :param workspace: Parent workspace for data_object and data components.
    :param param_dict: Dictionary of params to run the inversion.
    :param data_object: Parent object for data components.
    :param resolution: Resolution for downsampling.
    :param window_center_x: X center of the window.
    :param window_center_y: Y center of the window.
    :param window_width: Width of the window.
    :param window_height: Height of the window.
    :param window_azimuth: Azimuth of the window.
    :param ignore_values: Values to be ignored.
    :param detrend_type: Method to be used for the detrending.
    :param detrend_order: Order of the polynomial to be used for detrend.
    :param components: List of active data components.
    :param data_dict: Dictionary of data components and uncertainties.

    :return: Updated data_dict with processed data.
    """
    if data_dict is None:
        components, data_dict = get_data_dict(workspace, param_dict)

    # Windowing
    new_data_object, data_dict, locations = window_data(
        data_object,
        components,
        data_dict,
        workspace,
        window_azimuth,
        window_center_x,
        window_center_y,
        window_width,
        window_height,
        resolution,
    )

    # Ignore values
    if ignore_values is not None:
        data_dict = set_infinity_uncertainties(
            ignore_values,
            param_dict["forward_only"],
            components,
            data_dict,
        )
    # Detrending
    if detrend_type is not None and detrend_order is not None:
        data_dict = detrend_data(
            detrend_type,
            detrend_order,
            components,
            data_dict,
            locations,
        )

    # Add processed data to data object
    update_dict = {}
    update_dict["data_object"] = new_data_object.uid
    for comp in components:
        for key in [comp + "_channel", comp + "_uncertainty"]:
            if key not in data_dict.keys():
                continue
            data = data_dict[key]
            if key in data_dict.keys():
                update_dict[key] = new_data_object.get_entity(data["name"])[0].uid
    # Update update_dict with new radar uid
    if "receivers_radar_drape" in param_dict:
        radar = param_dict["receivers_radar_drape"]
        if radar is not None:
            update_dict["receivers_radar_drape"] = new_data_object.get_entity(
                radar.name
            )[0].uid

    return update_dict
