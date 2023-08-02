#  Copyright (c) 2023 Mira Geoscience Ltd.
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
import uuid

import numpy as np
from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.shared.utils import is_uuid

from geoapps.inversion.utils import calculate_2D_trend
from geoapps.shared_utils.utils import filter_xy, get_locations


def window_data(
    data_object,
    components,
    data_dict,
    workspace,
    window_azimuth,
    window_center_x,
    window_center_y,
    window_width,
    window_height,
    mesh,
    resolution,
) -> (np.ndarray, dict, np.ndarray):
    """
    Get locations and mask for detrending data.

    :param workspace: New workspace.
    :param param_dict: Dictionary of params to give to _run_params.

    :return locations: Data object locations.
    :return mask: Mask for windowing data.
    """
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
    # Get angle
    angle = None
    if mesh is not None:
        if hasattr(mesh, "rotation"):
            angle = -1 * mesh.rotation
    # Get mask
    mask = filter_xy(
        locations[:, 0],
        locations[:, 1],
        window=window,
        angle=angle,
        distance=resolution,
    )

    new_data_object = data_object.copy(
        parent=None,
        copy_children=True,
        clear_cache=False,
        mask=mask,
        name=data_object.name + "_processed",
    )

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
    detrend_type,
    detrend_order,
    components,
    data_dict,
    locations,
):
    """
    Detrend data and update data values in param_dict.

    :param param_dict: Dictionary of params to create self._run_params.
    :param workspace: Output workspace.
    :param detrend_order: Order of the polynomial to be used.
    :param detrend_type: Method to be used for the detrending.
        "all": Use all points.
        "perimeter": Only use points on the convex hull .

    :return: Updated param_dict with updated data.
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
    ignore_values,
    forward_only,
    components,
    data_dict,
) -> np.ndarray:
    """
    Use ignore_value ignore_type to set uncertainties to infinity.
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


def parse_ignore_values(ignore_values, forward_only) -> tuple[float, str]:
    """
    Returns an ignore value and type ('<', '>', or '=') from params data.
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


def get_data_dict(workspace, param_dict):
    """ """
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
    param_dict,
    data_object,
    resolution,
    window_center_x,
    window_center_y,
    window_width,
    window_height,
    window_azimuth=None,
    ignore_values=None,
    detrend_type=None,
    detrend_order=None,
    components=None,
    data_dict=None,
):
    """ """
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
        param_dict["mesh"],
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
    if "receivers_radar_drape" in param_dict:
        radar = param_dict["receivers_radar_drape"]
        if radar is not None:
            update_dict["receivers_radar_drape"] = new_data_object.get_entity(
                radar.name
            )[0].uid

    return update_dict
