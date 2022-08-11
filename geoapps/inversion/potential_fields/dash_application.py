#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os
import uuid
import warnings
from copy import copy
from time import time

import numpy as np
from dash import Input, Output, State, callback_context, no_update
from flask import Flask
from geoh5py.data import ReferencedData
from geoh5py.objects import BlockModel, Curve, Grid2D, Octree, Points, Surface
from geoh5py.shared import Entity
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash
from matplotlib import colors
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly.tools import mpl_to_plotly

from geoapps.base.application import BaseApplication
from geoapps.base.dash_application import BaseDashApplication
from geoapps.base.plot import PlotSelection2D
from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.inversion.potential_fields.gravity.params import GravityParams
from geoapps.inversion.potential_fields.layout import potential_fields_layout
from geoapps.inversion.potential_fields.magnetic_scalar.params import (
    MagneticScalarParams,
)
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.inversion.potential_fields.magnetic_vector.params import (
    MagneticVectorParams,
)
from geoapps.shared_utils.utils import filter_xy, rotate_xyz
from geoapps.utils import geophysical_systems, warn_module_not_found
from geoapps.utils.list import find_value
from geoapps.utils.plotting import format_labels, plot_plan_data_selection
from geoapps.utils.string import string_2_list


class InversionApp(BaseDashApplication):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = MagneticVectorParams
    _select_multiple = True
    _add_groups = False
    _run_params = None
    _sensor = None
    _topography = None
    inversion_parameters = None
    defaults = {}

    def __init__(self, ui_json=None, **kwargs):
        app_initializer.update(kwargs)
        if ui_json is not None and os.path.exists(ui_json.path):
            self.params = self._param_class(ui_json)
        else:
            self.params = self._param_class(**app_initializer)

        super().__init__(**self.params.to_dict())

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

        self.app.layout = potential_fields_layout

        # Callbacks relating to layout
        self.app.callback(
            Output(component_id="inducing_params_div", component_property="style"),
            Input(component_id="inversion_type", component_property="value"),
        )(InversionApp.update_inducing_params_visibility)
        self.app.callback(
            Output(component_id="starting_magnetic_vector", component_property="style"),
            Output(
                component_id="reference_magnetic_vector", component_property="style"
            ),
            Output(component_id="starting_magnetic_scalar", component_property="style"),
            Output(
                component_id="reference_magnetic_scalar", component_property="style"
            ),
            Output(component_id="starting_gravity", component_property="style"),
            Output(component_id="reference_gravity", component_property="style"),
            Input(component_id="inversion_type", component_property="value"),
        )(InversionApp.update_inversion_div_visibility)
        self.app.callback(
            Output(component_id="uncertainty_floor", component_property="style"),
            Output(component_id="uncertainty_channel", component_property="style"),
            Input(component_id="uncertainty_options", component_property="value"),
        )(InversionApp.update_uncertainty_visibility)
        self.app.callback(
            Output(component_id="topography_none_div", component_property="style"),
            Output(component_id="topography_object_div", component_property="style"),
            Output(component_id="topography_sensor_div", component_property="style"),
            Output(component_id="topography_constant_div", component_property="style"),
            Input(component_id="topography_options", component_property="value"),
        )(InversionApp.update_topography_visibility)
        for model_type in ["starting", "reference"]:
            for param in [
                "eff_susceptibility",
                "inclination",
                "declination",
                "susceptibility",
                "density",
            ]:
                self.app.callback(
                    Output(
                        component_id=model_type + "_" + param + "_const_div",
                        component_property="style",
                    ),
                    Output(
                        component_id=model_type + "_" + param + "_mod_div",
                        component_property="style",
                    ),
                    Input(
                        component_id=model_type + "_" + param + "_options",
                        component_property="value",
                    ),
                )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="lower_bound_const_div", component_property="style"),
            Output(component_id="lower_bound_mod_div", component_property="style"),
            Input(component_id="lower_bound_options", component_property="value"),
        )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="upper_bound_const_div", component_property="style"),
            Output(component_id="upper_bound_mod_div", component_property="style"),
            Input(component_id="upper_bound_options", component_property="value"),
        )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="starting_model_div", component_property="style"),
            Output(component_id="mesh_div", component_property="style"),
            Output(component_id="reference_model_div", component_property="style"),
            Output(component_id="regularization_div", component_property="style"),
            Output(component_id="upper_lower_bound_div", component_property="style"),
            Output(component_id="detrend_div", component_property="style"),
            Output(component_id="ignore_values_div", component_property="style"),
            Output(component_id="optimization_div", component_property="style"),
            Input(component_id="param_dropdown", component_property="value"),
        )(InversionApp.update_inversion_params_visibility)

        # Update component options from changing inversion type
        self.app.callback(
            Output(component_id="component", component_property="options"),
            Input(component_id="inversion_type", component_property="value"),
        )(self.update_component_list)

        # Update object and data dropdowns
        self.app.callback(
            Output(component_id="data_object", component_property="options"),
            Output(component_id="data_object", component_property="value"),
            Output(component_id="ui_json", component_property="data"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="channel", component_property="options"),
            # Output(component_id="channel", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_channel_options)
        self.app.callback(
            Output(component_id="uncertainty_channel", component_property="options"),
            # Output(component_id="uncertainty_channel", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_channel_options)
        self.app.callback(
            Output(component_id="receivers_radar_drape", component_property="options"),
            Output(component_id="receivers_radar_drape", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="data_object", component_property="value"),
        )(self.update_data_options)

        # Update input data channel and uncertainties from component
        self.app.callback(
            Output(component_id="full_components", component_property="data"),
            Input(component_id="ui_json", component_property="data"),
            Input(component_id="full_components", component_property="data"),
            Input(component_id="channel_bool", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="uncertainty_options", component_property="value"),
            Input(component_id="uncertainty_floor", component_property="value"),
            Input(component_id="uncertainty_channel", component_property="value"),
            State(component_id="component", component_property="value"),
            State(component_id="component", component_property="options"),
        )(self.update_full_components)
        self.app.callback(
            Output(component_id="channel_bool", component_property="value"),
            Output(component_id="channel", component_property="value"),
            Output(component_id="uncertainty_options", component_property="value"),
            Output(component_id="uncertainty_floor", component_property="value"),
            Output(component_id="uncertainty_channel", component_property="value"),
            Input(component_id="component", component_property="value"),
            Input(component_id="full_components", component_property="data"),
        )(self.update_input_channel)

        # Update from ui.json
        self.app.callback(
            # Object selection
            Output(component_id="inversion_type", component_property="value"),
            Output(component_id="inducing_field_strength", component_property="value"),
            Output(
                component_id="inducing_field_inclination", component_property="value"
            ),
            Output(
                component_id="inducing_field_declination", component_property="value"
            ),
            # Input Data
            Output(component_id="resolution", component_property="value"),
            # Topography
            Output(component_id="z_from_topo", component_property="value"),
            Output(component_id="receivers_offset_x", component_property="value"),
            Output(component_id="receivers_offset_y", component_property="value"),
            Output(component_id="receivers_offset_z", component_property="value"),
            Output(component_id="topography_offset", component_property="value"),
            Output(component_id="topography_constant", component_property="value"),
            # Inversion - mesh
            Output(component_id="u_cell_size", component_property="value"),
            Output(component_id="v_cell_size", component_property="value"),
            Output(component_id="w_cell_size", component_property="value"),
            Output(component_id="octree_levels_topo", component_property="value"),
            Output(component_id="octree_levels_obs", component_property="value"),
            Output(component_id="max_distance", component_property="value"),
            Output(component_id="horizontal_padding", component_property="value"),
            Output(component_id="vertical_padding", component_property="value"),
            Output(component_id="depth_core", component_property="value"),
            # Inversion - regularization
            Output(component_id="alpha_s", component_property="value"),
            Output(component_id="alpha_x", component_property="value"),
            Output(component_id="alpha_y", component_property="value"),
            Output(component_id="alpha_z", component_property="value"),
            Output(component_id="s_norm", component_property="value"),
            Output(component_id="x_norm", component_property="value"),
            Output(component_id="y_norm", component_property="value"),
            Output(component_id="z_norm", component_property="value"),
            # Inversion - upper-lower bounds
            # Output(component_id="lower_bound", component_property="value"),
            # Output(component_id="upper_bound", component_property="value"),
            # Inversion - detrend
            Output(component_id="detrend_type", component_property="value"),
            Output(component_id="detrend_order", component_property="value"),
            # Inversion - ignore values
            Output(component_id="ignore_values", component_property="value"),
            # Inversion - optimization
            Output(component_id="max_iterations", component_property="value"),
            Output(component_id="chi_factor", component_property="value"),
            Output(component_id="initial_beta_ratio", component_property="value"),
            Output(component_id="max_cg_iterations", component_property="value"),
            Output(component_id="tol_cg", component_property="value"),
            Output(component_id="n_cpu", component_property="value"),
            Output(component_id="tile_spatial", component_property="value"),
            # Output
            Output(component_id="ga_group_name", component_property="value"),
            Input(component_id="ui_json", component_property="data"),
        )(self.update_remainder_from_ui_json)

        # Plot callbacks
        self.app.callback(
            Output(component_id="window_center_x", component_property="min"),
            Output(component_id="window_center_x", component_property="max"),
            Output(component_id="window_center_x", component_property="value"),
            Output(component_id="window_center_y", component_property="min"),
            Output(component_id="window_center_y", component_property="max"),
            Output(component_id="window_center_y", component_property="value"),
            Output(component_id="window_width", component_property="min"),
            Output(component_id="window_width", component_property="max"),
            Output(component_id="window_width", component_property="value"),
            Output(component_id="window_height", component_property="min"),
            Output(component_id="window_height", component_property="max"),
            Output(component_id="window_height", component_property="value"),
            Input(component_id="data_object", component_property="value"),
        )(self.set_bounding_box)
        # Update plot
        self.app.callback(
            Output(component_id="plot", component_property="figure"),
            Output(component_id="data_count", component_property="children"),
            Input(component_id="data_object", component_property="value"),
            Input(component_id="channel", component_property="value"),
            Input(component_id="resolution", component_property="value"),
            Input(component_id="window_center_x", component_property="value"),
            Input(component_id="window_center_y", component_property="value"),
            Input(component_id="window_width", component_property="value"),
            Input(component_id="window_height", component_property="value"),
            Input(component_id="window_azimuth", component_property="value"),
            Input(component_id="zoom_extent", component_property="value"),
            Input(component_id="colorbar", component_property="value"),
        )(self.plot_selection)

        """
        # Update mesh
        self.app.callback(
            Input(component_id="window_width", component_property="value"),
            Input(component_id="window_height", component_property="value"),
            Input(component_id="resolution", component_property="value"),
        )(self.update_octree_param)

        # Callbacks for clicking buttons
        self.app.callback(
            Input(component_id="write_input", component_property="n_clicks"),
        )(self.write_trigger)
        self.app.callback(
            Input(component_id="compute", component_property="n_clicks"),
        )(self.trigger_click)
        """

    @staticmethod
    def update_inducing_params_visibility(selection):
        if selection in ["magnetic vector", "magnetic scalar"]:
            return {"display": "inline-block"}
        else:
            return {"display": "none"}

    @staticmethod
    def update_inversion_div_visibility(selection):
        if selection == "magnetic vector":
            return (
                {"display": "block"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "magnetic scalar":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "gravity":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "block"},
            )

    @staticmethod
    def update_uncertainty_visibility(selection):
        if selection == "Floor":
            return (
                {"display": "block"},
                {"display": "none"},
            )
        elif selection == "Channel":
            return (
                {"display": "none"},
                {"display": "block"},
            )

    @staticmethod
    def update_topography_visibility(selection):
        if selection == "None":
            return (
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "Object":
            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "Relative to Sensor":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
            )
        elif selection == "Constant":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
            )

    @staticmethod
    def update_model_visibility(selection):
        if selection == "Constant":
            return {"display": "block"}, {"display": "none"}
        elif selection == "Model":
            return {"display": "none"}, {"display": "block"}
        elif selection == "None":
            return {"display": "none"}, {"display": "none"}

    @staticmethod
    def update_inversion_params_visibility(selection):
        if selection == "starting model":
            return (
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "mesh":
            return (
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "reference model":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "regularization":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "upper-lower bounds":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "detrend":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
                {"display": "none"},
            )
        elif selection == "ignore values":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
                {"display": "none"},
            )
        elif selection == "optimization":
            return (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "inline-block"},
            )

    @staticmethod
    def update_component_list(inversion_type):
        if inversion_type in ["magnetic vector", "magnetic scalar"]:
            data_type_list = [
                "tmi",
                "bx",
                "by",
                "bz",
                "bxx",
                "bxy",
                "bxz",
                "byy",
                "byz",
                "bzz",
            ]
        else:
            data_type_list = [
                "gx",
                "gy",
                "gz",
                "gxx",
                "gxy",
                "gxz",
                "gyy",
                "gyz",
                "gzz",
                "uv",
            ]
        return data_type_list

    # Update input data dropdown options

    def update_channel_options(self, ui_json: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json: Uploaded ui.json.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json" in triggers:
            # value = ui_json["channel"]["value"]
            value = None
            options = self.get_data_options("ui_json", ui_json, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json, object_uid)

        return options

    def update_data_options(self, ui_json: dict, object_uid: str) -> (list, str):
        """
        Update data subset options and values from selected object.

        :param ui_json: Uploaded ui.json.
        :param object_uid: Selected object from dropdown.

        :return options: Options for data subset dropdown.
        :return value: Value for data subset dropdown.
        """
        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]

        if "ui_json" in triggers:
            # value = ui_json["channel"]["value"]
            value = None
            options = self.get_data_options("ui_json", ui_json, object_uid)
        else:
            value = None
            options = self.get_data_options("data_object", ui_json, object_uid)

        return options, value

    @staticmethod
    def update_full_components(
        ui_json,
        full_components,
        channel_bool,
        channel,
        uncertainty_type,
        uncertainty_floor,
        uncertainty_channel,
        component,
        component_options,
    ):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json":
            full_components = {}
            for comp in component_options:
                # Get channel value
                if is_uuid(ui_json[comp + "_channel"]["value"]):
                    channel = str(ui_json[comp + "_channel"]["value"])
                else:
                    channel = None
                # Get channel_bool value
                if ui_json[comp + "_channel_bool"]:
                    channel_bool = [True]
                else:
                    channel_bool = []
                # Get uncertainty value
                if (type(ui_json[comp + "_uncertainty"]["value"]) == float) or (
                    type(ui_json[comp + "_uncertainty"]["value"]) == int
                ):
                    uncertainty_type = "Floor"
                    uncertainty_floor = ui_json[comp + "_uncertainty"]["value"]
                    uncertainty_channel = None
                elif is_uuid(ui_json[comp + "_uncertainty"]["value"]):
                    uncertainty_type = "Channel"
                    uncertainty_floor = None
                    uncertainty_channel = str(ui_json[comp + "_uncertainty"]["value"])

                full_components[comp] = {
                    "channel_bool": channel_bool,
                    "channel": channel,
                    "uncertainty_type": uncertainty_type,
                    "uncertainty_floor": uncertainty_floor,
                    "uncertainty_channel": uncertainty_channel,
                }
        else:
            full_components[component] = {
                "channel_bool": channel_bool,
                "channel": channel,
                "uncertainty_type": uncertainty_type,
                "uncertainty_floor": uncertainty_floor,
                "uncertainty_channel": uncertainty_channel,
            }
        return full_components

    @staticmethod
    def update_input_channel(component, full_components):
        if full_components and component is not None:
            channel_bool = full_components[component]["channel_bool"]
            channel = full_components[component]["channel"]
            uncertainty_type = full_components[component]["uncertainty_type"]
            uncertainty_floor = full_components[component]["uncertainty_floor"]
            uncertainty_channel = full_components[component]["uncertainty_channel"]
            return (
                channel_bool,
                channel,
                uncertainty_type,
                uncertainty_floor,
                uncertainty_channel,
            )
        else:
            return no_update, no_update, no_update, no_update, no_update

    def format_labels(
        self, x, y, axs, labels=None, aspect="equal", tick_format="%i", **kwargs
    ):
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

    def plot_plan_data_selection(self, entity, data, **kwargs):
        """
        Plot data values in 2D with contours

        :param entity: `geoh5py.objects`
            Input object with either `vertices` or `centroids` property.
        :param data: `geoh5py.data`
            Input data with `values` property.

        :return ax:
        :return out:
        :return indices:
        :return line_selection:
        :return contour_set:
        """
        indices = None
        line_selection = None
        contour_set = None
        values = None
        figure = None
        out = None

        print(1)
        if isinstance(entity, (Grid2D, Points, Curve, Surface)):
            if "figure" not in kwargs.keys():
                figure = go.Figure()
            else:
                figure = kwargs["figure"]
        else:
            return figure, out, indices, line_selection, contour_set
        print(2)
        # for collection in axis.collections:
        #     collection.remove()

        if getattr(entity, "vertices", None) is not None:
            locations = entity.vertices
        else:
            locations = entity.centroids

        if "resolution" not in kwargs.keys():
            resolution = 0
        else:
            resolution = kwargs["resolution"]
        print(3)
        if "indices" in kwargs.keys():
            indices = kwargs["indices"]
            if isinstance(indices, np.ndarray) and np.all(indices == False):
                indices = None

        if isinstance(getattr(data, "values", None), np.ndarray) and not isinstance(
            data.values[0], str
        ):
            values = np.asarray(data.values, dtype=float).copy()
            values[values == -99999] = np.nan
        elif isinstance(data, str) and (data in "XYZ"):
            values = locations[:, "XYZ".index(data)]
        print(4)
        if values is not None and (values.shape[0] != locations.shape[0]):
            values = None

        color_norm = None
        if "color_norm" in kwargs.keys():
            color_norm = kwargs["color_norm"]

        window = None
        if "window" in kwargs.keys():
            window = kwargs["window"]
        print(5)
        if (
            data is not None
            and getattr(data, "entity_type", None) is not None
            and getattr(data.entity_type, "color_map", None) is not None
        ):
            new_cmap = data.entity_type.color_map._values
            map_vals = new_cmap["Value"].copy()
            cmap = colors.ListedColormap(
                np.c_[
                    new_cmap["Red"] / 255,
                    new_cmap["Green"] / 255,
                    new_cmap["Blue"] / 255,
                ]
            )
            color_norm = colors.BoundaryNorm(map_vals, cmap.N)
        else:
            cmap = "Spectral_r"
        print(6)
        if isinstance(entity, Grid2D):
            x = entity.centroids[:, 0].reshape(entity.shape, order="F")
            y = entity.centroids[:, 1].reshape(entity.shape, order="F")
            indices = filter_xy(x, y, resolution, window=window)

            ind_x, ind_y = (
                np.any(indices, axis=1),
                np.any(indices, axis=0),
            )

            X = x[ind_x, :][:, ind_y]
            Y = y[ind_x, :][:, ind_y]

            if values is not None:
                values = np.asarray(
                    values.reshape(entity.shape, order="F"), dtype=float
                )
                values[indices == False] = np.nan
                values = values[ind_x, :][:, ind_y]

            if np.any(values):
                # out = axis.pcolormesh(
                #    X, Y, values, cmap=cmap, norm=color_norm, shading="auto"
                # )
                pass
            print(7)
            if (
                "contours" in kwargs.keys()
                and kwargs["contours"] is not None
                and np.any(values)
            ):
                contour_set = axis.contour(
                    X, Y, values, levels=kwargs["contours"], colors="k", linewidths=1.0
                )
            print(8)
        else:
            x, y = entity.vertices[:, 0], entity.vertices[:, 1]
            if indices is None:
                indices = filter_xy(
                    x,
                    y,
                    resolution,
                    window=window,
                )
            X, Y = x[indices], y[indices]

            if values is not None:
                values = values[indices]

            if "marker_size" not in kwargs.keys():
                marker_size = 50
            else:
                marker_size = kwargs["marker_size"]
            print(9)
            # out = axis.scatter(X, Y, marker_size, values, cmap=cmap, norm=color_norm)

            if (
                "contours" in kwargs.keys()
                and kwargs["contours"] is not None
                and np.any(values)
            ):
                ind = ~np.isnan(values)
                contour_set = axis.tricontour(
                    X[ind],
                    Y[ind],
                    values[ind],
                    levels=kwargs["contours"],
                    colors="k",
                    linewidths=1.0,
                )
        print(10)
        if "collections" in kwargs.keys():
            for collection in kwargs["collections"]:
                axis.add_collection(copy(collection))

        if "zoom_extent" in kwargs.keys() and kwargs["zoom_extent"] and np.any(values):
            ind = ~np.isnan(values.ravel())
            x = X.ravel()[ind]
            y = Y.ravel()[ind]

        if np.any(x) and np.any(y):
            width = x.max() - x.min()
            height = y.max() - y.min()

            # format_labels(x, y, axis, **kwargs)
            figure.update_layout(
                xaxis_range=[x.min() - width * 0.1, x.max() + width * 0.1],
                yaxis_range=[y.min() - height * 0.1, y.max() + height * 0.1],
            )
        print(11)
        if "colorbar" in kwargs.keys() and kwargs["colorbar"]:
            # plt.colorbar(out, ax=axis)
            pass

        line_selection = np.zeros_like(indices, dtype=bool)
        if "highlight_selection" in kwargs.keys() and isinstance(
            kwargs["highlight_selection"], dict
        ):
            for key, values in kwargs["highlight_selection"].items():

                if not np.any(entity.workspace.get_entity(key)):
                    continue

                line_data = entity.workspace.get_entity(key)[0]
                if isinstance(line_data, ReferencedData):
                    values = [
                        key
                        for key, value in line_data.value_map.map.items()
                        if value in values
                    ]

                for line in values:
                    ind = np.where(line_data.values == line)[0]
                    x, y, values = (
                        locations[ind, 0],
                        locations[ind, 1],
                        entity.workspace.get_entity(key)[0].values[ind],
                    )
                    ind_line = filter_xy(x, y, resolution, window=window)
                    axis.scatter(
                        x[ind_line], y[ind_line], marker_size * 2, "k", marker="+"
                    )
                    line_selection[ind[ind_line]] = True
        print(12)
        return figure, out, indices, line_selection, contour_set

    def plot_selection(
        self,
        object,
        data,
        resolution,
        center_x,
        center_y,
        width,
        height,
        azimuth,
        zoom_extent,
        colorbar,
    ):
        if object is not None and data is not None:
            obj = self.workspace.get_entity(uuid.UUID(object))[0]

            data_obj = self.workspace.get_entity(uuid.UUID(data))[0]

            if isinstance(obj, (Grid2D, Surface, Points, Curve)):
                figure = go.Figure()
                corners = np.r_[
                    np.c_[-1.0, -1.0],
                    np.c_[-1.0, 1.0],
                    np.c_[1.0, 1.0],
                    np.c_[1.0, -1.0],
                    np.c_[-1.0, -1.0],
                ]
                corners[:, 0] *= width / 2
                corners[:, 1] *= height / 2
                corners = rotate_xyz(corners, [0, 0], -azimuth)
                figure.add_trace(
                    go.Scatter(
                        x=corners[:, 0] + center_x,
                        y=corners[:, 1] + center_y,
                        mode="lines",
                        marker_color="black",
                    )
                )
                print(figure)
                figure, _, ind_filter, _, _ = self.plot_plan_data_selection(
                    obj,
                    data_obj,
                    **{
                        "figure": figure,
                        "resolution": resolution,
                        "window": {
                            "center": [center_x, center_y],
                            "size": [width, height],
                            "azimuth": azimuth,
                        },
                        "zoom_extent": zoom_extent,
                        "resize": True,
                        "colorbar": colorbar,
                    },
                )
                print(figure)
                print("out")
                data_count = f"Data Count: {ind_filter.sum()}"

                return figure, [data_count]
            else:
                return no_update, no_update

    def set_bounding_box(self, data_object):
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        obj = self.workspace.get_entity(uuid.UUID(data_object))[0]
        if isinstance(obj, Grid2D):
            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()
        elif isinstance(obj, (Points, Curve, Surface)):
            lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
            lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
        else:
            return

        width = lim_x[1] - lim_x[0]
        height = lim_y[1] - lim_y[0]

        window_center_x_max = lim_x[1] + width * 0.1
        window_center_x_value = np.mean(lim_x)
        window_center_x_min = lim_x[0] - width * 0.1

        window_center_y_max = lim_y[1] + height * 0.1
        window_center_y_value = np.mean(lim_y)
        window_center_y_min = lim_y[0] - height * 0.1

        window_width_max = width * 1.2
        window_width_value = window_width_max / 2.0
        window_width_min = 0

        window_height_max = height * 1.2
        window_height_min = 0
        window_height_value = window_height_max / 2.0

        return (
            window_center_x_min,
            window_center_x_max,
            window_center_x_value,
            window_center_y_min,
            window_center_y_max,
            window_center_y_value,
            window_width_min,
            window_width_max,
            window_width_value,
            window_height_min,
            window_height_max,
            window_height_value,
        )

    def update_octree_param(self, window_width, window_height, resolution):
        dl = resolution
        self._mesh_octree.u_cell_size.value = f"{dl/2:.0f}"
        self._mesh_octree.v_cell_size.value = f"{dl / 2:.0f}"
        self._mesh_octree.w_cell_size.value = f"{dl / 2:.0f}"
        self._mesh_octree.depth_core.value = np.ceil(
            np.min([window_width, window_height]) / 2.0
        )
        self._mesh_octree.horizontal_padding.value = (
            np.max([window_width, window_width]) / 2
        )
        resolution.indices = None
        self.write.button_style = "warning"
        self._run_params = None
        self.trigger.button_style = "danger"

    def write_trigger(self, _):
        # Widgets values populate params dictionary
        # *** populate self.params with component values
        param_dict = {}

        # Create a new workapce and copy objects into it
        temp_geoh5 = f"{self.ga_group_name.value}_{time():.0f}.geoh5"
        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as new_workspace:

            param_dict["geoh5"] = new_workspace

            for elem in [
                self,
                self._mesh_octree,
                self._topography_group,
                self._starting_model_group,
                self._reference_model_group,
                self._lower_bound_group,
                self._upper_bound_group,
            ]:
                obj, data = elem.get_selected_entities()

                if obj is not None:
                    new_obj = new_workspace.get_entity(obj.uid)[0]
                    if new_obj is None:
                        new_obj = obj.copy(parent=new_workspace, copy_children=False)
                    for d in data:
                        if new_workspace.get_entity(d.uid)[0] is None:
                            d.copy(parent=new_obj)

            if self.inversion_type.value == "magnetic vector":
                for elem in [
                    self._starting_inclination_group,
                    self._starting_declination_group,
                    self._reference_inclination_group,
                    self._reference_declination_group,
                ]:
                    obj, data = elem.get_selected_entities()
                    if obj is not None:
                        new_obj = new_workspace.get_entity(obj.uid)[0]
                        if new_obj is None:
                            new_obj = obj.copy(
                                parent=new_workspace, copy_children=False
                            )
                        for d in data:
                            if new_workspace.get_entity(d.uid)[0] is None:
                                d.copy(parent=new_obj)

            new_obj = new_workspace.get_entity(self.objects.value)
            if len(new_obj) == 0 or new_obj[0] is None:
                print("An object with data must be selected to write the input file.")
                return

            new_obj = new_obj[0]
            for key in self.data_channel_choices.options:
                widget = getattr(self, f"{key}_uncertainty_channel")
                if widget.value is not None:
                    param_dict[f"{key}_uncertainty"] = str(widget.value)
                    if new_workspace.get_entity(widget.value)[0] is None:
                        self.workspace.get_entity(widget.value)[0].copy(
                            parent=new_obj, copy_children=False
                        )
                else:
                    widget = getattr(self, f"{key}_uncertainty_floor")
                    param_dict[f"{key}_uncertainty"] = widget.value

                if getattr(self, f"{key}_channel_bool").value:
                    if not self.forward_only.value:
                        self.workspace.get_entity(
                            getattr(self, f"{key}_channel").value
                        )[0].copy(parent=new_obj)

            if self.receivers_radar_drape.value is not None:
                self.workspace.get_entity(self.receivers_radar_drape.value)[0].copy(
                    parent=new_obj
                )

            for key in self.__dict__:
                attr = getattr(self, key)
                if isinstance(attr, Widget) and hasattr(attr, "value"):
                    value = attr.value
                    if isinstance(value, uuid.UUID):
                        value = new_workspace.get_entity(value)[0]
                    if hasattr(self.params, key):
                        param_dict[key.lstrip("_")] = value
                else:
                    sub_keys = []
                    if isinstance(attr, (ModelOptions, TopographyOptions)):
                        sub_keys = [attr.identifier + "_object", attr.identifier]
                        attr = self
                    elif isinstance(attr, (MeshOctreeOptions, SensorOptions)):
                        sub_keys = attr.params_keys
                    for sub_key in sub_keys:
                        value = getattr(attr, sub_key)
                        if isinstance(value, Widget) and hasattr(value, "value"):
                            value = value.value
                        if isinstance(value, uuid.UUID):
                            value = new_workspace.get_entity(value)[0]

                        if hasattr(self.params, sub_key):
                            param_dict[sub_key.lstrip("_")] = value

            # Create new params object and write
            ifile = InputFile(
                ui_json=self.params.input_file.ui_json,
                validation_options={"disabled": True},
            )
            param_dict["geoh5"] = new_workspace
            param_dict["resolution"] = None  # No downsampling for dcip
            self._run_params = self.params.__class__(input_file=ifile, **param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=self.export_directory.selected_path,
            )

        self.write.button_style = ""
        self.trigger.button_style = "success"

    def trigger_click(self, _):
        """"""
        if self._run_params is None:
            warnings.warn("Input file must be written before running.")
            return

        self.run(self._run_params)
        self.trigger.button_style = ""

    @staticmethod
    def run(params):
        """
        Trigger the inversion.
        """
        if not isinstance(
            params, (MagneticVectorParams, MagneticScalarParams, GravityParams)
        ):
            raise ValueError(
                "Parameter 'inversion_type' must be one of "
                "'magnetic vector', 'magnetic scalar' or 'gravity'"
            )

        os.system(
            "start cmd.exe @cmd /k "
            + f"python -m {params.run_command} "
            + f'"{params.input_file.path_name}"'
        )


app = InversionApp()
app.app.run_server(host="127.0.0.1", port=8050, debug=True)
