#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# https://stackoverflow.com/questions/49851280/showing-a-simple-matplotlib-plot-in-plotly-dash

from __future__ import annotations

import os
import webbrowser

import numpy as np
from dash import Input, Output
from flask import Flask
from geoh5py.objects import BlockModel, Curve, Octree, Points, Surface
from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

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
from geoapps.utils import geophysical_systems, warn_module_not_found
from geoapps.utils.list import find_value
from geoapps.utils.string import string_2_list


class InversionApp:
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

        # super().__init__(**self.params.to_dict())

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
            Output(component_id="topography_none", component_property="style"),
            Output(component_id="topography_object", component_property="style"),
            Output(component_id="topography_sensor", component_property="style"),
            Output(component_id="topography_constant", component_property="style"),
            Input(component_id="topography_options", component_property="value"),
        )(InversionApp.update_topography_visibility)

        for model_type in ["starting", "ref"]:
            for param in ["susceptibility", "inclination", "declination"]:
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
            Output(component_id="lower_bounds_const_div", component_property="style"),
            Output(component_id="lower_bounds_mod_div", component_property="style"),
            Input(component_id="lower_bounds_options", component_property="value"),
        )(InversionApp.update_model_visibility)
        self.app.callback(
            Output(component_id="upper_bounds_const_div", component_property="style"),
            Output(component_id="upper_bounds_mod_div", component_property="style"),
            Input(component_id="upper_bounds_options", component_property="value"),
        )(InversionApp.update_model_visibility)

        self.app.callback(
            Output(component_id="starting_model_div", component_property="style"),
            Output(component_id="mesh_div", component_property="style"),
            Output(component_id="reference_model_div", component_property="style"),
            Output(component_id="regularization_div", component_property="style"),
            Output(component_id="upper_lower_bounds_div", component_property="style"),
            Output(component_id="detrend_div", component_property="style"),
            Output(component_id="ignore_values_div", component_property="style"),
            Output(component_id="optimization_div", component_property="style"),
            Input(component_id="param_dropdown", component_property="value"),
        )(InversionApp.update_inversion_params_visibility)

    @staticmethod
    def update_inducing_params_visibility(selection):
        if selection == "magnetic vector" or selection == "magnetic scalar":
            return {"display": "inline-block"}
        else:
            return {"display": "none"}

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

    def run(self):
        # The reloader has not yet run - open the browser
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=True)


app = InversionApp()
app.app.run()
