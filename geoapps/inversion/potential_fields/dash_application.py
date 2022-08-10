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

    def run(self):
        # The reloader has not yet run - open the browser
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=False)


app = InversionApp()
app.app.run()
