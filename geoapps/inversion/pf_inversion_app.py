#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import multiprocessing
import os
import os.path as path
import uuid
from collections import OrderedDict

import ipywidgets as widgets
import numpy as np
from geoh5py.objects import BlockModel, Octree, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import (
    Button,
    Checkbox,
    Dropdown,
    FloatText,
    HBox,
    IntText,
    Label,
    Layout,
    VBox,
    Widget,
)

from geoapps.io.Gravity.params import GravityParams
from geoapps.io.MagneticScalar.params import MagneticScalarParams
from geoapps.io.MagneticVector.constants import app_initializer
from geoapps.io.MagneticVector.params import MagneticVectorParams
from geoapps.plotting import PlotSelection2D
from geoapps.selection import LineOptions, ObjectDataSelection, TopographyOptions
from geoapps.utils import geophysical_systems
from geoapps.utils.utils import find_value, string_2_list


def inversion_defaults():
    """
    Get defaults for gravity, magnetics and EM1D inversions
    """
    return {
        "units": {
            "gravity": "g/cc",
            "magnetic vector": "SI",
            "magnetic scalar": "SI",
        },
        "property": {
            "gravity": "density",
            "magnetic vector": "effective susceptibility",
            "magnetic scalar": "susceptibility",
        },
        "reference_value": {
            "gravity": 0.0,
            "magnetic vector": 0.0,
            "magnetic scalar": 0.0,
        },
        "starting_value": {
            "gravity": 1e-4,
            "magnetic vector": 1e-4,
            "magnetic scalar": 1e-4,
        },
        "component": {
            "gravity": "gz",
            "magnetic vector": "tmi",
            "magnetic scalar": "tmi",
        },
    }


class InversionApp(PlotSelection2D):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = MagneticVectorParams
    _select_multiple = True
    _add_groups = False
    _sensor = None
    _lines = None
    _topography = None
    inversion_parameters = None
    defaults = {}

    def __init__(self, ui_json=None, **kwargs):
        if "plot_result" in kwargs:
            self.plot_result = kwargs["plot_result"]

        app_initializer.update(kwargs)
        if ui_json is not None and path.exists(ui_json):
            self.params = self._param_class.from_path(ui_json)
        else:
            if "h5file" in app_initializer.keys():
                app_initializer["geoh5"] = app_initializer.pop("h5file")
                app_initializer["workspace"] = app_initializer["geoh5"]

            self.params = self._param_class(**app_initializer)

        self.data_object = self.objects
        self.defaults.update(self.params.to_dict(ui_json_format=False))
        self.defaults.pop("workspace", None)
        self.em_system_specs = geophysical_systems.parameters()
        self._data_count = (Label("Data Count: 0"),)
        self._forward_only = Checkbox(
            value=False,
            description="Forward only",
        )
        self._inducing_field_strength = widgets.FloatText(
            description="Amplitude (nT)",
        )
        self._inducing_field_inclination = widgets.FloatText(
            description="Inclination (d.dd)",
        )
        self._inducing_field_inclination.observe(
            self.inducing_field_inclination_change, names="value"
        )
        self._inducing_field_declination = widgets.FloatText(
            description="Declination (d.dd)",
        )
        self._inducing_field_declination.observe(
            self.inducing_field_declination_change, names="value"
        )
        self._inversion_type = Dropdown(
            options=["magnetic vector", "magnetic scalar", "gravity"],
            description="Inversion Type:",
        )
        self._write = Button(
            value=False,
            description="Write input",
            button_style="warning",
            icon="check",
        )
        self.defaults.update(self.params.to_dict(ui_json_format=False))
        self._ga_group_name = widgets.Text(
            value="Inversion_", description="Save as:", disabled=False
        )
        self._chi_factor = FloatText(
            value=1, description="Target misfit", disabled=False
        )
        self._lower_bound_group = ModelOptions("lower_bound", **self.defaults)
        self._upper_bound_group = ModelOptions("upper_bound", **self.defaults)
        self._ignore_values = widgets.Text(
            description="Value (i.e. '<0' for no negatives)",
        )
        self._max_iterations = IntText(value=10, description="Max beta Iterations")
        self._max_cg_iterations = IntText(value=30, description="Max CG Iterations")
        self._tol_cg = FloatText(value=1e-3, description="CG Tolerance")
        self._n_cpu = IntText(
            value=int(multiprocessing.cpu_count() / 2), description="Max CPUs"
        )
        self._tile_spatial = IntText(value=1, description="Number of tiles")
        # self._initial_beta = FloatText(value=1e2, description="Value:")
        self._initial_beta_ratio = FloatText(
            value=1e2, description="Beta ratio (phi_d/phi_m):"
        )
        self._initial_beta_panel = HBox([self._initial_beta_ratio])
        self._optimization = VBox(
            [
                self._max_iterations,
                self._chi_factor,
                self._initial_beta_panel,
                self._max_cg_iterations,
                self._tol_cg,
                self._n_cpu,
                self._tile_spatial,
            ]
        )
        self._starting_model_group = ModelOptions("starting_model", **self.defaults)
        self._starting_model_group.options.options = ["Constant", "Model"]
        self._starting_inclination_group = ModelOptions(
            "starting_inclination",
            description="Starting Inclination",
            units="Degree",
            **self.defaults,
        )
        self._starting_inclination_group.options.options = ["Constant", "Model"]
        self._starting_declination_group = ModelOptions(
            "starting_declination",
            description="Starting Declination",
            units="Degree",
            **self.defaults,
        )
        self._starting_declination_group.options.options = ["Constant", "Model"]
        self._reference_model_group = ModelOptions("reference_model", **self.defaults)
        self._reference_model_group.options.observe(self.update_ref)
        self._reference_inclination_group = ModelOptions(
            "reference_inclination",
            description="Reference Inclination",
            units="Degree",
            **self.defaults,
        )
        self._reference_declination_group = ModelOptions(
            "reference_declination",
            description="Reference Declination",
            units="Degree",
            **self.defaults,
        )
        self._topography_group = TopographyOptions(**self.defaults)
        self._topography_group.identifier = "topography"
        self._sensor = SensorOptions(
            objects=self._objects,
            **self.defaults,
        )
        self._detrend_data = Checkbox(description="Detrend data")
        self._detrend_order = IntText(description="Order", min=0, max=2, value=0)
        self._detrend_type = Dropdown(
            description="Method", options=["all", "corners"], value="all"
        )
        self._detrend_panel = VBox(
            [self._detrend_data, self._detrend_order, self._detrend_type]
        )
        self._detrend_data.observe(self.detrend_panel_change)
        self._alpha_s = widgets.FloatText(
            min=0,
            value=1,
            description="Reference Model (s)",
        )
        self._alpha_x = widgets.FloatText(
            min=0,
            value=1,
            description="EW-gradient (x)",
        )
        self._alpha_y = widgets.FloatText(
            min=0,
            value=1,
            description="NS-gradient (y)",
        )
        self._alpha_z = widgets.FloatText(
            min=0,
            value=1,
            description="Vertical-gradient (z)",
        )
        self._s_norm = widgets.FloatText(
            value=2,
            description="",
            continuous_update=False,
        )
        self._x_norm = widgets.FloatText(
            value=2,
            description="",
            continuous_update=False,
        )
        self._y_norm = widgets.FloatText(
            value=2,
            description="",
            continuous_update=False,
        )
        self._z_norm = widgets.FloatText(
            value=2,
            description="",
            continuous_update=False,
        )
        self._norms = VBox(
            [
                Label("Lp-norms"),
                self._s_norm,
                self._x_norm,
                self._y_norm,
                self._z_norm,
            ]
        )
        self._alphas = VBox(
            [
                Label("Scaling (alphas)"),
                self._alpha_s,
                self._alpha_x,
                self._alpha_y,
                self._alpha_z,
            ]
        )
        self.bound_panel = HBox(
            [
                VBox([Label("Lower Bounds"), self._lower_bound_group.main]),
                VBox(
                    [
                        Label("Upper Bounds"),
                        self._upper_bound_group.main,
                    ]
                ),
            ]
        )
        self._mesh_octree = MeshOctreeOptions(**self.defaults)
        self.inversion_options = {
            "starting model": VBox(
                [
                    self._starting_model_group.main,
                    self._starting_inclination_group.main,
                    self._starting_declination_group.main,
                ]
            ),
            "mesh": self._mesh_octree.main,
            "reference model": VBox(
                [
                    self._reference_model_group.main,
                    self._reference_inclination_group.main,
                    self._reference_declination_group.main,
                ]
            ),
            "regularization": HBox([self._alphas, self._norms]),
            "upper-lower bounds": self.bound_panel,
            "detrend": self._detrend_panel,
            "ignore values": VBox([self._ignore_values]),
            "optimization": self._optimization,
        }
        self.option_choices = widgets.Dropdown(
            options=list(self.inversion_options.keys()),
            value=list(self.inversion_options.keys())[0],
            disabled=False,
        )
        self.option_choices.observe(self.inversion_option_change, names="value")
        self.data_channel_choices = widgets.Dropdown(description="Component:")
        self.data_channel_panel = widgets.VBox([self.data_channel_choices])
        self.survey_type_panel = HBox([self.inversion_type])
        self.inversion_type.observe(self.inversion_type_observer, names="value")
        self.objects.observe(self.object_observer, names="value")
        self.data_channel_choices.observe(
            self.data_channel_choices_observer, names="value"
        )
        super().__init__(**self.defaults)

        for item in ["window_width", "window_height", "resolution"]:
            getattr(self, item).observe(self.update_octree_param, names="value")

        self.write.on_click(self.write_trigger)

    @property
    def alphas(self):
        return self._alphas

    @property
    def alpha_s(self):
        return self._alpha_s

    @property
    def alpha_x(self):
        return self._alpha_x

    @property
    def alpha_y(self):
        return self._alpha_y

    @property
    def alpha_z(self):
        return self._alpha_z

    # @property
    # def initial_beta(self):
    #     return self._initial_beta

    @property
    def initial_beta(self):
        return self._initial_beta_ratio

    @property
    def initial_beta_options(self):
        return self._initial_beta_options

    @property
    def chi_factor(self):
        return self._chi_factor

    @property
    def detrend_data(self):
        return self._detrend_data

    @property
    def detrend_order(self):
        return self._detrend_order

    @property
    def detrend_type(self):
        return self._detrend_type

    @property
    def ignore_values(self):
        return self._ignore_values

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @property
    def n_cpu(self):
        """
        ipywidgets.IntText()
        """
        return self._n_cpu

    @property
    def tile_spatial(self):
        """
        ipywidgets.IntText()
        """
        return self._tile_spatial

    @property
    def mesh(self):
        return self._mesh

    @property
    def norms(self):
        return self._norms

    @property
    def s_norm(self):
        return self._s_norm

    @property
    def x_norm(self):
        return self._x_norm

    @property
    def y_norm(self):
        return self._y_norm

    @property
    def z_norm(self):
        return self._z_norm

    @property
    def optimization(self):
        return self._optimization

    @property
    def out_group(self):
        return self._ga_group_name

    @property
    def ga_group_name(self):
        return self._ga_group_name

    @property
    def reference_model(self):
        if self._reference_model_group.options.value == "Model":
            return self._reference_model_group.data.value
        elif self._reference_model_group.options.value == "Constant":
            return self._reference_model_group.constant.value
        else:
            return None

    @reference_model.setter
    def reference_model(self, value):
        if isinstance(value, float):
            self._reference_model_group.options.value = "Constant"
            self._reference_model_group.constant.value = value
        elif value is None:
            self._reference_model_group.options.value = "None"
        else:
            self._reference_model_group.data.value = value

    @property
    def reference_model_object(self):
        return self._reference_model_group.objects

    @property
    def reference_inclination(self):
        if self._reference_inclination_group.options.value == "Model":
            return self._reference_inclination_group.data.value
        elif self._reference_inclination_group.options.value == "Constant":
            return self._reference_inclination_group.constant.value
        else:
            return None

    @reference_inclination.setter
    def reference_inclination(self, value):
        if isinstance(value, float):
            self._reference_inclination_group.options.value = "Constant"
            self._reference_inclination_group.constant.value = value
        elif value is None:
            self._reference_inclination_group.options.value = "None"
        else:
            self._reference_inclination_group.data.value = value

    @property
    def reference_inclination_object(self):
        return self._reference_inclination_group.objects

    @property
    def reference_declination(self):
        if self._reference_declination_group.options.value == "Model":
            return self._reference_declination_group.data.value
        elif self._reference_declination_group.options.value == "Constant":
            return self._reference_declination_group.constant.value
        else:
            return None

    @reference_declination.setter
    def reference_declination(self, value):
        if isinstance(value, float):
            self._reference_declination_group.options.value = "Constant"
            self._reference_declination_group.constant.value = value
        elif value is None:
            self._reference_declination_group.options.value = "None"
        else:
            self._reference_declination_group.data.value = value

    @property
    def reference_declination_object(self):
        return self._reference_declination_group.objects

    @property
    def starting_model(self):
        if self._starting_model_group.options.value == "Model":
            return self._starting_model_group.data.value
        elif self._starting_model_group.options.value == "Constant":
            return self._starting_model_group.constant.value
        else:
            return None

    @starting_model.setter
    def starting_model(self, value):
        if isinstance(value, float):
            self._starting_model_group.options.value = "Constant"
            self._starting_model_group.constant.value = value
        else:
            self._starting_model_group.data.value = value

    @property
    def starting_model_object(self):
        return self._starting_model_group.objects

    @property
    def starting_inclination(self):
        if self._starting_inclination_group.options.value == "Model":
            return self._starting_inclination_group.data.value
        elif self._starting_inclination_group.options.value == "Constant":
            return self._starting_inclination_group.constant.value
        else:
            return None

    @starting_inclination.setter
    def starting_inclination(self, value):
        if isinstance(value, float):
            self._starting_inclination_group.options.value = "Constant"
            self._starting_inclination_group.constant.value = value
        elif value is None:
            self._starting_inclination_group.options.value = "None"
        else:
            self._starting_inclination_group.data.value = value

    @property
    def starting_inclination_object(self):
        return self._starting_inclination_group.objects

    @property
    def starting_declination(self):
        if self._starting_declination_group.options.value == "Model":
            return self._starting_declination_group.data.value
        elif self._starting_declination_group.options.value == "Constant":
            return self._starting_declination_group.constant.value
        else:
            return None

    @starting_declination.setter
    def starting_declination(self, value):
        if isinstance(value, float):
            self._starting_declination_group.options.value = "Constant"
            self._starting_declination_group.constant.value = value
        elif value is None:
            self._starting_declination_group.options.value = "None"
        else:
            self._starting_declination_group.data.value = value

    @property
    def starting_declination_object(self):
        return self._starting_declination_group.objects

    @property
    def lower_bound(self):
        if self._lower_bound_group.options.value == "Model":
            return self._lower_bound_group.data.value
        elif self._lower_bound_group.options.value == "Constant":
            return self._lower_bound_group.constant.value
        else:
            return None

    @lower_bound.setter
    def lower_bound(self, value):
        if isinstance(value, float):
            self._lower_bound_group.options.value = "Constant"
            self._lower_bound_group.constant.value = value
        elif value is None:
            self._lower_bound_group.options.value = "None"
        else:
            self._lower_bound_group.data.value = value

    @property
    def lower_bound_object(self):
        return self._lower_bound_group.objects

    @property
    def upper_bound(self):
        if self._upper_bound_group.options.value == "Model":
            return self._upper_bound_group.data.value
        elif self._upper_bound_group.options.value == "Constant":
            return self._upper_bound_group.constant.value
        else:
            return None

    @upper_bound.setter
    def upper_bound(self, value):
        if isinstance(value, float):
            self._upper_bound_group.options.value = "Constant"
            self._upper_bound_group.constant.value = value
        elif value is None:
            self._upper_bound_group.options.value = "None"
        else:
            self._upper_bound_group.data.value = value

    @property
    def upper_bound_object(self):
        return self._upper_bound_group.objects

    @property
    def tol_cg(self):
        return self._tol_cg

    @property
    def data_count(self):
        """"""
        return self._data_count

    @property
    def forward_only(self):
        """"""
        return self._forward_only

    @property
    def inducing_field_strength(self):
        """"""
        return self._inducing_field_strength

    @property
    def inducing_field_inclination(self):
        """"""
        return self._inducing_field_inclination

    @property
    def inducing_field_declination(self):
        """"""
        return self._inducing_field_declination

    @property
    def lines(self):
        if getattr(self, "_lines", None) is None:
            self._lines = LineOptions(workspace=self._workspace, objects=self._objects)
            self.lines.lines.observe(self.update_selection, names="value")
        return self._lines

    @property
    def main(self):
        if getattr(self, "_main", None) is None:
            self._main = VBox(
                [
                    self.project_panel,
                    VBox(
                        [
                            self.objects,
                            self.survey_type_panel,
                        ],
                        layout=Layout(border="solid"),
                    ),
                    VBox(
                        [
                            Label("Input Data"),
                            HBox(
                                [
                                    self.data_channel_panel,
                                    self.window_selection,
                                ],
                            ),
                        ],
                        layout=Layout(border="solid"),
                    ),
                    HBox(
                        [
                            VBox(
                                [
                                    Label("Topography"),
                                    self.topography_group.main,
                                ]
                            ),
                            VBox(
                                [
                                    Label("Sensor Location"),
                                    self.sensor.main,
                                ]
                            ),
                        ],
                        layout=Layout(border="solid"),
                    ),
                    VBox(
                        [
                            Label("Inversion Parameters"),
                            self.forward_only,
                            HBox(
                                [
                                    self.option_choices,
                                    self.inversion_options[self.option_choices.value],
                                ],
                            ),
                        ],
                        layout=Layout(border="solid"),
                    ),
                    VBox(
                        [
                            Label("Output"),
                            self._ga_group_name,
                            self.export_directory,
                            self.write,
                            self.trigger,
                        ],
                        layout=Layout(border="solid"),
                    ),
                ]
            )
        return self._main

    @property
    def sensor(self):
        return self._sensor

    @property
    def z_from_topo(self):
        return self.sensor.z_from_topo

    @property
    def receivers_radar_drape(self):
        return self.sensor.data

    @property
    def receivers_offset_x(self):
        return self.sensor.receivers_offset_x

    @property
    def receivers_offset_y(self):
        return self.sensor.receivers_offset_y

    @property
    def receivers_offset_z(self):
        return self.sensor.receivers_offset_z

    @property
    def starting_channel(self):
        """"""
        return self._starting_channel

    @property
    def inversion_type(self):
        """"""
        return self._inversion_type

    @property
    def topography(self):
        if self._topography_group.options.value == "Object":
            return self._topography_group.data.value
        elif self._topography_group.options.value == "Constant":
            return self._topography_group.constant.value
        else:
            return None

    @topography.setter
    def topography(self, value):
        if isinstance(value, float):
            self._topography_group.constant.value = value
            self._topography_group.options.value = "Constant"
        elif value is None:
            self._topography_group.options.value = "None"
        else:
            self._topography_group.options.value = "Object"
            self._topography_group.data.value = value

    @property
    def topography_object(self):
        return self._topography_group.objects

    @property
    def topography_group(self):
        return self._topography_group

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self.workspace = Workspace(self.h5file)
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self.base_workspace_changes(workspace)
        self.update_objects_list()
        # self.lines.workspace = workspace
        self.sensor.workspace = workspace
        self._topography_group.workspace = workspace
        self._reference_model_group.workspace = workspace
        self._starting_model_group.workspace = workspace
        self._mesh_octree.workspace = workspace
        self._starting_inclination_group.workspace = workspace
        self._starting_declination_group.workspace = workspace
        self._reference_inclination_group.workspace = workspace
        self._reference_declination_group.workspace = workspace

    @property
    def write(self):
        """"""
        return self._write

    def detrend_panel_change(self, _):
        if self.detrend_data.value:
            self._detrend_panel.children = [
                self.detrend_data,
                self.detrend_order,
                self.detrend_type,
            ]
        else:
            self._detrend_panel.children = [self.detrend_data]

    # Observers
    def update_ref(self, _):
        alphas = [alpha.value for alpha in self.alphas.children]
        if self._reference_model_group.options.value == "None":
            alphas[0] = 0.0
        else:
            alphas[0] = 1.0
        self.alphas.value = ", ".join(list(map(str, alphas)))

    def inducing_field_inclination_change(self, _):
        if self.inversion_type.value == "magnetic vector":
            self._reference_inclination_group.constant.value = (
                self._inducing_field_inclination.value
            )
            self._starting_inclination_group.constant.value = (
                self._inducing_field_inclination.value
            )

    def inducing_field_declination_change(self, _):
        if self.inversion_type.value == "magnetic vector":
            self._reference_declination_group.constant.value = (
                self._inducing_field_declination.value
            )
            self._starting_declination_group.constant.value = (
                self._inducing_field_declination.value
            )

    def inversion_option_change(self, _):
        self._main.children[4].children[2].children = [
            self.option_choices,
            self.inversion_options[self.option_choices.value],
        ]

    def trigger_click(self, _):
        """"""
        self.run(self.params)
        self.trigger.button_style = ""

    def inversion_type_observer(self, _):
        """
        Change the application on change of system
        """
        params = self.params.to_dict(ui_json_format=False)
        if self.inversion_type.value == "magnetic vector" and not isinstance(
            self.params, MagneticVectorParams
        ):
            self._param_class = MagneticVectorParams
            params["inversion_type"] = "magnetic vector"
            params["out_group"] = "VectorInversion"

        elif self.inversion_type.value == "magnetic scalar" and not isinstance(
            self.params, MagneticScalarParams
        ):
            params["inversion_type"] = "magnetic scalar"
            params["out_group"] = "SusceptibilityInversion"
            self._param_class = MagneticScalarParams
        elif self.inversion_type.value == "gravity" and not isinstance(
            self.params, GravityParams
        ):
            params["inversion_type"] = "gravity"
            params["out_group"] = "GravityInversion"
            self._param_class = GravityParams

        self.params = self._param_class(verbose=False)
        self.ga_group_name.value = self.params.defaults["out_group"]

        if self.inversion_type.value in ["magnetic vector", "magnetic scalar"]:
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
        flag = self.inversion_type.value
        self._reference_model_group.units = inversion_defaults()["units"][flag]
        self._reference_model_group.constant.value = inversion_defaults()[
            "reference_value"
        ][flag]
        self._reference_model_group.description.value = (
            "Reference " + inversion_defaults()["property"][flag]
        )
        self._starting_model_group.units = inversion_defaults()["units"][flag]
        self._starting_model_group.constant.value = inversion_defaults()[
            "starting_value"
        ][flag]
        self._starting_model_group.description.value = (
            "Starting " + inversion_defaults()["property"][flag]
        )
        data_channel_options = {}
        self.data_channel_choices.options = data_type_list

        if self.workspace.get_entity(self.objects.value):
            obj, _ = self.get_selected_entities()
            children_list = {child.uid: child.name for child in obj.children}
            ordered = OrderedDict(sorted(children_list.items(), key=lambda t: t[1]))
            options = [
                [name, uid]
                for uid, name in ordered.items()
                if "visual parameter" not in name.lower()
            ]
        else:
            options = []

        def channel_setter(caller):
            channel = caller["owner"]
            data_widget = getattr(self, f"{channel.header}_group")
            entity = self.workspace.get_entity(self.objects.value)[0]
            if channel.value is None or channel.value not in [
                child.uid for child in entity.children
            ]:
                data_widget.children[0].value = False
                data_widget.children[3].children[0].value = 1.0
            else:
                data_widget.children[0].value = True
                values = self.workspace.get_entity(channel.value)[0].values
                if values is not None and values.dtype in [
                    np.float32,
                    np.float64,
                    np.int32,
                ]:
                    data_widget.children[3].children[0].value = np.round(
                        np.percentile(np.abs(values[~np.isnan(values)]), 5), 5
                    )

            # Trigger plot update
            if self.data_channel_choices.value == channel.header:
                self.plotting_data = channel.value
                self.refresh.value = False
                self.refresh.value = True

        def uncert_setter(caller):
            channel = caller["owner"]
            data_widget = getattr(self, f"{channel.header}_group")
            if isinstance(channel, FloatText) and channel.value > 0:
                data_widget.children[3].children[1].value = None
            elif channel.value is not None:
                data_widget.children[3].children[0].value = 0.0

        def value_setter(self, key, value):
            """Assign value or channel"""
            if isinstance(value, float):
                getattr(self, key + "_floor").value = value
            else:
                getattr(self, key + "_channel").value = (
                    uuid.UUID(value) if isinstance(value, str) else value
                )

        for key in data_type_list:

            if hasattr(self, f"{key}_group"):
                data_channel_options[key] = getattr(self, f"{key}_group", None)
            else:
                setattr(
                    self,
                    f"{key}_channel_bool",
                    Checkbox(
                        value=False,
                        indent=False,
                        description="Active",
                    ),
                )
                setattr(self, f"{key}_channel", Dropdown(description="Channel:"))
                setattr(
                    self, f"{key}_uncertainty_floor", FloatText(description="Floor:")
                )
                setattr(
                    self,
                    f"{key}_uncertainty_channel",
                    Dropdown(
                        description="[Optional] Channel:",
                        style={"description_width": "initial"},
                    ),
                )
                setattr(
                    self,
                    f"{key}_group",
                    VBox(
                        [
                            getattr(self, f"{key}_channel_bool"),
                            getattr(self, f"{key}_channel"),
                            Label("Uncertainties"),
                            VBox(
                                [
                                    getattr(self, f"{key}_uncertainty_floor"),
                                    getattr(self, f"{key}_uncertainty_channel"),
                                ]
                            ),
                        ]
                    ),
                )

                setattr(InversionApp, f"{key}_uncertainty", value_setter)

                data_channel_options[key] = getattr(self, f"{key}_group")
                data_channel_options[key].children[3].children[0].header = key
                data_channel_options[key].children[3].children[1].header = key
                data_channel_options[key].children[1].header = key
                data_channel_options[key].children[1].observe(
                    channel_setter, names="value"
                )
                data_channel_options[key].children[3].children[0].observe(
                    uncert_setter, names="value"
                )
                data_channel_options[key].children[3].children[1].observe(
                    uncert_setter, names="value"
                )
            data_channel_options[key].children[1].options = [["", None]] + options
            data_channel_options[key].children[3].children[1].options = [
                ["", None]
            ] + options

            # data_channel_options[key].children[1].value = find_value(options, [key])

        self.data_channel_choices.value = inversion_defaults()["component"][
            self.inversion_type.value
        ]
        self.data_channel_choices.data_channel_options = data_channel_options
        self.data_channel_panel.children = [
            self.data_channel_choices,
            data_channel_options[self.data_channel_choices.value],
        ]
        self.write.button_style = "warning"
        self.trigger.button_style = "danger"
        if self.inversion_type.value in ["magnetic vector", "magnetic scalar"]:
            self.survey_type_panel.children = [
                self.inversion_type,
                VBox(
                    [
                        Label("Inducing Field Parameters"),
                        self.inducing_field_strength,
                        self.inducing_field_inclination,
                        self.inducing_field_declination,
                    ]
                ),
            ]
        else:
            self.survey_type_panel.children = [self.inversion_type]

        if self.inversion_type.value == "magnetic scalar":
            self._lower_bound_group.options.value = "Constant"
            self._lower_bound_group.constant.value = 0.0
        else:
            self._lower_bound_group.options.value = "None"

        if self.inversion_type.value == "magnetic vector":
            self.inversion_options["starting model"].children = [
                self._starting_model_group.main,
                self._starting_inclination_group.main,
                self._starting_declination_group.main,
            ]
            self.inversion_options["reference model"].children = [
                self._reference_model_group.main,
                self._reference_inclination_group.main,
                self._reference_declination_group.main,
            ]
        else:
            self.inversion_options["starting model"].children = [
                self._starting_model_group.main,
            ]
            self.inversion_options["reference model"].children = [
                self._reference_model_group.main,
            ]

    def object_observer(self, _):
        """ """
        self.resolution.indices = None
        if self.workspace.get_entity(self.objects.value):
            self.update_data_list(None)
            self.sensor.update_data_list(None)
            self.inversion_type_observer(None)
            self.write.button_style = "warning"
            self.trigger.button_style = "danger"

    def data_channel_choices_observer(self, _):
        if hasattr(
            self.data_channel_choices, "data_channel_options"
        ) and self.data_channel_choices.value in (
            self.data_channel_choices.data_channel_options.keys()
        ):
            data_widget = self.data_channel_choices.data_channel_options[
                self.data_channel_choices.value
            ]
            self.data_channel_panel.children = [self.data_channel_choices, data_widget]

            if (
                self.workspace.get_entity(self.objects.value)
                and data_widget.children[1].value is None
            ):
                _, data_list = self.get_selected_entities()
                options = [[data.name, data.uid] for data in data_list]
                data_widget.children[1].value = find_value(
                    options, [self.data_channel_choices.value]
                )

            self.plotting_data = data_widget.children[1].value
            self.refresh.value = False
            self.refresh.value = True

        self.write.button_style = "warning"
        self.trigger.button_style = "danger"

    def update_octree_param(self, _):
        dl = self.resolution.value
        self._mesh_octree.u_cell_size.value = f"{dl/2:.0f}"
        self._mesh_octree.v_cell_size.value = f"{dl / 2:.0f}"
        self._mesh_octree.w_cell_size.value = f"{dl / 2:.0f}"
        self._mesh_octree.depth_core.value = np.ceil(
            np.min([self.window_width.value, self.window_height.value]) / 2.0
        )
        self._mesh_octree.horizontal_padding.value = (
            np.max([self.window_width.value, self.window_width.value]) / 2
        )
        self.resolution.indices = None
        self.write.button_style = "warning"
        self.trigger.button_style = "danger"

    def update_selection(self, _):
        self.highlight_selection = {self.lines.data.value: self.lines.lines.value}
        self.refresh.value = False
        self.refresh.value = True

    def write_trigger(self, _):
        # Copy object to work geoh5
        new_workspace = Workspace(
            path.join(
                self.export_directory.selected_path,
                self._ga_group_name.value + ".geoh5",
            )
        )
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
                        new_obj = obj.copy(parent=new_workspace, copy_children=False)
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
                setattr(self.params, f"{key}_uncertainty", str(widget.value))
                if new_workspace.get_entity(widget.value)[0] is None:
                    self.workspace.get_entity(widget.value)[0].copy(
                        parent=new_obj, copy_children=False
                    )
            else:
                widget = getattr(self, f"{key}_uncertainty_floor")
                setattr(self.params, f"{key}_uncertainty", widget.value)

            if getattr(self, f"{key}_channel_bool").value:
                self.workspace.get_entity(getattr(self, f"{key}_channel").value)[
                    0
                ].copy(parent=new_obj)

        if self.receivers_radar_drape.value is not None:
            self.workspace.get_entity(self.receivers_radar_drape.value)[0].copy(
                parent=new_obj
            )

        self.params.geoh5 = new_workspace.h5file
        self.params.workspace = new_workspace

        for key in self.__dict__:
            try:
                attr = getattr(self, key)
                if isinstance(attr, Widget):
                    setattr(self.params, key, attr.value)
                else:
                    sub_keys = []
                    if isinstance(attr, (ModelOptions, TopographyOptions)):
                        sub_keys = [attr.identifier, attr.identifier + "_object"]
                        attr = self
                    elif isinstance(attr, (MeshOctreeOptions, SensorOptions)):
                        sub_keys = attr.params_keys
                    for sub_key in sub_keys:
                        value = getattr(attr, sub_key)
                        if isinstance(value, Widget):
                            value = value.value
                        if isinstance(value, uuid.UUID):
                            value = str(value)
                        setattr(self.params, sub_key, value)

            except AttributeError:
                continue

        self.params.write_input_file(
            name=self._ga_group_name.value + ".ui.json",
            path=self.export_directory.selected_path,
        )
        self.write.button_style = ""
        self.trigger.button_style = "success"

    @staticmethod
    def run(params):

        if isinstance(params, MagneticVectorParams):
            inversion_routine = "magnetic_vector_inversion"
        elif isinstance(params, MagneticScalarParams):
            inversion_routine = "magnetic_scalar_inversion"
        elif isinstance(params, GravityParams):
            inversion_routine = "grav_inversion"

        else:
            raise ValueError(
                "Parameter 'inversion_type' must be one of "
                "'magnetic vector', 'magnetic scalar' or 'gravity'"
            )
        os.system(
            "start cmd.exe @cmd /k "
            + f"python -m geoapps.drivers.{inversion_routine} "
            + f"{params.input_file.filepath}"
        )

    def file_browser_change(self, _):
        """
        Change the target h5file
        """
        if not self.file_browser._select.disabled:
            _, extension = path.splitext(self.file_browser.selected)

            if extension == ".json" and getattr(self, "_param_class", None) is not None:

                # Read the inversion type first...
                with open(self.file_browser.selected) as f:
                    data = json.load(f)

                if data["inversion_type"] == "gravity":
                    self._param_class = GravityParams
                elif data["inversion_type"] == "magnetic vector":
                    self._param_class = MagneticVectorParams
                elif data["inversion_type"] == "magnetic scalar":
                    self._param_class = MagneticScalarParams

                self.params = getattr(self, "_param_class").from_path(
                    self.file_browser.selected
                )
                self.refresh.value = False
                self.__populate__(**self.params.to_dict(ui_json_format=False))
                self.refresh.value = True

            elif extension == ".geoh5":
                self.h5file = self.file_browser.selected


class SensorOptions(ObjectDataSelection):
    """
    Define the receiver spatial parameters
    """

    _options = None
    defaults = {}
    params_keys = [
        "receivers_offset_x",
        "receivers_offset_y",
        "receivers_offset_z",
        "z_from_topo",
        "receivers_radar_drape",
    ]

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        self._receivers_offset_x = FloatText(description="dx (+East)", value=0.0)
        self._receivers_offset_y = FloatText(description="dy (+North)", value=0.0)
        self._receivers_offset_z = FloatText(description="dz (+ve up)", value=0.0)
        self._z_from_topo = Checkbox(description="Set Z from topo + offsets")
        self.data.description = "Radar (Optional):"
        self._receivers_radar_drape = self.data
        super().__init__(**self.defaults)

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.z_from_topo,
                    Label("Offsets"),
                    self._receivers_offset_x,
                    self._receivers_offset_y,
                    self._receivers_offset_z,
                    self._receivers_radar_drape,
                ]
            )

        return self._main

    @property
    def offset(self):
        return self._offset

    @property
    def receivers_radar_drape(self):
        return self._receivers_radar_drape

    @property
    def receivers_offset_x(self):
        return self._receivers_offset_x

    @property
    def receivers_offset_y(self):
        return self._receivers_offset_y

    @property
    def receivers_offset_z(self):
        return self._receivers_offset_z

    @property
    def z_from_topo(self):
        return self._z_from_topo


class MeshOctreeOptions(ObjectDataSelection):
    """
    Widget used for the creation of an octree meshes
    """

    params_keys = [
        "mesh",
        "mesh_from_params",
        "u_cell_size",
        "v_cell_size",
        "w_cell_size",
        "octree_levels_topo",
        "octree_levels_obs",
        "depth_core",
        "horizontal_padding",
        "vertical_padding",
        "max_distance",
    ]

    def __init__(self, **kwargs):
        self._mesh = self.objects
        self._mesh_from_params = Checkbox(value=False, description="Create")
        self._mesh_from_params.observe(self.from_params_choice, names="value")
        self._u_cell_size = widgets.FloatText(
            value=25.0,
            description="",
        )
        self._v_cell_size = widgets.FloatText(
            value=25.0,
            description="",
        )
        self._w_cell_size = widgets.FloatText(
            value=25.0,
            description="",
        )
        self._octree_levels_topo = widgets.Text(
            value="0, 0, 0, 2",
            description="# Cells below topography",
        )
        self._octree_levels_obs = widgets.Text(
            value="5, 5, 5, 5",
            description="# Cells below sensors",
        )
        self._depth_core = FloatText(
            value=500,
            description="Minimum depth (m)",
        )
        self._horizontal_padding = widgets.FloatText(
            value=1000.0,
            description="Horizontal padding (m)",
        )
        self._vertical_padding = widgets.FloatText(
            value=1000.0,
            description="Vertical padding (m)",
        )
        self._max_distance = FloatText(
            value=1000,
            description="Maximum distance (m)",
        )
        self._parameters = widgets.VBox(
            [
                Label("Core cell size (u, v, z)"),
                self._u_cell_size,
                self._v_cell_size,
                self._w_cell_size,
                Label("Refinement Layers"),
                self._octree_levels_topo,
                self._octree_levels_obs,
                self._max_distance,
                Label("Dimensions"),
                self._horizontal_padding,
                self._vertical_padding,
                self._depth_core,
            ]
        )
        self._main = VBox([self.objects, self.mesh_from_params])
        self._objects.observe(self.mesh_selection, names="value")

        super().__init__(**kwargs)

    @property
    def main(self):
        return self._main

    @property
    def mesh(self):
        return self._mesh

    @property
    def mesh_from_params(self):
        return self._mesh_from_params

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @property
    def w_cell_size(self):
        return self._w_cell_size

    @property
    def depth_core(self):
        return self._depth_core

    @property
    def max_distance(self):
        return self._max_distance

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @octree_levels_obs.getter
    def octree_levels_obs(self):
        return string_2_list(self._octree_levels_obs.value)

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @octree_levels_topo.getter
    def octree_levels_topo(self):
        return string_2_list(self._octree_levels_topo.value)

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @property
    def main(self):
        return self._main

    def from_params_choice(self, _):
        if self._mesh_from_params.value:
            self._main.children = [
                self.objects,
                self.mesh_from_params,
                self._parameters,
            ]
            if self._objects.value is not None:
                self._objects.value = None
        else:
            self._main.children = [self.objects, self.mesh_from_params]

    def mesh_selection(self, _):
        if self._objects.value is not None:
            self._mesh_from_params.value = False


class ModelOptions(ObjectDataSelection):
    """
    Widgets for the selection of model options
    """

    defaults = {}

    def __init__(self, identifier: str = None, **kwargs):
        self._units = "Units"
        self._identifier = identifier
        self._object_types = (BlockModel, Octree, Surface)
        self._options = widgets.RadioButtons(
            options=["Model", "Constant", "None"],
            value="Constant",
            disabled=False,
        )
        self._options.observe(self.update_panel, names="value")
        self.objects.description = "Object"

        self.data.description = "Values"
        self._constant = FloatText(description=self.units)
        self._description = Label()

        super().__init__(**kwargs)

        self.objects.observe(self.objects_setter, names="value")
        self.selection_widget = self.main
        self._main = widgets.VBox(
            [self._description, widgets.VBox([self._options, self._constant])]
        )

    def update_panel(self, _):

        if self._options.value == "Model":
            self._main.children[1].children = [self._options, self.selection_widget]
            self._main.children[1].children[1].layout.visibility = "visible"
        elif self._options.value == "Constant":
            self._main.children[1].children = [self._options, self._constant]
            self._main.children[1].children[1].layout.visibility = "visible"
        else:
            self._main.children[1].children[1].layout.visibility = "hidden"

    def objects_setter(self, _):
        if self.objects.value is not None:
            self.options.value = "Model"

    @property
    def constant(self):
        return self._constant

    @property
    def description(self):
        return self._description

    @property
    def identifier(self):
        return self._identifier

    @property
    def options(self):
        return self._options

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = value
        self._constant.description = value
