#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
import multiprocessing
import os
import os.path as path
import uuid
import warnings
from collections import OrderedDict
from time import time

import numpy as np
from geoh5py.data import Data
from geoh5py.objects import Octree
from geoh5py.shared import Entity
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps.base.plot import PlotSelection2D
from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.inversion.potential_fields.magnetic_vector.constants import app_initializer
from geoapps.utils import geophysical_systems, warn_module_not_found
from geoapps.utils.list import find_value

from ...base.application import BaseApplication

with warn_module_not_found():
    import ipywidgets as widgets
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

from .gravity.params import GravityParams
from .magnetic_scalar.params import MagneticScalarParams
from .magnetic_vector.params import MagneticVectorParams


def inversion_defaults():
    """
    Get defaults for gravity, magnetics and EM1D inversion
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
    _run_params = None
    _sensor = None
    _topography = None
    inversion_parameters = None

    def __init__(self, ui_json=None, **kwargs):
        if "plot_result" in kwargs:
            self.plot_result = kwargs["plot_result"]
            kwargs.pop("plot_result")

        app_initializer.update(kwargs)
        if ui_json is not None and path.exists(ui_json):
            ifile = InputFile.read_ui_json(ui_json)
            self.params = self._param_class(ifile, **kwargs)
        else:
            self.params = self._param_class(**app_initializer)
        self.data_object = self.objects

        for key, value in self.params.to_dict().items():
            if isinstance(value, Entity):
                self.defaults[key] = value.uid
            else:
                self.defaults[key] = value

        self.defaults["tmi_channel_bool"] = True
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
            description="inversion Type:",
        )
        self._store_sensitivities = Dropdown(
            options=["ram", "disk"], description="Storage device:", value="disk"
        )
        self._write = Button(
            value=False,
            description="Write input",
            button_style="warning",
            icon="check",
        )
        self._ga_group_name = widgets.Text(
            value="Inversion_", description="Save as:", disabled=False
        )
        self._chi_factor = FloatText(
            value=1, description="Target misfit", disabled=False
        )
        self._mesh_octree = MeshOctreeOptions(**self.defaults)
        self._lower_bound_group = ModelOptions(
            "lower_bound",
            add_xyz=False,
            objects=self._mesh_octree.mesh,
            **self.defaults,
        )
        self._upper_bound_group = ModelOptions(
            "upper_bound",
            add_xyz=False,
            objects=self._mesh_octree.mesh,
            **self.defaults,
        )
        self._ignore_values = widgets.Text(
            description="Value (i.e. '<0' for no negatives)",
        )
        self._max_global_iterations = IntText(
            value=30, description="Max beta iterations"
        )
        self._max_irls_iterations = IntText(value=10, description="Max IRLS iterations")
        self._max_cg_iterations = IntText(value=30, description="Max CG Iterations")
        self._coolingRate = IntText(value=1, description="Iterations per beta")
        self._coolingFactor = FloatText(value=2, description="Beta cooling factor")
        self._tol_cg = FloatText(value=1e-3, description="CG Tolerance")
        self._n_cpu = IntText(
            value=int(multiprocessing.cpu_count() / 2), description="Max CPUs"
        )
        self._tile_spatial = IntText(value=1, description="Number of tiles")
        self._initial_beta_ratio = FloatText(
            value=1e2, description="Beta ratio (phi_d/phi_m):"
        )
        self._initial_beta_panel = HBox([self._initial_beta_ratio])
        self._optimization = VBox(
            [
                self._max_global_iterations,
                self._max_irls_iterations,
                self._coolingRate,
                self._coolingFactor,
                self._chi_factor,
                self._initial_beta_panel,
                self._max_cg_iterations,
                self._tol_cg,
                self._n_cpu,
                self._store_sensitivities,
                self._tile_spatial,
            ]
        )
        self._starting_model_group = ModelOptions(
            "starting_model",
            add_xyz=False,
            objects=self._mesh_octree.mesh,
            **self.defaults,
        )
        self._starting_model_group.options.options = ["Constant", "Model"]
        self._starting_inclination_group = ModelOptions(
            "starting_inclination",
            objects=self._mesh_octree.mesh,
            description="Starting Inclination",
            units="Degree",
            add_xyz=False,
            **self.defaults,
        )
        self._starting_inclination_group.options.options = ["Constant", "Model"]
        self._starting_declination_group = ModelOptions(
            "starting_declination",
            objects=self._mesh_octree.mesh,
            description="Starting Declination",
            units="Degree",
            add_xyz=False,
            **self.defaults,
        )
        self._starting_declination_group.options.options = ["Constant", "Model"]
        self._reference_model_group = ModelOptions(
            "reference_model",
            add_xyz=False,
            objects=self._mesh_octree.mesh,
            **self.defaults,
        )
        self._reference_model_group.options.observe(self.update_ref)
        self._reference_inclination_group = ModelOptions(
            "reference_inclination",
            objects=self._mesh_octree.mesh,
            description="Reference Inclination",
            units="Degree",
            add_xyz=False,
            **self.defaults,
        )
        self._reference_declination_group = ModelOptions(
            "reference_declination",
            objects=self._mesh_octree.mesh,
            description="Reference Declination",
            units="Degree",
            add_xyz=False,
            **self.defaults,
        )
        self._topography_group = TopographyOptions(add_xyz=False, **self.defaults)
        self._topography_group.identifier = "topography"
        self._sensor = SensorOptions(
            objects=self._objects,
            add_xyz=False,
            receivers_offset_z=self.defaults["receivers_offset_z"],
            z_from_topo=self.defaults["z_from_topo"],
            receivers_radar_drape=self.defaults["receivers_radar_drape"],
        )

        self._detrend_type = Dropdown(
            description="Method", options=["", "all", "perimeter"], value="all"
        )
        self._detrend_order = widgets.IntText(description="Order", min=0, value=0)
        self._detrend_panel = VBox([self._detrend_type, self._detrend_order])
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
            options=list(self.inversion_options),
            value=list(self.inversion_options)[0],
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
        self.plotting_data = None
        super().__init__(**self.defaults)
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

    @property
    def initial_beta(self):
        return self._initial_beta_ratio

    @property
    def chi_factor(self):
        return self._chi_factor

    @property
    def coolingRate(self):
        return self._coolingRate

    @property
    def coolingFactor(self):
        return self._coolingFactor

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
    def max_global_iterations(self):
        return self._max_global_iterations

    @property
    def max_irls_iterations(self):
        return self._max_irls_iterations

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @property
    def n_cpu(self):
        return self._n_cpu

    @property
    def tile_spatial(self):
        return self._tile_spatial

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
    def store_sensitivities(self):
        return self._store_sensitivities

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
                            Label("inversion Parameters"),
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
    def receivers_offset_z(self):
        return self.sensor.receivers_offset_z

    @property
    def inversion_type(self):
        """"""
        return self._inversion_type

    @property
    def topography(self):
        if self.topography_group.options.value == "Object":
            return self.topography_group.data.value
        elif self.topography_group.options.value == "Constant":
            return self.topography_group.constant.value
        else:
            return None

    @topography.setter
    def topography(self, value):
        if isinstance(value, float):
            self.topography_group.constant.value = value
            self.topography_group.options.value = "Constant"
        elif value is None:
            self.topography_group.options.value = "None"
        else:
            self.topography_group.options.value = "Object"
            self.topography_group.data.value = value

    @property
    def topography_object(self):
        return self.topography_group.objects

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
            self.workspace = Workspace(self.h5file, mode="r")
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self.base_workspace_changes(workspace)
        self.update_objects_list()
        self.sensor.workspace = workspace
        self._topography_group.workspace = workspace
        self._reference_model_group.workspace = workspace
        self._starting_model_group.workspace = workspace
        self._mesh_octree.workspace = workspace
        self._starting_inclination_group.workspace = workspace
        self._starting_declination_group.workspace = workspace
        self._reference_inclination_group.workspace = workspace
        self._reference_declination_group.workspace = workspace
        self._upper_bound_group.workspace = workspace
        self._lower_bound_group.workspace = workspace

    @property
    def write(self):
        """"""
        return self._write

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
        if self._run_params is None:
            warnings.warn("Input file must be written before running.")
            return

        self.run(self._run_params)
        self.trigger.button_style = ""

    def inversion_type_observer(self, _):
        """
        Change the application on change of system
        """
        if self.inversion_type.value == "magnetic vector" and not isinstance(
            self.params, MagneticVectorParams
        ):
            self._param_class = MagneticVectorParams

        elif self.inversion_type.value == "magnetic scalar" and not isinstance(
            self.params, MagneticScalarParams
        ):
            self._param_class = MagneticScalarParams
        elif self.inversion_type.value == "gravity" and not isinstance(
            self.params, GravityParams
        ):
            self._param_class = GravityParams

        self.params = self._param_class(validate=False, verbose=False)
        self.ga_group_name.value = self.params.out_group

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

        def checkbox_setter(caller):
            channel = caller["owner"]
            data_widget = getattr(self, f"{channel.header}_group")
            if not channel.value:
                data_widget.children[1].value = None

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
                data_channel_options[key].children[0].header = key
                data_channel_options[key].children[0].observe(
                    checkbox_setter, names="value"
                )
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

        self.data_channel_choices.value = inversion_defaults()["component"][
            self.inversion_type.value
        ]
        self.data_channel_choices.data_channel_options = data_channel_options
        self.update_data_channel_options()
        self.data_channel_panel.children = [
            self.data_channel_choices,
            data_channel_options[self.data_channel_choices.value],
        ]
        self.write.button_style = "warning"
        self._run_params = None
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

        self.params.validate = True

    def object_observer(self, _):
        """ """
        self.resolution.indices = None

        if self.workspace.get_entity(self.objects.value)[0] is None:
            return

        self.update_data_list(None)
        self.update_data_channel_options()
        self.sensor.update_data_list(None)
        self.inversion_type_observer(None)
        self.write.button_style = "warning"
        self._run_params = None
        self.trigger.button_style = "danger"

    def update_data_channel_options(self):
        if getattr(self.data_channel_choices, "data_channel_options", None) is None:
            return

        obj, _ = self.get_selected_entities()
        if obj is not None:
            children_list = {child.uid: child.name for child in obj.children}
            ordered = OrderedDict(sorted(children_list.items(), key=lambda t: t[1]))
            options = [
                [name, uid]
                for uid, name in ordered.items()
                if "visual parameter" not in name.lower()
            ]
        else:
            options = []

        for channel_option in self.data_channel_choices.data_channel_options.values():
            channel_option.children[1].options = [["", None]] + options
            channel_option.children[3].children[1].options = [["", None]] + options

    def data_channel_choices_observer(self, _):
        if hasattr(
            self.data_channel_choices, "data_channel_options"
        ) and self.data_channel_choices.value in (
            self.data_channel_choices.data_channel_options
        ):
            data_widget = self.data_channel_choices.data_channel_options[
                self.data_channel_choices.value
            ]
            self.data_channel_panel.children = [self.data_channel_choices, data_widget]
            obj, data_list = self.get_selected_entities()
            if (
                obj is not None
                and data_list is not None
                and data_widget.children[1].value is None
            ):
                options = [[data.name, data.uid] for data in data_list]
                data_widget.children[1].value = find_value(
                    options, [self.data_channel_choices.value]
                )

            self.plotting_data = data_widget.children[1].value
            self.refresh.value = False
            self.refresh.value = True

        self.write.button_style = "warning"
        self._run_params = None
        self.trigger.button_style = "danger"

    def write_trigger(self, _):

        # Widgets values populate params dictionary
        param_dict = {}
        for key in self.__dict__:
            try:
                if isinstance(getattr(self, key), Widget) and hasattr(self.params, key):
                    value = getattr(self, key).value
                    if key[0] == "_":
                        key = key[1:]

                    if (
                        isinstance(value, uuid.UUID)
                        and self.workspace.get_entity(value)[0] is not None
                    ):
                        value = self.workspace.get_entity(value)[0]

                    param_dict[key] = value

            except AttributeError:
                continue

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
                        if d is not None and new_workspace.get_entity(d.uid)[0] is None:
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
                            if (
                                isinstance(d, Data)
                                and new_workspace.get_entity(d.uid)[0] is None
                            ):
                                d.copy(parent=new_obj)

            new_obj = new_workspace.get_entity(self.objects.value)
            if len(new_obj) == 0 or new_obj[0] is None:
                print("An object with data must be selected to write the input file.")
                return

            new_obj = new_obj[0]
            for key in self.data_channel_choices.options:
                if not self.forward_only.value:
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
                    else:
                        param_dict[f"{key}_channel_bool"] = True

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
                    if isinstance(attr, TopographyOptions):
                        sub_keys = [attr.identifier + "_object", attr.identifier]
                        attr = self
                    elif isinstance(attr, ModelOptions):
                        sub_keys = [attr.identifier]
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
            self._run_params = self.params.__class__(**param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=self.export_directory.selected_path,
            )

        self.write.button_style = ""
        self.trigger.button_style = "success"

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

    def file_browser_change(self, _):
        """
        Change the target h5file
        """
        if not self.file_browser._select.disabled:  # pylint: disable=protected-access
            _, extension = path.splitext(self.file_browser.selected)

            if isinstance(self.geoh5, Workspace):
                self.geoh5.close()

            if extension == ".json" and getattr(self, "_param_class", None) is not None:

                # Read the inversion type first...
                with open(self.file_browser.selected, encoding="utf8") as f:
                    data = json.load(f)

                if data["inversion_type"] == "gravity":
                    self._param_class = GravityParams
                elif data["inversion_type"] == "magnetic vector":
                    self._param_class = MagneticVectorParams
                elif data["inversion_type"] == "magnetic scalar":
                    self._param_class = MagneticScalarParams

                self.params = getattr(self, "_param_class")(
                    InputFile.read_ui_json(self.file_browser.selected)
                )
                self.params.geoh5.open(mode="r")

                params = self.params.to_dict(ui_json_format=False)
                if params["resolution"] is None:
                    params["resolution"] = 0

                self.refresh.value = False
                self.__populate__(**params)
                self.refresh.value = True

            elif extension == ".geoh5":
                self.h5file = self.file_browser.selected


class SensorOptions(ObjectDataSelection):
    """
    Define the receiver spatial parameters
    """

    _options = None
    params_keys = [
        "receivers_offset_z",
        "z_from_topo",
        "receivers_radar_drape",
    ]

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
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
                    self._receivers_offset_z,
                    self._receivers_radar_drape,
                ]
            )

        return self._main

    @property
    def receivers_radar_drape(self):
        return self._receivers_radar_drape

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

    _object_types = (Octree,)
    params_keys = [
        "mesh",
    ]

    def __init__(self, **kwargs):
        self._mesh = self.objects
        self._main = VBox([self.objects])

        super().__init__(**kwargs)

    @property
    def main(self):
        return self._main

    @property
    def mesh(self):
        return self._mesh


class ModelOptions(ObjectDataSelection):
    """
    Widgets for the selection of model options
    """

    def __init__(self, identifier: str = None, **kwargs):
        self._units = "Units"
        self._identifier = identifier
        self._object_types = (Octree,)
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
        self.selection_widget = self.data
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
