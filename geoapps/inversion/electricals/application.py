# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import json
import multiprocessing
import os
import uuid
import warnings
from pathlib import Path
from time import time

import numpy as np
from geoh5py.data import Data
from geoh5py.objects import CurrentElectrode, PotentialElectrode
from geoh5py.shared.exceptions import AssociationValidationError
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from simpeg_drivers.electricals.direct_current.three_dimensions.params import (
    DirectCurrent3DParams,
)
from simpeg_drivers.electricals.induced_polarization.three_dimensions.params import (
    InducedPolarization3DParams,
)

from geoapps.base.application import BaseApplication
from geoapps.base.plot import PlotSelection2D
from geoapps.base.selection import ObjectDataSelection, TopographyOptions
from geoapps.inversion.components.preprocessing import preprocess_data
from geoapps.inversion.electricals.direct_current.three_dimensions.constants import (
    app_initializer,
)
from geoapps.inversion.potential_fields.application import (
    MeshOctreeOptions,
    ModelOptions,
)
from geoapps.shared_utils.utils import DrapeOptions, WindowOptions
from geoapps.utils import warn_module_not_found
from geoapps.utils.list import find_value


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


def inversion_defaults():
    """
    Get defaults for DCIP inversion
    """
    return {
        "units": {
            "direct current 3d": "S/m",
            "induced polarization 3d": "V/V",
        },
        "property": {
            "direct current 3d": "conductivity",
            "induced polarization 3d": "chargeability",
        },
        "reference_value": {
            "direct current 3d": 1e-1,
            "induced polarization 3d": 0.0,
        },
        "starting_value": {
            "direct current 3d": 1e-1,
            "induced polarization 3d": 1e-4,
        },
        "component": {
            "direct current 3d": "potential",
            "induced polarization 3d": "chargeability",
        },
    }


class InversionApp(PlotSelection2D):
    """
    Application for the inversion of potential field data using SimPEG
    """

    _param_class = DirectCurrent3DParams
    _select_multiple = True
    _add_groups = False
    _sensor = None
    _lines = None
    _topography = None
    _run_params = None
    _object_types = (PotentialElectrode,)
    _exclusion_types = (CurrentElectrode,)
    inversion_parameters = None

    def __init__(self, ui_json=None, plot_result=True, **kwargs):
        if ui_json is not None and Path(ui_json.path).exists():
            self.params = self._param_class(ui_json)
        else:
            app_initializer.update(kwargs)

            try:
                self.params = self._param_class(**app_initializer)

            except AssociationValidationError:
                for key, value in app_initializer.items():
                    if isinstance(value, uuid.UUID):
                        app_initializer[key] = None

                self.params = self._param_class(**app_initializer)

            extras = {
                key: value
                for key, value in app_initializer.items()
                if key not in self.params.param_names
            }
            self._app_initializer = extras

        self.data_object = self.objects
        self.defaults.update(self.params.to_dict())

        self._data_count = (Label("Data Count: 0"),)
        self._forward_only = Checkbox(
            value=False,
            description="Forward only",
        )
        self._inversion_type = Dropdown(
            options=["direct current 3d", "induced polarization 3d"],
            description="inversion Type:",
        )
        self._write = Button(
            description="Write input",
            button_style="warning",
            icon="check",
        )
        self.defaults.update(self.params.to_dict())
        self._ga_group_name = widgets.Text(
            value="Inversion_", description="Save as:", disabled=False
        )
        self._chi_factor = FloatText(
            value=1, description="Target misfit", disabled=False
        )

        self._mesh_octree = MeshOctreeOptions(workspace=self.defaults.get("geoh5"))
        self.mesh = self._mesh_octree.mesh

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
        self._store_sensitivities = Dropdown(
            options=["ram", "disk"], description="Storage device:", value="disk"
        )
        self._max_global_iterations = IntText(
            value=10, description="Max beta Iterations"
        )
        self._max_irls_iterations = IntText(value=10, description="Max IRLS iterations")
        self._coolingRate = IntText(value=1, description="Iterations per beta")
        self._coolingFactor = FloatText(value=2, description="Beta cooling factor")
        self._max_cg_iterations = IntText(value=30, description="Max CG Iterations")
        self._sens_wts_threshold = FloatText(
            value=0.001, description="Threshold sensitivity weights"
        )
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
                self._sens_wts_threshold,
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
        self._starting_model_group.options.value = "Constant"
        self._conductivity_model_group = ModelOptions(
            "conductivity_model",
            add_xyz=False,
            objects=self._mesh_octree.mesh,
            **self.defaults,
        )
        self._conductivity_model_group.options.options = ["Model"]
        self._reference_model_group = ModelOptions(
            "reference_model",
            add_xyz=False,
            objects=self._mesh_octree.mesh,
            **self.defaults,
        )
        self._reference_model_group.options.observe(self.update_ref)
        self._topography_group = TopographyOptions(add_xyz=False, **self.defaults)
        self._topography_group.identifier = "topography"
        self._sensor = SensorOptions(
            objects=self._objects,
            object_types=self._object_types,
            exclusion_types=self._exclusion_types,
            add_xyz=False,
        )
        self._alpha_s = widgets.FloatText(
            min=0,
            value=1,
            description="Reference Model (s)",
        )
        self._length_scale_x = widgets.FloatText(
            min=0,
            value=1,
            description="EW-gradient (x)",
        )
        self._length_scale_y = widgets.FloatText(
            min=0,
            value=1,
            description="NS-gradient (y)",
        )
        self._length_scale_z = widgets.FloatText(
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
                self._length_scale_x,
                self._length_scale_y,
                self._length_scale_z,
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
            "conductivity model": VBox(
                [
                    self._conductivity_model_group.main,
                ]
            ),
            "starting model": VBox(
                [
                    self._starting_model_group.main,
                ]
            ),
            "mesh": self._mesh_octree.main,
            "reference model": VBox(
                [
                    self._reference_model_group.main,
                ]
            ),
            "regularization": HBox([self._alphas, self._norms]),
            "upper-lower bounds": self.bound_panel,
            "ignore values": VBox([self._ignore_values]),
            "optimization": self._optimization,
        }
        self.option_choices = widgets.Dropdown(
            options=list(self.inversion_options)[1:],
            value=list(self.inversion_options)[1],
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
        self.plotting = None
        self._starting_channel = None
        self._mesh = self._mesh_octree.mesh
        self._detrend_type = None
        self._detrend_order = None
        self._initial_beta_options = None
        super().__init__(plot_result=plot_result, **self.defaults)

        self.write.on_click(self.write_trigger)

    @property
    def alphas(self):
        return self._alphas

    @property
    def alpha_s(self):
        return self._alpha_s

    @property
    def length_scale_x(self):
        return self._length_scale_x

    @property
    def length_scale_y(self):
        return self._length_scale_y

    @property
    def length_scale_z(self):
        return self._length_scale_z

    # @property
    # def initial_beta(self):
    #     return self._initial_beta

    @property
    def coolingRate(self):
        return self._coolingRate

    @property
    def coolingFactor(self):
        return self._coolingFactor

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
    def conductivity_model(self):
        if self._conductivity_model_group.options.value == "Model":
            return self._conductivity_model_group.data.value
        elif self._conductivity_model_group.options.value == "Constant":
            return self._conductivity_model_group.constant.value
        else:
            return None

    @conductivity_model.setter
    def conductivity_model(self, value):
        if isinstance(value, float):
            self._conductivity_model_group.options.value = "Constant"
            self._conductivity_model_group.constant.value = value
        else:
            self._conductivity_model_group.data.value = value

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
    def sens_wts_threshold(self):
        return self._sens_wts_threshold

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

    # @property
    # def lines(self):
    #     if getattr(self, "_lines", None) is None:
    #         self._lines = LineOptions(workspace=self._workspace, objects=self._objects)
    #         self.lines.lines.observe(self.update_selection, names="value")
    #     return self._lines

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
            self.resolution.disabled = True
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
        assert isinstance(workspace, Workspace), (
            f"Workspace must be of class {Workspace}"
        )
        self.base_workspace_changes(workspace)
        self.update_objects_list()
        # self.lines.workspace = workspace
        self.sensor.workspace = workspace
        self._topography_group.workspace = workspace
        self._reference_model_group.workspace = workspace
        self._starting_model_group.workspace = workspace
        self._conductivity_model_group.workspace = workspace
        self._mesh_octree.workspace = workspace
        self.plotting_data = None

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

    def inversion_option_change(self, _):
        self.main.children[4].children[2].children = [
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
        params = self.params.to_dict()
        if self.inversion_type.value == "direct current 3d" and not isinstance(
            self.params, DirectCurrent3DParams
        ):
            self._param_class = DirectCurrent3DParams
            params["inversion_type"] = "direct current 3d"
            params["out_group"] = "DCInversion"
            self.option_choices.options = list(self.inversion_options)[1:]

        elif self.inversion_type.value == "induced polarization 3d" and not isinstance(
            self.params, InducedPolarization3DParams
        ):
            self._param_class = InducedPolarization3DParams
            params["inversion_type"] = "induced polarization 3d"
            params["out_group"] = "ChargeabilityInversion"
            self.option_choices.options = list(self.inversion_options)

        self.params = self._param_class(
            # validator_opts={"ignore_requirements": True}
        )

        if self.inversion_type.value in ["direct current 3d"]:
            data_type_list = ["potential"]
        else:
            data_type_list = ["chargeability"]

        if getattr(self.params, "_out_group", None) is not None:
            self.ga_group_name.value = self.params.out_group.name
        else:
            self.ga_group_name.value = (
                self.params.inversion_type.capitalize() + "Inversion"
            )

        flag = self.inversion_type.value
        self._reference_model_group.units = inversion_defaults()["units"][flag]
        self._reference_model_group.options.value = "Constant"
        self._reference_model_group.constant.value = inversion_defaults()[
            "reference_value"
        ][flag]
        self._reference_model_group.description.value = (
            "Reference " + inversion_defaults()["property"][flag]
        )
        self._starting_model_group.units = inversion_defaults()["units"][flag]
        self._starting_model_group.options.value = "Constant"
        self._starting_model_group.constant.value = inversion_defaults()[
            "starting_value"
        ][flag]
        self._starting_model_group.description.value = (
            "Starting " + inversion_defaults()["property"][flag]
        )
        data_channel_options = {}
        self.data_channel_choices.options = data_type_list
        obj, _ = self.get_selected_entities()

        if obj is not None:
            options = [
                [name, obj.get_data(name)[0].uid]
                for name in sorted(obj.get_data_list())
                if "visual parameter" not in name.lower()
            ]
        else:
            options = []

        for key in data_type_list:

            def channel_setter(caller):
                channel = caller["owner"]
                data_widget = getattr(self, f"{channel.header}_group")
                entity = self.workspace.get_entity(self.objects.value)[0]

                if entity is None:
                    return

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
                            np.percentile(np.abs(values[~np.isnan(values)]), 10), 5
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

                def value_setter(self, key, value):
                    """Assign value or channel"""
                    if isinstance(value, float):
                        getattr(self, key + "_floor").value = value
                    else:
                        getattr(self, key + "_channel").value = (
                            uuid.UUID(value) if isinstance(value, str) else value
                        )

                setattr(InversionApp, f"{key}_uncertainty", (value_setter))
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

        self.data_channel_choices.value = inversion_defaults()["component"][
            self.inversion_type.value
        ]
        self.data_channel_choices.data_channel_options = data_channel_options
        self.data_channel_panel.children = [
            self.data_channel_choices,
            data_channel_options[self.data_channel_choices.value],
        ]
        self.write.button_style = "warning"
        self._run_params = None
        self.trigger.button_style = "danger"

        if self.inversion_type.value == "direct current 3d":
            self._lower_bound_group.options.value = "Constant"
            self._lower_bound_group.constant.value = 1e-5
        else:
            self._lower_bound_group.options.value = "Constant"
            self._lower_bound_group.constant.value = 0.0

    def object_observer(self, _):
        """ """
        self.resolution.indices = None
        obj = self.workspace.get_entity(self.objects.value)[0]
        if obj is None:
            return

        self.update_data_list(None)
        self.sensor.update_data_list(None)
        self.inversion_type_observer(None)
        self.write.button_style = "warning"
        self._run_params = None
        self.trigger.button_style = "danger"

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
        if (
            "induced polarization 3d" in self.inversion_type.value
            and self._conductivity_model_group.data.value is None
        ):
            print(
                "A conductivity model is required for IP inversion. Check your inversion options."
            )
            return

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
            with fetch_active_workspace(self.workspace):
                param_dict["geoh5"] = new_workspace

                for elem in [
                    self,
                    self._mesh_octree,
                    self._topography_group,
                    self._starting_model_group,
                    self._conductivity_model_group,
                    self._reference_model_group,
                    self._lower_bound_group,
                    self._upper_bound_group,
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

                new_obj = new_workspace.get_entity(self.objects.value)[0]

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
                    receiver_drape = self.workspace.get_entity(
                        self.receivers_radar_drape.value
                    )[0]
                else:
                    receiver_drape = None

            for key in self.__dict__:
                if "resolution" in key:
                    continue

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
                        sub_keys = [attr.identifier, attr.identifier + "_object"]
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
            param_dict["geoh5"] = new_workspace

            if param_dict.get("reference_model", None) is None:
                param_dict["reference_model"] = param_dict["starting_model"]
                param_dict["alpha_s"] = 0.0

            window_options = WindowOptions(
                center_x=self.window_center_x.value,
                center_y=self.window_center_y.value,
                width=self.window_width.value,
                height=self.window_height.value,
                azimuth=self.window_azimuth.value,
            )
            drape_options = DrapeOptions(
                topography_object=param_dict["topography_object"],
                topography=param_dict["topography"],
                z_from_topo=param_dict["z_from_topo"],
                receivers_offset_z=param_dict["receivers_offset_z"],
                receivers_radar_drape=receiver_drape,
            )

            # Pre-processing
            update_dict = preprocess_data(
                ws,
                param_dict,
                param_dict["data_object"],
                window_options,
                drape_options=drape_options,
            )
            param_dict.update(update_dict)

            self._run_params = self.params.__class__(**param_dict)
            self._run_params.write_input_file(
                name=temp_geoh5.replace(".geoh5", ".ui.json"),
                path=self.export_directory.selected_path,
            )

        self.write.button_style = ""
        self.trigger.button_style = "success"

    @staticmethod
    def run(params):
        if not isinstance(params, (DirectCurrent3DParams, InducedPolarization3DParams)):
            raise TypeError(
                "Parameter 'inversion_type' must be one of "
                "'direct current 3d' or 'induced polarization 3d'"
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
            extension = Path(self.file_browser.selected).suffix

            if isinstance(self.geoh5, Workspace):
                self.geoh5.close()

            if extension == ".json" and getattr(self, "_param_class", None) is not None:
                # Read the inversion type first...
                with open(self.file_browser.selected, encoding="utf8") as f:
                    data = json.load(f)

                if data["inversion_type"] == "direct current 3d":
                    self._param_class = DirectCurrent3DParams

                elif data["inversion_type"] == "induced polarization 3d":
                    self._param_class = InducedPolarization3DParams

                self.params = self._param_class(
                    InputFile.read_ui_json(self.file_browser.selected)
                )
                self.params.geoh5.open(mode="r")
                self.refresh.value = False
                self.__populate__(**self.params.to_dict())
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
        "receivers_offset_z",
        "z_from_topo",
        "receivers_radar_drape",
    ]

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        self._receivers_offset_z = FloatText(description="dz (+ve up)", value=0.0)
        self._z_from_topo = Checkbox(
            description="Set Z from topo + offsets", value=True
        )
        self.data.description = "Radar (Optional):"
        self._receivers_radar_drape = self.data
        self._offset = None
        super().__init__(**self.defaults)

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.z_from_topo,
                    Label("Offsets"),
                    self._receivers_offset_z,
                    # self._receivers_radar_drape,
                ]
            )

        return self._main

    @property
    def receivers_radar_drape(self):
        return self._receivers_radar_drape

    @property
    def offset(self):
        return self._offset

    @property
    def receivers_offset_z(self):
        return self._receivers_offset_z

    @property
    def z_from_topo(self):
        return self._z_from_topo
