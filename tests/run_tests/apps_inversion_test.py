#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0212

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from dash._callback_context import context_value
from dash._utils import AttributeDict
from geoh5py.shared import Entity
from geoh5py.ui_json.input_file import InputFile
from geoh5py.workspace import Workspace
from ipywidgets import Widget

from geoapps.inversion.airborne_electromagnetics.application import InversionApp
from geoapps.inversion.electricals.application import InversionApp as DCInversionApp
from geoapps.inversion.electricals.direct_current.three_dimensions import (
    DirectCurrent3DParams,
)
from geoapps.inversion.electricals.induced_polarization.three_dimensions import (
    InducedPolarization3DParams,
)
from geoapps.inversion.potential_fields.magnetic_vector.application import (
    MagneticVectorApp,
)

from .. import PROJECT, PROJECT_DCIP

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)


def test_mag_inversion(tmp_path: Path):
    """Tests the jupyter application for mag-mvi"""
    temp_workspace = tmp_path / "contour.geoh5"

    with Workspace(PROJECT) as ws:
        with Workspace(temp_workspace) as new_geoh5:
            data_object = ws.get_entity(UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[
                0
            ]
            data_object.copy(parent=new_geoh5, copy_children=True)

            mesh = ws.get_entity(UUID("{a8f3b369-10bd-4ca8-8bd6-2d2595bddbdf}"))[0]
            mesh.copy(parent=new_geoh5, copy_children=True)

            topography_object = ws.get_entity(
                UUID("ab3c2083-6ea8-4d31-9230-7aad3ec09525")
            )[0]
            topography_object.copy(parent=new_geoh5, copy_children=True)

            app = MagneticVectorApp(
                geoh5=new_geoh5,
                output_path=str(tmp_path),
                data_object=data_object,
                mesh=mesh,
                topography_object=topography_object,
            )

            monitoring_directory = str(tmp_path)
            full_components = {
                "tmi": {
                    "channel_bool": True,
                    "channel": "{44822654-b6ae-45b0-8886-2d845f80f422}",
                    "uncertainty_type": "Floor",
                    "uncertainty_floor": 1.0,
                    "uncertainty_channel": None,
                }
            }

            context_value.set(
                AttributeDict(
                    **{
                        "triggered_inputs": [
                            {"prop_id": "write_input.n_clicks", "value": 1}
                        ]
                    }
                )
            )

            # Test export
            app.write_trigger(
                n_clicks=0,
                live_link=[False],
                data_object=str(app.params.data_object.uid),
                full_components=full_components,
                resolution=app.params.resolution,
                window_center_x=app.params.window_center_x,
                window_center_y=app.params.window_center_y,
                window_width=app.params.window_width,
                window_height=app.params.window_height,
                fix_aspect_ratio=[app.params.fix_aspect_ratio],
                colorbar=[app.params.colorbar],
                topography_object=str(app.params.topography_object.uid),
                topography=app.params.topography,
                z_from_topo=[app.params.z_from_topo],
                receivers_offset_z=app.params.receivers_offset_z,
                receivers_radar_drape=app.params.receivers_radar_drape,
                forward_only=[app.params.forward_only],
                starting_model_options="Constant",
                starting_model_data=None,
                starting_model_const=4.5,
                mesh=str(app.params.mesh.uid),
                reference_model_options="Model",
                reference_model_data="{44822654-b6ae-45b0-8886-2d845f80f422}",
                reference_model_const=None,
                alpha_s=app.params.alpha_s,
                alpha_x=app.params.alpha_x,
                alpha_y=app.params.alpha_y,
                alpha_z=app.params.alpha_z,
                s_norm=app.params.s_norm,
                x_norm=app.params.x_norm,
                y_norm=app.params.y_norm,
                z_norm=app.params.z_norm,
                lower_bound_options="Model",
                lower_bound_data="{44822654-b6ae-45b0-8886-2d845f80f422}",
                lower_bound_const=None,
                upper_bound_options="Constant",
                upper_bound_data=None,
                upper_bound_const=3.5,
                detrend_type=app.params.detrend_type,
                detrend_order=app.params.detrend_order,
                ignore_values=app.params.ignore_values,
                max_global_iterations=app.params.max_global_iterations,
                max_irls_iterations=app.params.max_irls_iterations,
                coolingRate=app.params.coolingRate,
                coolingFactor=app.params.coolingFactor,
                chi_factor=app.params.chi_factor,
                initial_beta_ratio=app.params.initial_beta_ratio,
                max_cg_iterations=app.params.max_cg_iterations,
                tol_cg=app.params.tol_cg,
                n_cpu=app.params.n_cpu,
                store_sensitivities=app.params.store_sensitivities,
                tile_spatial=app.params.tile_spatial,
                out_group=app.params.out_group,
                monitoring_directory=monitoring_directory,
                inducing_field_strength=app.params.inducing_field_strength,
                inducing_field_inclination=app.params.inducing_field_inclination,
                inducing_field_declination=app.params.inducing_field_declination,
                starting_inclination_options="Constant",
                starting_inclination_data=None,
                starting_inclination_const=5.5,
                reference_inclination_options="None",
                reference_inclination_data=None,
                reference_inclination_const=None,
                starting_declination_options="Constant",
                starting_declination_data=None,
                starting_declination_const=1.5,
                reference_declination_options="None",
                reference_declination_data=None,
                reference_declination_const=None,
            )

    filename = next(tmp_path.glob("VectorInversion_*.geoh5"))
    with Workspace(filename) as workspace:
        assert len(workspace.get_entity(app.params.data_object)) == 1
        assert len(workspace.get_entity(app.params.mesh)) == 1
        assert len(workspace.get_entity(app.params.topography_object)) == 1


def test_dc_inversion(tmp_path: Path):
    """Tests the jupyter application for dc inversion"""
    with Workspace(PROJECT_DCIP) as ws:
        with Workspace(tmp_path / "invtest.geoh5") as new_geoh5:
            new_topo = ws.get_entity(UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"))[
                0
            ].copy(parent=new_geoh5)
            # dc object
            currents = ws.get_entity(UUID("{c2403ce5-ccfd-4d2f-9ffd-3867154cb871}"))[0]
            currents.copy(parent=new_geoh5)
            ws.get_entity(UUID("{da109284-aa8c-4824-a647-29951109b058}"))[0].copy(
                parent=new_geoh5
            )
    changes = {
        "topography_object": new_topo.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
    }
    side_effects = {}
    app = DCInversionApp(geoh5=str(PROJECT_DCIP), plot_result=False)
    app.geoh5 = str(tmp_path / "invtest.geoh5")

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write_trigger(None)
    app.write_trigger(None)  # Check that this can run more than once
    ifile = InputFile.read_ui_json(getattr(app, "_run_params").input_file.path_name)

    params_reload = DirectCurrent3DParams(ifile)

    for param, value in changes.items():
        p_value = getattr(params_reload, param)
        p_value = p_value.uid if isinstance(p_value, Entity) else p_value
        assert p_value == value, f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        p_value = getattr(params_reload, param)
        p_value = p_value.uid if isinstance(p_value, Entity) else p_value
        assert (
            p_value == value
        ), f"Side effect parameter {param} not saved and loaded correctly."

    # Test the groups
    groups = [
        "topography",
        "reference_model",
        "starting_model",
        "upper_bound",
        "lower_bound",
    ]

    for group in groups:
        if "Constant" in getattr(app, "_" + group + "_group").options.options:
            setattr(app, group, 1.0)
            assert (
                getattr(app, "_" + group + "_group").options.value == "Constant"
            ), f"Property group {group} did not reset to 'Constant'"

        if "None" in getattr(app, "_" + group + "_group").options.options:
            setattr(app, group, None)
            assert (
                getattr(app, "_" + group + "_group").options.value == "None"
            ), f"Property group {group} did not reset to 'None'"


def test_ip_inversion(tmp_path: Path):
    """Tests the jupyter application for dc inversion"""
    with Workspace(PROJECT_DCIP) as ws:
        with Workspace(tmp_path / "invtest.geoh5") as new_geoh5:
            new_topo = ws.get_entity(UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"))[
                0
            ].copy(parent=new_geoh5)
            # Conductivity mesh + model
            ws.get_entity(UUID("{da109284-aa8c-4824-a647-29951109b058}"))[0].copy(
                parent=new_geoh5
            )

            # dc object
            currents = ws.get_entity(UUID("{c2403ce5-ccfd-4d2f-9ffd-3867154cb871}"))[0]
            currents.copy(parent=new_geoh5)

    changes = {
        "topography_object": new_topo.uid,
        "z_from_topo": False,
        "forward_only": False,
        "mesh": UUID("{da109284-aa8c-4824-a647-29951109b058}"),
        "inversion_type": "induced polarization 3d",
        "chargeability_channel": UUID("502e7256-aafa-4016-969f-5cc3a4f27315"),
        "conductivity_model": UUID("d8846bc7-4c2f-4ced-bbf6-e0ebafd76826"),
    }
    side_effects = {"starting_model": 1e-4}
    app = DCInversionApp(geoh5=str(PROJECT_DCIP), plot_result=False)
    app.mesh.value = None
    with new_geoh5.open(mode="r"):
        app.geoh5 = new_geoh5

        for param, value in changes.items():
            if isinstance(getattr(app, param), Widget):
                getattr(app, param).value = value
            else:
                setattr(app, param, value)

        app.write_trigger(None)
    ifile = InputFile.read_ui_json(getattr(app, "_run_params").input_file.path_name)
    params_reload = InducedPolarization3DParams(ifile)

    for param, value in changes.items():
        p_value = getattr(params_reload, param)
        p_value = p_value.uid if isinstance(p_value, Entity) else p_value
        assert p_value == value, f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        p_value = getattr(params_reload, param)
        p_value = p_value.uid if isinstance(p_value, Entity) else p_value
        assert (
            p_value == value
        ), f"Side effect parameter {param} not saved and loaded correctly."

    groups = [
        "topography",
        "reference_model",
        "starting_model",
        "conductivity_model",
        "upper_bound",
        "lower_bound",
    ]

    for group in groups:
        if "Constant" in getattr(app, "_" + group + "_group").options.options:
            setattr(app, group, 1.0)
            assert (
                getattr(app, "_" + group + "_group").options.value == "Constant"
            ), f"Property group {group} did not reset to 'Constant'"

        if "None" in getattr(app, "_" + group + "_group").options.options:
            setattr(app, group, None)
            assert (
                getattr(app, "_" + group + "_group").options.value == "None"
            ), f"Property group {group} did not reset to 'None'"


def test_em1d_inversion(tmp_path: Path):
    """Tests the jupyter application for em1d inversion."""
    with Workspace(PROJECT) as ws:
        with Workspace(tmp_path / "invtest.geoh5") as new_geoh5:
            new_obj = ws.get_entity(UUID("{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}"))[
                0
            ].copy(parent=new_geoh5)

            prop_group_uid = new_obj.property_groups[0].uid

    changes = {
        "objects": new_obj.uid,
        "data": prop_group_uid,
    }
    side_effects = {"system": "VTEM (2007)"}
    app = InversionApp(geoh5=PROJECT, plot_result=False)
    app.workspace = new_geoh5

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    for key, value in side_effects.items():
        assert getattr(app, key).value == value, f"Failed to change {key} with {value}."

    app.topography.options.value = "Constant"
    app.write_trigger(None)
