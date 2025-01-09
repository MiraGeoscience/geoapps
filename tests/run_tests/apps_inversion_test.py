# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# pylint: disable=W0212

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from dash._callback_context import context_value
from dash._utils import AttributeDict
from geoh5py.objects import Points
from geoh5py.shared import Entity
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json.input_file import InputFile
from geoh5py.workspace import Workspace
from ipywidgets import Widget
from simpeg_drivers.electricals.direct_current.three_dimensions import (
    DirectCurrent3DParams,
)
from simpeg_drivers.electricals.induced_polarization.three_dimensions import (
    InducedPolarization3DParams,
)

from geoapps.inversion.electricals.application import InversionApp as DCInversionApp
from geoapps.inversion.electromagnetics.application import (
    InversionApp as EMInversionApp,
)
from geoapps.inversion.potential_fields.magnetic_vector.application import (
    MagneticVectorApp,
)
from tests import (  # pylint: disable=no-name-in-module
    PROJECT,
    PROJECT_DCIP,
    PROJECT_TEM,
)

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)


def test_mag_inversion(tmp_path: Path):
    """Tests the dash application for mag-mvi"""
    temp_workspace = tmp_path / "mag_inversion.geoh5"

    with Workspace(PROJECT) as ws:
        with Workspace(temp_workspace) as new_geoh5:
            data_object = ws.get_entity(UUID("{7aaf00be-adbf-4540-8333-8ac2c2a3c31a}"))[
                0
            ]
            data_object.copy(parent=new_geoh5, copy_children=True)

            mesh = ws.get_entity(UUID("{f6b08e3b-9a85-45ab-a487-4700e3ca1917}"))[0]
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
                    "channel": "{a342e416-946a-4162-9604-6807ccb06073}",
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
                data_object=str(app.params.data_object.uid),
                full_components=full_components,
                resolution=50.0,
                window_center_x=314600.0,
                window_center_y=6072300.0,
                window_width=1000.0,
                window_height=1500.0,
                topography_object=str(
                    app.params.topography_object.uid  # pylint: disable=no-member
                ),
                topography=app.params.topography,
                z_from_topo=[app.params.z_from_topo],
                receivers_offset_z=app.params.receivers_offset_z,
                receivers_radar_drape=app.params.receivers_radar_drape,
                forward_only=[],
                starting_model_options="Constant",
                starting_model_data=None,
                starting_model_const=4.5,
                mesh=str(app.params.mesh.uid),
                reference_model_options="Model",
                reference_model_data="{44822654-b6ae-45b0-8886-2d845f80f422}",
                reference_model_const=None,
                alpha_s=app.params.alpha_s,
                length_scale_x=app.params.length_scale_x,
                length_scale_y=app.params.length_scale_y,
                length_scale_z=app.params.length_scale_z,
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
                detrend_type="all",
                detrend_order=0,
                ignore_values="",
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
                ga_group=app.params.ga_group,
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

    filename = next(tmp_path.glob("MagneticVectorInversion_*.json"))
    ifile = InputFile.read_ui_json(filename)
    with ifile.data["geoh5"].open():
        data = ifile.data["data_object"]
        assert isinstance(data, Points)
        assert data.n_vertices == 418
        assert ifile.data["mesh"].uid == mesh.uid
        assert ifile.data["topography_object"].uid == topography_object.uid


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
            ws.get_entity(UUID("{eab26a47-6050-4e72-bb95-bd4457b65f47}"))[0].copy(
                parent=new_geoh5
            )
    changes = {
        "topography_object": new_topo.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
        "reference_model": None,
    }
    side_effects = {"reference_model": changes["starting_model"], "alpha_s": 0}
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
        if param == "reference_model":
            continue

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
            ws.get_entity(UUID("{eab26a47-6050-4e72-bb95-bd4457b65f47}"))[0].copy(
                parent=new_geoh5
            )

            # dc object
            currents = ws.get_entity(UUID("{c2403ce5-ccfd-4d2f-9ffd-3867154cb871}"))[0]
            currents.copy(parent=new_geoh5)

    changes = {
        "topography_object": new_topo.uid,
        "z_from_topo": False,
        "forward_only": False,
        "mesh": UUID("{eab26a47-6050-4e72-bb95-bd4457b65f47}"),
        "inversion_type": "induced polarization 3d",
        "chargeability_channel": UUID("502e7256-aafa-4016-969f-5cc3a4f27315"),
        "conductivity_model": UUID("a096af7c-12b1-4fd2-a95c-22611ea924c6"),
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
        if param == "chargeability_channel":
            assert p_value != value and is_uuid(p_value)
        else:
            assert (
                p_value == value
            ), f"Parameter {param} not saved and loaded correctly."

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
    with Workspace(PROJECT_TEM) as ws:
        with Workspace(tmp_path / "invtest.geoh5") as new_geoh5:
            new_obj = ws.get_entity(UUID("34698019-cde6-4b43-8d53-a040b25c989a"))[
                0
            ].copy(parent=new_geoh5)

            data_group_uid = new_obj.find_or_create_property_group("dbdt_z").uid
            uncert_group_uid = new_obj.find_or_create_property_group(
                "dbdt_z_uncert"
            ).uid

    changes = {
        "objects": new_obj.uid,
        "data": data_group_uid,
        "_uncertainties": uncert_group_uid,
    }
    side_effects = {"system": "Airborne TEM Survey"}
    app = EMInversionApp(geoh5=PROJECT_TEM, plot_result=False)
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
