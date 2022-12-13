#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0212

from os import path
from uuid import UUID

from geoh5py.shared import Entity
from geoh5py.ui_json.input_file import InputFile
from geoh5py.ui_json.utils import str2list
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
from geoapps.inversion.potential_fields.application import (
    InversionApp as MagInversionApp,
)

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

project = "./FlinFlon.geoh5"

geoh5 = Workspace(project)

project_dcip = "./FlinFlon_dcip.geoh5"


def test_mag_inversion(tmp_path):
    """Tests the jupyter application for mag-mvi"""
    with Workspace(project) as ws:
        with Workspace(path.join(tmp_path, "invtest.geoh5")) as new_geoh5:
            new_topo = ws.get_entity(UUID("ab3c2083-6ea8-4d31-9230-7aad3ec09525"))[
                0
            ].copy(parent=new_geoh5)
            new_obj = ws.get_entity(UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[
                0
            ].copy(parent=new_geoh5)
            topo_val = new_topo.add_data({"elev": {"values": new_topo.vertices[:, 2]}})
            ws.get_entity(UUID("{a8f3b369-10bd-4ca8-8bd6-2d2595bddbdf}"))[0].copy(
                parent=new_geoh5
            )

    changes = {
        "data_object": new_obj.uid,
        "tmi_channel": UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
        "inducing_field_inclination": 35,
        "topography_object": new_topo.uid,
        "topography": topo_val.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
        "window_width": 100.0,
    }
    side_effects = {"starting_inclination": 35}
    app = MagInversionApp(geoh5=new_geoh5.h5file, plot_result=False)

    assert (
        len(getattr(app, "_lower_bound_group").objects.options) == 2
    ), "Lower bound group did not reset properly on workspace change."

    assert (
        len(getattr(app, "_upper_bound_group").objects.options) == 2
    ), "Upper bound group did not reset properly on workspace change."

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write_trigger(None)
    app.write_trigger(None)  # Check to make sure this can be run twice

    new_app = MagInversionApp(plot_result=False)
    new_app._file_browser.reset(
        path=getattr(app, "_run_params").input_file.path,
        filename=getattr(app, "_run_params").input_file.name,
    )
    new_app._file_browser._apply_selection()
    new_app.file_browser_change(None)

    with new_app.params.geoh5:
        objs = new_app.params.geoh5.list_entities_name
        check_objs = [
            new_obj.uid,
            UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
            new_topo.uid,
            topo_val.uid,
        ]
        for o in check_objs:
            assert o in objs.keys()

        for param, value in changes.items():
            p_value = getattr(new_app.params, param)
            p_value = p_value.uid if isinstance(p_value, Entity) else p_value
            assert p_value == str2list(
                value
            ), f"Parameter {param} not saved and loaded correctly."

        for param, value in side_effects.items():
            p_value = getattr(new_app.params, param)
            p_value = p_value.uid if isinstance(p_value, Entity) else p_value
            assert (
                p_value == value
            ), f"Side effect parameter {param} not saved and loaded correctly."

        # Test the groups
        groups = [
            "topography",
            "reference_model",
            "starting_model",
            "starting_inclination",
            "starting_declination",
            "reference_inclination",
            "reference_declination",
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


def test_dc_inversion(tmp_path):
    """Tests the jupyter application for dc inversion"""
    with Workspace(project_dcip) as ws:
        with Workspace(path.join(tmp_path, "invtest.geoh5")) as new_geoh5:
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
    app = DCInversionApp(geoh5=project_dcip, plot_result=False)
    app.geoh5 = path.join(tmp_path, "invtest.geoh5")

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


def test_ip_inversion(tmp_path):
    """Tests the jupyter application for dc inversion"""
    with Workspace(project_dcip) as ws:
        with Workspace(path.join(tmp_path, "invtest.geoh5")) as new_geoh5:
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
    app = DCInversionApp(geoh5=project_dcip, plot_result=False)
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


def test_em1d_inversion(tmp_path):
    """Tests the jupyter application for em1d inversion."""
    with Workspace(project) as ws:
        with Workspace(path.join(tmp_path, "invtest.geoh5")) as new_geoh5:
            new_obj = ws.get_entity(UUID("{bb208abb-dc1f-4820-9ea9-b8883e5ff2c6}"))[
                0
            ].copy(parent=new_geoh5)

            prop_group_uid = new_obj.property_groups[0].uid

    changes = {
        "objects": new_obj.uid,
        "data": prop_group_uid,
    }
    side_effects = {"system": "VTEM (2007)"}
    app = InversionApp(geoh5=project, plot_result=False)
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
