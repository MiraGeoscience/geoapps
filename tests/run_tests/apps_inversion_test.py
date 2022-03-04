#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from os import path
from uuid import UUID

from geoh5py.ui_json.input_file import InputFile
from geoh5py.workspace import Workspace
from ipywidgets import Widget

from geoapps.drivers.direct_current_inversion import DirectCurrentParams
from geoapps.drivers.induced_polarization_inversion import InducedPolarizationParams
from geoapps.drivers.magnetic_vector_inversion import MagneticVectorParams
from geoapps.inversion.dcip_inversion_app import InversionApp as DCInversionApp
from geoapps.inversion.pf_inversion_app import InversionApp as MagInversionApp

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

project = "./FlinFlon.geoh5"

geoh5 = Workspace(project)

project_dcip = "./FlinFlon_dcip.geoh5"


def test_mag_inversion(tmp_path):
    """Tests the jupyter application for mag-mvi"""
    ws = Workspace(project)
    new_geoh5 = Workspace(path.join(tmp_path, "invtest.geoh5"))

    new_topo = ws.get_entity(UUID("ab3c2083-6ea8-4d31-9230-7aad3ec09525"))[0].copy(
        parent=new_geoh5
    )
    new_obj = ws.get_entity(UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[0].copy(
        parent=new_geoh5
    )
    topo_val = new_topo.add_data({"elev": {"values": new_topo.vertices[:, 2]}})
    changes = {
        "data_object": new_obj.uid,
        "tmi_channel": UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
        "inducing_field_inclination": 35,
        "topography_object": new_topo.uid,
        "topography": topo_val.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
    }
    side_effects = {"starting_inclination": 35}
    app = MagInversionApp(geoh5=project, plot_result=False)
    app.geoh5 = new_geoh5

    assert (
        len(app._lower_bound_group.objects.options) == 2
    ), "Lower bound group did not reset properly on workspace change."

    assert (
        len(app._upper_bound_group.objects.options) == 2
    ), "Upper bound group did not reset properly on workspace change."

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write_trigger(None)
    ifile = InputFile.read_ui_json(app.params.input_file.path_name)
    params_reload = MagneticVectorParams(ifile)
    objs = params_reload.geoh5.list_entities_name
    check_objs = [
        new_obj.uid,
        UUID("{44822654-b6ae-45b0-8886-2d845f80f422}"),
        new_topo.uid,
        topo_val.uid,
    ]
    for o in check_objs:
        assert o in objs.keys()

    for param, value in changes.items():
        assert (
            getattr(params_reload, param) == value
        ), f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        assert (
            getattr(params_reload, param) == value
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

        if "Model" in getattr(app, "_" + group + "_group").options.options:
            getattr(app, "_" + group + "_group").objects.value = new_topo.uid
            setattr(app, group, new_topo.children[1].uid)
            assert (
                getattr(app, "_" + group + "_group").options.value == "Model"
            ), f"Property group {group} did not reset to 'Model'"
        #
        # setattr(app, group, 1.)


def test_dc_inversion(tmp_path):
    """Tests the jupyter application for dc inversion"""
    ws = Workspace(project_dcip)
    new_geoh5 = Workspace(path.join(tmp_path, "invtest.geoh5"))
    new_topo = ws.get_entity(UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"))[0].copy(
        parent=new_geoh5
    )
    # dc object
    currents = ws.get_entity(UUID("{c2403ce5-ccfd-4d2f-9ffd-3867154cb871}"))[0]
    currents.copy(parent=new_geoh5)
    changes = {
        "topography_object": new_topo.uid,
        "z_from_topo": False,
        "forward_only": False,
        "starting_model": 0.01,
    }
    side_effects = {}
    app = DCInversionApp(geoh5=project_dcip, plot_result=False)
    app.geoh5 = new_geoh5

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write_trigger(None)
    ifile = InputFile.read_ui_json(app.params.input_file.path_name)
    params_reload = DirectCurrentParams(ifile)

    for param, value in changes.items():
        assert (
            getattr(params_reload, param) == value
        ), f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        assert (
            getattr(params_reload, param) == value
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

        if "Model" in getattr(app, "_" + group + "_group").options.options:
            getattr(app, "_" + group + "_group").objects.value = new_topo.uid
            setattr(app, group, new_topo.children[1].uid)
            assert (
                getattr(app, "_" + group + "_group").options.value == "Model"
            ), f"Property group {group} did not reset to 'Model'"


def test_ip_inversion(tmp_path):
    """Tests the jupyter application for dc inversion"""
    ws = Workspace(project_dcip)
    new_geoh5 = Workspace(path.join(tmp_path, "invtest.geoh5"))
    new_topo = ws.get_entity(UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}"))[0].copy(
        parent=new_geoh5
    )
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
        "starting_model": 0.01,
        "inversion_type": "induced polarization",
        "chargeability_channel": UUID("502e7256-aafa-4016-969f-5cc3a4f27315"),
        "conductivity_model_object": UUID("da109284-aa8c-4824-a647-29951109b058"),
        "conductivity_model": UUID("d8846bc7-4c2f-4ced-bbf6-e0ebafd76826"),
    }
    side_effects = {"starting_model": 1e-4}
    app = DCInversionApp(geoh5=project_dcip, plot_result=False)
    app.geoh5 = new_geoh5

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.write.click()
    ifile = InputFile.read_ui_json(app.params.input_file.path_name)
    params_reload = InducedPolarizationParams(ifile)

    for param, value in changes.items():
        if param not in side_effects.keys():
            assert (
                getattr(params_reload, param) == value
            ), f"Parameter {param} not saved and loaded correctly."

    for param, value in side_effects.items():
        assert (
            getattr(params_reload, param) == value
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

        if "Model" in getattr(app, "_" + group + "_group").options.options:
            getattr(app, "_" + group + "_group").objects.value = new_topo.uid
            setattr(app, group, new_topo.children[1].uid)
            assert (
                getattr(app, "_" + group + "_group").options.value == "Model"
            ), f"Property group {group} did not reset to 'Model'"
