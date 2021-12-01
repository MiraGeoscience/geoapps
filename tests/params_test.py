#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
from copy import deepcopy
from uuid import UUID, uuid4

import numpy as np
import pytest
from geoh5py.workspace import Workspace

from geoapps.io import InputFile, Params
from geoapps.io.DirectCurrent import DirectCurrentParams
from geoapps.io.Gravity import GravityParams
from geoapps.io.InducedPolarization import InducedPolarizationParams
from geoapps.io.MagneticScalar import MagneticScalarParams
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.io.MagneticVector.constants import default_ui_json as MVI_defaults
from geoapps.io.MagneticVector.constants import validations as MVI_validations
from geoapps.io.Octree import OctreeParams
from geoapps.io.PeakFinder import PeakFinderParams
from geoapps.utils.testing import Geoh5Tester

workspace = Workspace("./FlinFlon.geoh5")


def setup_params(tmp, ui, params_class):
    geotest = Geoh5Tester(workspace, tmp, "test.geoh5", ui, params_class)
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("gz_channel", "{6de9177a-8277-4e17-b76c-2b8b05dcf23c}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    return geotest


# def test_inversion_type(tmp_path):
#     geotest = setup_params(tmp_path, MVI_defaults, MagneticVectorParams)
#     geotest.set_param("inversion_type", "nogood")
#     ws, params = geotest.make()
#     assert True


######################  Setup  ###########################

tmpfile = lambda path: os.path.join(path, "test.ui.json")
wrkstr = "FlinFlon.geoh5"
workspace = Workspace(wrkstr)


def tmp_input_file(filepath, idict):
    with open(filepath, "w") as f:
        json.dump(idict, f)


def catch_invalid_generator(
    tmp_path, param, invalid_value, validation_type, workspace=None, parent=None
):

    key_map = {
        "value": "values",
        "type": "types",
        "shape": "shapes",
        "reqs": "reqs",
        "uuid": "uuid",
    }
    pvalidations = MVI_validations[param][key_map[validation_type]]
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    ifile.write_ui_json(MVI_defaults, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    if isinstance(ui[param], dict):
        ui[param]["value"] = invalid_value
        ui[param]["enabled"] = True
        ui[param]["isValue"] = True
    else:
        ui[param] = invalid_value
    ui["geoh5"] = None
    if validation_type == "value":
        err = ValueError
        assertions = [
            "Must be",
            param,
            "value",
            str(invalid_value),
            *(str(v) for v in pvalidations),
        ]

    elif validation_type == "type":
        err = TypeError
        types = set(pvalidations + [type(invalid_value)])
        assertions = ["Must be", param, "type", *(t.__name__ for t in types)]
    elif validation_type == "shape":
        err = ValueError
        shapes = set(pvalidations + [np.array(invalid_value).shape])
        assertions = ["Must be", param, "shape", *(str(s) for s in shapes)]
    elif validation_type == "reqs":
        err = KeyError
        assertions = ["Unsatisfied", param]
        req = pvalidations[0]
        hasval = len(req) > 1
        preq = req[1] if hasval else req[0]
        ui[preq]["value"] = None
        ui[preq]["enabled"] = False
        assertions += [str(k) for k in req]
    elif validation_type == "uuid":
        err = (ValueError, IndexError)
        if workspace is None:
            assertions = [param, "uuid", invalid_value, "valid uuid"]
        if workspace is not None and parent is None:
            uuid_str = str(uuid4())
            ui[param]["value"] = uuid_str
            assertions = [param, "uuid", uuid_str, "Address does"]
        if workspace is not None and parent is not None:
            ui[param]["value"] = "{c02e0470-0c3e-4119-8ac1-0aacba5334af}"
            ui[param]["parent"] = parent
            ui[parent]["value"] = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"

            assertions = [param, "uuid", invalid_value, "child of"]

    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)

    with pytest.raises(err) as excinfo:
        ifile = InputFile(filepath)
        MagneticVectorParams(ifile, geoh5=workspace)

    for a in assertions:
        assert a in str(excinfo.value)


def param_test_generator(tmp_path, param, value, workspace=workspace):
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    wrkstr = workspace.h5file
    ifile.write_ui_json(MVI_defaults, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    if isinstance(ui[param], dict):
        if "isValue" in ui[param].keys():
            if isinstance(value, UUID):
                ui[param]["isValue"] = False
            else:
                ui[param]["isValue"] = True
        ui[param]["value"] = value
        ui[param]["enabled"] = True
    else:
        ui[param] = value
    ui["geoh5"] = None
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    ifile = InputFile(filepath)
    params = MagneticVectorParams(ifile, validate=False, geoh5=workspace)

    try:
        value = UUID(str(value))
    except ValueError:
        pass

    pval = getattr(params, param)

    assert pval == value


def test_params_initialize():
    for params in [
        MagneticScalarParams(validate=False),
        MagneticVectorParams(validate=False),
        GravityParams(validate=False),
        DirectCurrentParams(validate=False),
        InducedPolarizationParams(validate=False),
        OctreeParams(validate=False),
        PeakFinderParams(validate=False),
    ]:
        check = []
        for k, v in params.defaults.items():
            if " " in k:
                continue
                check.append(getattr(params, k) == v)
        assert all(check)

    params = MagneticVectorParams(u_cell_size=9999, validate=False, geoh5=workspace)
    assert params.u_cell_size == 9999
    params = GravityParams(u_cell_size=9999, validate=False, geoh5=workspace)
    assert params.u_cell_size == 9999
    params = OctreeParams(vertical_padding=500, validate=False, geoh5=workspace)
    assert params.vertical_padding == 500
    params = PeakFinderParams(center=1000, validate=False, geoh5=workspace)
    assert params.center == 1000


def test_default_input_file(tmp_path):

    for params_class in [
        MagneticScalarParams,
        MagneticVectorParams,
        GravityParams,
        DirectCurrentParams,
        InducedPolarizationParams,
    ]:
        filename = os.path.join(tmp_path, "test.ui.json")
        params = params_class(validate=False)
        params.write_input_file(name=filename, default=True)
        ifile = InputFile(filename)

        # check that reads back into input file with defaults
        check = []
        for k, v in ifile.data.items():
            if " " in k:
                continue
            check.append(v == params.defaults[k])
        assert all(check)

        # check that params constructed from_path is defaulted
        ifile = InputFile(filename)
        params2 = params_class(ifile)
        check = []
        for k, v in params2.to_dict(ui_json_format=False).items():
            if " " in k:
                continue
            check.append(v == ifile.data[k])
        assert all(check)

        # check that params constructed from_input_file is defaulted
        params3 = params_class(ifile)
        check = []
        for k, v in params3.to_dict(ui_json_format=False).items():
            if " " in k:
                continue
            check.append(v == ifile.data[k])
        assert all(check)


def test_update(tmp_path):
    new_params = {
        "u_cell_size": 5,
    }
    params = MagneticVectorParams(validate=False)
    params.update(new_params)
    assert params.u_cell_size == 5

    new_params = {
        "topography_object": {
            "main": True,
            "group": "Topography",
            "label": "Object",
            "meshType": [
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
                "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
            ],
            "value": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
        }
    }
    params.update(new_params)
    assert params.topography_object == UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    new_params = {
        "topography": {
            "association": "Vertex",
            "dataType": "Float",
            "group": "Topography",
            "main": True,
            "dependency": "forward_only",
            "dependencyType": "hide",
            "isValue": False,
            "label": "Elevation",
            "parent": "topography_object",
            "property": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "value": 0.0,
        }
    }

    params.update(new_params)
    assert params.topography == UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")


def test_params_constructors(tmp_path):
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    ui = deepcopy(MVI_defaults)
    ui["geoh5"] = wrkstr
    ifile.write_ui_json(ui)

    params1 = MagneticVectorParams(
        input_file=InputFile(filepath), validate=False, geoh5=workspace
    )
    params2 = MagneticVectorParams(input_file=ifile, validate=False, geoh5=workspace)


def test_active_set():
    params = MagneticVectorParams(
        default=False,
        validate=False,
        forward_only=True,
        geoh5=workspace,
        inversion_type="magnetic vector",
        u_cell_size=2,
    )
    assert "inversion_type" in params.active_set()
    assert "u_cell_size" in params.active_set()


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "magnetic scalar"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "em", "value", workspace=workspace)


def test_validate_inducing_field_strength(tmp_path):
    param = "inducing_field_strength"
    newval = 60000
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_inducing_field_inclination(tmp_path):
    param = "inducing_field_inclination"
    newval = 44
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_inducing_field_declination(tmp_path):
    param = "inducing_field_declination"
    newval = 9
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_topography_object(tmp_path):
    param = "topography_object"
    newval = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, True, "type", workspace=workspace)
    catch_invalid_generator(tmp_path, param, "lsdkfj", "uuid", workspace=workspace)
    catch_invalid_generator(tmp_path, param, "", "uuid", workspace=workspace)


def test_validate_topography(tmp_path):
    param = "topography"
    newval = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, True, "type", workspace=workspace)
    catch_invalid_generator(tmp_path, param, newval, "reqs", workspace=workspace)


def test_validate_data_object(tmp_path):
    param = "data_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, 2, "type", workspace=workspace)


def test_validate_tmi_channel(tmp_path):
    param = "tmi_channel"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, 4, "type", workspace=workspace)


def test_validate_tmi_uncertainty(tmp_path):
    param = "tmi_uncertainty"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_starting_model_object(tmp_path):
    param = "starting_model_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_starting_inclination_object(tmp_path):
    param = "starting_inclination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_starting_declination_object(tmp_path):
    param = "starting_declination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_starting_model(tmp_path):
    param = "starting_model"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_starting_inclination(tmp_path):
    param = "starting_inclination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_starting_declination(tmp_path):
    param = "starting_declination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_tile_spatial(tmp_path):
    param = "tile_spatial"
    newval = 9
    invalidval = {}
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, invalidval, "type", workspace=workspace)


def test_validate_receivers_radar_drape(tmp_path):
    param = "receivers_radar_drape"
    newval = str(uuid4())
    invalidval = {}
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, invalidval, "type", workspace=workspace)


def test_validate_receivers_offset_x(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_receivers_offset_y(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_receivers_offset_z(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_ignore_values(tmp_path):
    param = "ignore_values"
    newval = "12345"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_resolution(tmp_path):
    param = "resolution"
    newval = 10
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_detrend_order(tmp_path):
    param = "detrend_order"
    newval = 2
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_detrend_type(tmp_path):
    param = "detrend_type"
    newval = "perimeter"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "sdf", "value", workspace=workspace)


def test_validate_max_chunk_size(tmp_path):
    param = "max_chunk_size"
    newval = 256
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "asdf", "type", workspace=workspace)


def test_validate_chunk_by_rows(tmp_path):
    param = "chunk_by_rows"
    newval = True
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "sdf", "type", workspace=workspace)


def test_validate_output_tile_files(tmp_path):
    param = "output_tile_files"
    newval = True
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "sdf", "type", workspace=workspace)


def test_validate_mesh(tmp_path):
    param = "mesh"
    newval = "{c02e0470-0c3e-4119-8ac1-0aacba5334af}"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_u_cell_size(tmp_path):
    param = "u_cell_size"
    newval = 9
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "sdf", "type", workspace=workspace)


def test_validate_v_cell_size(tmp_path):
    param = "v_cell_size"
    newval = 9
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "sdf", "type", workspace=workspace)


def test_validate_w_cell_size(tmp_path):
    param = "w_cell_size"
    newval = 9
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "sdf", "type", workspace=workspace)


def test_validate_octree_levels_topo(tmp_path):
    param = "octree_levels_topo"
    newval = [1, 2, 3]
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_octree_levels_obs(tmp_path):
    param = "octree_levels_obs"
    newval = [1, 2, 3]
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_depth_core(tmp_path):
    param = "depth_core"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_max_distance(tmp_path):
    param = "max_distance"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_horizontal_padding(tmp_path):
    param = "horizontal_padding"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_vertical_padding(tmp_path):
    param = "vertical_padding"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_window_center_x(tmp_path):
    param = "window_center_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_window_center_y(tmp_path):
    param = "window_center_y"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_window_width(tmp_path):
    param = "window_width"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_window_height(tmp_path):
    param = "window_height"
    newval = 99
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    newval = "voxel"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "value", workspace=workspace)


def test_validate_chi_factor(tmp_path):
    param = "chi_factor"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_max_iterations(tmp_path):
    param = "max_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_max_cg_iterations(tmp_path):
    param = "max_cg_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_max_global_iterations(tmp_path):
    param = "max_global_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_initial_beta(tmp_path):
    param = "initial_beta"
    newval = 2
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_initial_beta_ratio(tmp_path):
    param = "initial_beta_ratio"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_tol_cg(tmp_path):
    param = "tol_cg"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_alpha_s(tmp_path):
    param = "alpha_s"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_alpha_x(tmp_path):
    param = "alpha_x"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_alpha_y(tmp_path):
    param = "alpha_y"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_alpha_z(tmp_path):
    param = "alpha_z"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_s_norm(tmp_path):
    param = "s_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_x_norm(tmp_path):
    param = "x_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_y_norm(tmp_path):
    param = "y_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_z_norm(tmp_path):
    param = "z_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_reference_model_object(tmp_path):
    param = "reference_model_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_reference_inclination_object(tmp_path):
    param = "reference_inclination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_reference_declination_object(tmp_path):
    param = "reference_declination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_reference_model(tmp_path):
    param = "reference_model"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_reference_inclination(tmp_path):
    param = "reference_inclination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_reference_declination(tmp_path):
    param = "reference_declination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_gradient_type(tmp_path):
    param = "gradient_type"
    newval = "components"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "value", workspace=workspace)


def test_validate_lower_bound(tmp_path):
    param = "lower_bound"
    newval = -1000
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_upper_bound(tmp_path):
    param = "upper_bound"
    newval = 1000
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_parallelized(tmp_path):
    param = "parallelized"
    newval = False
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_n_cpu(tmp_path):
    param = "n_cpu"
    newval = 12
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_max_ram(tmp_path):
    param = "max_ram"
    newval = 10
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)


def test_validate_workspace(tmp_path):
    param = "workspace"
    newval = "../assets/something.geoh5py"
    # param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, 4, "type", workspace=workspace)


def test_validate_out_group(tmp_path):
    param = "out_group"
    newval = "test_"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_no_data_value(tmp_path):
    param = "no_data_value"
    newval = 5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "lskjdf", "type", workspace=workspace)


def test_input_file_construction():

    params_classes = [
        GravityParams,
        MagneticScalarParams,
        MagneticVectorParams,
        DirectCurrentParams,
        InducedPolarizationParams,
        OctreeParams,
        PeakFinderParams,
    ]

    for params_class in params_classes:
        filename = "test.ui.json"
        for forward_only in [True, False]:
            params = params_class(forward_only=forward_only, validate=False)
            params.write_input_file(name=filename, default=True)
            ifile = InputFile(filename)
            params = params_class(ifile, validate=False)

            check = []
            for k, v in params.defaults.items():
                if " " in k:
                    continue
                    check.append(getattr(params, k) == v)

            assert all(check)


def test_gravity_inversion_type():
    params = GravityParams(validate=True)
    params.inversion_type = "gravity"
    with pytest.raises(ValueError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["inversion_type", "alskdj", "gravity"]]
    )


def test_gz_channel_bool():
    params = GravityParams(validate=True)
    params.gz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gz_channel_bool", "type", "str", "bool"]]
    )


def test_gz_channel():
    params = GravityParams(validate=True)
    params.gz_channel = str(uuid4())
    params.gz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gz_channel", "type", "int", "str", "UUID"]]
    )


def test_gz_uncertainty():
    params = GravityParams(validate=True)
    params.gz_uncertainty = str(uuid4())
    params.gz_uncertainty = uuid4()
    params.gz_uncertainty = 4
    params.gz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gz_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_guv_channel_bool():
    params = GravityParams(validate=True)
    params.guv_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.guv_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["guv_channel_bool", "type", "str", "bool"]]
    )


def test_guv_channel():
    params = GravityParams(validate=True)
    params.guv_channel = str(uuid4())
    params.guv_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.guv_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["guv_channel", "type", "int", "str", "UUID"]]
    )


def test_guv_uncertainty():
    params = GravityParams(validate=True)
    params.guv_uncertainty = str(uuid4())
    params.guv_uncertainty = uuid4()
    params.guv_uncertainty = 4
    params.guv_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.guv_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "guv_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gxy_channel_bool():
    params = GravityParams(validate=True)
    params.gxy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gxy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxy_channel_bool", "type", "str", "bool"]]
    )


def test_gxy_channel():
    params = GravityParams(validate=True)
    params.gxy_channel = str(uuid4())
    params.gxy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gxy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxy_channel", "type", "int", "str", "UUID"]]
    )


def test_gxy_uncertainty():
    params = GravityParams(validate=True)
    params.gxy_uncertainty = str(uuid4())
    params.gxy_uncertainty = uuid4()
    params.gxy_uncertainty = 4
    params.gxy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gxy_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gxy_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gxx_channel_bool():
    params = GravityParams(validate=True)
    params.gxx_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gxx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxx_channel_bool", "type", "str", "bool"]]
    )


def test_gxx_channel():
    params = GravityParams(validate=True)
    params.gxx_channel = str(uuid4())
    params.gxx_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gxx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxx_channel", "type", "int", "str", "UUID"]]
    )


def test_gxx_uncertainty():
    params = GravityParams(validate=True)
    params.gxx_uncertainty = str(uuid4())
    params.gxx_uncertainty = uuid4()
    params.gxx_uncertainty = 4
    params.gxx_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gxx_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gxx_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gyy_channel_bool():
    params = GravityParams(validate=True)
    params.gyy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gyy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gyy_channel_bool", "type", "str", "bool"]]
    )


def test_gyy_channel():
    params = GravityParams(validate=True)
    params.gyy_channel = str(uuid4())
    params.gyy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gyy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gyy_channel", "type", "int", "str", "UUID"]]
    )


def test_gyy_uncertainty():
    params = GravityParams(validate=True)
    params.gyy_uncertainty = str(uuid4())
    params.gyy_uncertainty = uuid4()
    params.gyy_uncertainty = 4
    params.gyy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gyy_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gyy_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gzz_channel_bool():
    params = GravityParams(validate=True)
    params.gzz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gzz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gzz_channel_bool", "type", "str", "bool"]]
    )


def test_gzz_channel():
    params = GravityParams(validate=True)
    params.gzz_channel = str(uuid4())
    params.gzz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gzz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gzz_channel", "type", "int", "str", "UUID"]]
    )


def test_gzz_uncertainty():
    params = GravityParams(validate=True)
    params.gzz_uncertainty = str(uuid4())
    params.gzz_uncertainty = uuid4()
    params.gzz_uncertainty = 4
    params.gzz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gzz_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gzz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gxz_channel_bool():
    params = GravityParams(validate=True)
    params.gxz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gxz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxz_channel_bool", "type", "str", "bool"]]
    )


def test_gxz_channel():
    params = GravityParams(validate=True)
    params.gxz_channel = str(uuid4())
    params.gxz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gxz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxz_channel", "type", "int", "str", "UUID"]]
    )


def test_gxz_uncertainty():
    params = GravityParams(validate=True)
    params.gxz_uncertainty = str(uuid4())
    params.gxz_uncertainty = uuid4()
    params.gxz_uncertainty = 4
    params.gxz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gxz_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gxz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gyz_channel_bool():
    params = GravityParams(validate=True)
    params.gyz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gyz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gyz_channel_bool", "type", "str", "bool"]]
    )


def test_gyz_channel():
    params = GravityParams(validate=True)
    params.gyz_channel = str(uuid4())
    params.gyz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gyz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gyz_channel", "type", "int", "str", "UUID"]]
    )


def test_gyz_uncertainty():
    params = GravityParams(validate=True)
    params.gyz_uncertainty = str(uuid4())
    params.gyz_uncertainty = uuid4()
    params.gyz_uncertainty = 4
    params.gyz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gyz_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gyz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gx_channel_bool():
    params = GravityParams(validate=True)
    params.gx_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gx_channel_bool", "type", "str", "bool"]]
    )


def test_gx_channel():
    params = GravityParams(validate=True)
    params.gx_channel = str(uuid4())
    params.gx_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gx_channel", "type", "int", "str", "UUID"]]
    )


def test_gx_uncertainty():
    params = GravityParams(validate=True)
    params.gx_uncertainty = str(uuid4())
    params.gx_uncertainty = uuid4()
    params.gx_uncertainty = 4
    params.gx_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gx_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gx_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gy_channel_bool():
    params = GravityParams(validate=True)
    params.gy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gy_channel_bool", "type", "str", "bool"]]
    )


def test_gy_channel():
    params = GravityParams(validate=True)
    params.gy_channel = str(uuid4())
    params.gy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gy_channel", "type", "int", "str", "UUID"]]
    )


def test_gy_uncertainty():
    params = GravityParams(validate=True)
    params.gy_uncertainty = str(uuid4())
    params.gy_uncertainty = uuid4()
    params.gy_uncertainty = 4
    params.gy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gy_uncertainty = workspace

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gy_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_isValue(tmp_path):
    # "starting_model"
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath

    mesh = workspace.get_entity("O2O_Interp_25m")[0]

    params = MagneticVectorParams(input_file=ifile, validate=False, workspace=workspace)
    params.starting_model_object = mesh.uid
    params.starting_model = 0.0

    params.write_input_file(name=filepath)

    with open(filepath) as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is True, "isValue should be True"

    params.starting_model = mesh.get_data("VTEM_model")[0].uid

    params.write_input_file(name=filepath)
    with open(filepath) as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is False, "isValue should be False"
