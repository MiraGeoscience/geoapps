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
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json, validations

######################  Setup  ###########################

tmpfile = lambda path: os.path.join(path, "test.ui.json")
wrkstr = "FlinFlon.geoh5"
workspace = Workspace(wrkstr)


def tmp_input_file(filepath, idict):
    with open(filepath, "w") as f:
        json.dump(idict, f)


def default_test_generator(tmp_path, param, newval):

    d_u_j = deepcopy(default_ui_json)
    params = MVIParams()
    assert getattr(params, param) == d_u_j[param]["default"]
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ifile.write_ui_json(default_ui_json, default=True, workspace=wrkstr)
    params = MVIParams.from_path(filepath)
    assert getattr(params, param) == d_u_j[param]["default"]
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["isValue"] = True
    ui[param]["value"] = newval
    ui[param]["visible"] = True
    ui[param]["enabled"] = False
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = MVIParams.from_path(filepath)
    assert getattr(params, param) == d_u_j[param]["default"]
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["isValue"] = True
    ui[param]["value"] = newval
    ui[param]["visible"] = False
    ui[param]["enabled"] = True
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = MVIParams.from_path(filepath)
    assert getattr(params, param) == d_u_j[param]["default"]


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
    pvalidations = validations[param][key_map[validation_type]]
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ifile.write_ui_json(default_ui_json, default=True, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["value"] = invalid_value
    ui[param]["visible"] = True
    ui[param]["enabled"] = True
    ui[param]["isValue"] = True
    if validation_type == "value":
        err = ValueError
        assertions = [
            "Must be",
            param,
            "value",
            str(invalid_value),
            *[str(v) for v in pvalidations],
        ]
    elif validation_type == "type":
        err = TypeError
        types = set(pvalidations + [type(invalid_value)])
        assertions = ["Must be", param, "type", *[t.__name__ for t in types]]
    elif validation_type == "shape":
        err = ValueError
        shapes = set(pvalidations + [np.array(invalid_value).shape])
        assertions = ["Must be", param, "shape", *[str(s) for s in shapes]]
    elif validation_type == "reqs":
        err = KeyError
        assertions = ["Unsatisfied", param]
        req = pvalidations[0]
        hasval = len(req) > 1
        preq = req[1] if hasval else req[0]
        ui[preq]["value"] = None
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
        params = MVIParams.from_path(filepath)

    for a in assertions:
        assert a in str(excinfo.value)


def param_test_generator(tmp_path, param, value):
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ifile.write_ui_json(default_ui_json, default=True, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["isValue"] = True
    ui[param]["value"] = value
    ui[param]["visible"] = True
    ui[param]["enabled"] = True
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = MVIParams.from_path(filepath)
    try:
        value = UUID(value)
        assert getattr(params, param) == value
    except:
        assert getattr(params, param) == value


######################  Tests  ###########################


def test_params_constructors(tmp_path):
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ui = default_ui_json.copy()
    ui["geoh5"] = wrkstr
    ifile.write_ui_json(ui, default=True)
    params1 = MVIParams.from_path(filepath)
    params2 = MVIParams.from_ifile(ifile)


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "mvic"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "em", "value")
    catch_invalid_generator(tmp_path, param, "mvi", "reqs")


def test_validate_forward_only(tmp_path):
    param = "forward_only"
    newval = False
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")
    catch_invalid_generator(tmp_path, param, True, "reqs")


def test_validate_inducing_field_strength(tmp_path):
    param = "inducing_field_strength"
    newval = 60000
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_inducing_field_inclination(tmp_path):
    param = "inducing_field_inclination"
    newval = 44
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_inducing_field_declination(tmp_path):
    param = "inducing_field_declination"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_topography_object(tmp_path):
    param = "topography_object"
    newval = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, True, "type")
    catch_invalid_generator(tmp_path, param, "lsdkfj", "uuid")
    catch_invalid_generator(tmp_path, param, "", "uuid", workspace=workspace)


def test_validate_topography(tmp_path):
    param = "topography"
    newval = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, True, "type")
    catch_invalid_generator(tmp_path, param, "lsdkfj", "uuid")
    catch_invalid_generator(
        tmp_path, param, "", "uuid", workspace=workspace, parent="topography_object"
    )


def test_validate_data_object(tmp_path):
    param = "data_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_tmi_channel(tmp_path):
    param = "tmi_channel"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_tmi_uncertainty(tmp_path):
    param = "tmi_uncertainty"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_model_object(tmp_path):
    param = "starting_model_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_inclination_object(tmp_path):
    param = "starting_inclination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_declination_object(tmp_path):
    param = "starting_declination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_model(tmp_path):
    param = "starting_model"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_inclination(tmp_path):
    param = "starting_inclination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_declination(tmp_path):
    param = "starting_declination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_tile_spatial(tmp_path):
    param = "tile_spatial"
    newval = 9
    invalidval = {}
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, invalidval, "type")


def test_validate_receivers_radar_drape(tmp_path):
    param = "receivers_radar_drape"
    newval = str(uuid4())
    invalidval = {}
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, invalidval, "type")


def test_validate_receivers_offset_x(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_receivers_offset_y(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_receivers_offset_z(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_gps_receivers_offset(tmp_path):
    param = "gps_receivers_offset"
    newval = str(uuid4())
    invalidval = {}
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, invalidval, "type")


def test_validate_ignore_values(tmp_path):
    param = "ignore_values"
    newval = "12345"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_resolution(tmp_path):
    param = "resolution"
    newval = 10
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_detrend_data(tmp_path):
    param = "detrend_data"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_detrend_order(tmp_path):
    param = "detrend_order"
    newval = 2
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 9, "value")


def test_validate_detrend_type(tmp_path):
    param = "detrend_type"
    newval = "corners"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "value")


def test_validate_max_chunk_size(tmp_path):
    param = "max_chunk_size"
    newval = 256
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "asdf", "type")


def test_validate_chunk_by_rows(tmp_path):
    param = "chunk_by_rows"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_output_tile_files(tmp_path):
    param = "output_tile_files"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_mesh(tmp_path):
    param = "mesh"
    newval = "{c02e0470-0c3e-4119-8ac1-0aacba5334af}"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_mesh_from_params(tmp_path):
    param = "mesh_from_params"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_core_cell_size_x(tmp_path):
    param = "core_cell_size_x"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_core_cell_size_y(tmp_path):
    param = "core_cell_size_y"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_core_cell_size_z(tmp_path):
    param = "core_cell_size_z"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_octree_levels_topo(tmp_path):
    param = "octree_levels_topo"
    newval = [1, 2, 3]
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_octree_levels_obs(tmp_path):
    param = "octree_levels_obs"
    newval = [1, 2, 3]
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_octree_levels_padding(tmp_path):
    param = "octree_levels_padding"
    newval = [1, 2, 3]
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_depth_core(tmp_path):
    param = "depth_core"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_max_distance(tmp_path):
    param = "max_distance"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_padding_distance_x(tmp_path):
    param = "padding_distance_x"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_padding_distance_y(tmp_path):
    param = "padding_distance_y"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_padding_distance_z(tmp_path):
    param = "padding_distance_z"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_center_x(tmp_path):
    param = "window_center_x"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_center_y(tmp_path):
    param = "window_center_y"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_width(tmp_path):
    param = "window_width"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_height(tmp_path):
    param = "window_height"
    newval = 99
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    newval = "voxel"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "value")


def test_validate_chi_factor(tmp_path):
    param = "chi_factor"
    newval = 0.5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_iterations(tmp_path):
    param = "max_iterations"
    newval = 2
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_cg_iterations(tmp_path):
    param = "max_cg_iterations"
    newval = 2
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_global_iterations(tmp_path):
    param = "max_global_iterations"
    newval = 2
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_initial_beta(tmp_path):
    param = "initial_beta"
    newval = 2
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_initial_beta_ratio(tmp_path):
    param = "initial_beta_ratio"
    newval = 0.5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_tol_cg(tmp_path):
    param = "tol_cg"
    newval = 0.1
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_s(tmp_path):
    param = "alpha_s"
    newval = 0.1
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_x(tmp_path):
    param = "alpha_x"
    newval = 0.1
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_y(tmp_path):
    param = "alpha_y"
    newval = 0.1
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_z(tmp_path):
    param = "alpha_z"
    newval = 0.1
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_smallness_norm(tmp_path):
    param = "smallness_norm"
    newval = 0.5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_x_norm(tmp_path):
    param = "x_norm"
    newval = 0.5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_y_norm(tmp_path):
    param = "y_norm"
    newval = 0.5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_z_norm(tmp_path):
    param = "z_norm"
    newval = 0.5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_reference_model_object(tmp_path):
    param = "reference_model_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_inclination_object(tmp_path):
    param = "reference_inclination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_declination_object(tmp_path):
    param = "reference_declination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_model(tmp_path):
    param = "reference_model"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_inclination(tmp_path):
    param = "reference_inclination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_declination(tmp_path):
    param = "reference_declination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_gradient_type(tmp_path):
    param = "gradient_type"
    newval = "components"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "value")


def test_validate_lower_bound(tmp_path):
    param = "lower_bound"
    newval = -1000
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_upper_bound(tmp_path):
    param = "upper_bound"
    newval = 1000
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_parallelized(tmp_path):
    param = "parallelized"
    newval = False
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_n_cpu(tmp_path):
    param = "n_cpu"
    newval = 12
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_ram(tmp_path):
    param = "max_ram"
    newval = 10
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_workspace(tmp_path):
    param = "workspace"
    newval = "../assets/something.geoh5py"
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_output_geoh5(tmp_path):
    param = "output_geoh5"
    newval = "../assets/something.geoh5py"
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param,  newval)
    ### test validation behaviour ###
    # catch_invalid_ge/nerator(tmp_path, param, 34, "type")


def test_validate_out_group(tmp_path):
    param = "out_group"
    newval = "test_"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_no_data_value(tmp_path):
    param = "no_data_value"
    newval = 5
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
