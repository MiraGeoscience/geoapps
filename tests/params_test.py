#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pytest
from geoh5py.workspace import Workspace

from geoapps.io import InputFile, Params
from geoapps.io.constants import default_ui_json

######################  Setup  ###########################

tmpfile = lambda path: os.path.join(path, "test.ui.json")
wrkstr = "FlinFlon.geoh5"
workspace = Workspace(wrkstr)


def tmp_input_file(filepath, idict):
    with open(filepath, "w") as f:
        json.dump(idict, f)


def default_test_generator(tmp_path, param, newval, wrkstr=None):

    d_u_j = deepcopy(default_ui_json)
    params = Params()
    assert getattr(params, param) == d_u_j[param]["default"]
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ifile.write_ui_json(default=True, workspace=wrkstr)
    params = Params.from_path(filepath)
    assert getattr(params, param) == d_u_j[param]["default"]
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["isValue"] = True
    ui[param]["value"] = newval
    ui[param]["visible"] = True
    ui[param]["enabled"] = False
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = Params.from_path(filepath)
    assert getattr(params, param) == d_u_j[param]["default"]
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["isValue"] = True
    ui[param]["value"] = newval
    ui[param]["visible"] = False
    ui[param]["enabled"] = True
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = Params.from_path(filepath)
    assert getattr(params, param) == d_u_j[param]["default"]


def catch_invalid_generator(tmp_path, param, invalid_value, validation_type):

    params = Params()
    params.workspace = workspace
    if validation_type == "type":
        err = TypeError
    else:
        err = ValueError
    with pytest.raises(err) as excinfo:
        params.__setattr__(param, invalid_value)

    assert str(validation_type) in str(excinfo.value)
    # assert str(invalid_value) in str(excinfo.value)
    assert param in str(excinfo.value)

    return params


def catch_invalid_uuid_generator(param, invalid_uuid, no_workspace):
    params = Params()
    params.workspace = workspace
    with pytest.raises(ValueError) as excinfo:
        params.__setattr__(param, invalid_uuid)
    assert f"UUID string: {invalid_uuid}" in str(excinfo.value)
    with pytest.raises(IndexError) as excinfo:
        params.__setattr__(param, no_workspace)
    assert f"UUID address {no_workspace}" in str(excinfo.value)


def param_test_generator(tmp_path, param, value, wrkstr=None):
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ifile.write_ui_json(default=True, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    ui[param]["isValue"] = True
    ui[param]["value"] = value
    ui[param]["visible"] = True
    ui[param]["enabled"] = True
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = Params.from_path(filepath)
    assert getattr(params, param) == value


######################  Tests  ###########################


def test_params_constructors(tmp_path):
    filepath = tmpfile(tmp_path)
    ifile = InputFile(filepath)
    ifile.write_ui_json(default=True)
    params1 = Params.from_path(filepath)
    params2 = Params.from_ifile(ifile)


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "gravity"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "em", "value")


def test_validate_forward_only(tmp_path):
    param = "forward_only"
    newval = False
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "test", "type")


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
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_topography(tmp_path):
    param = "topography"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_data_object(tmp_path):
    param = "data_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_tmi_channel(tmp_path):
    param = "tmi_channel"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_tmi_uncertainty(tmp_path):
    param = "tmi_uncertainty"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_starting_model_object(tmp_path):
    param = "starting_model_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_starting_inclination_object(tmp_path):
    param = "starting_inclination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_starting_declination_object(tmp_path):
    param = "starting_declination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_starting_model(tmp_path):
    param = "starting_model"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_starting_inclination(tmp_path):
    param = "starting_inclination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_starting_declination(tmp_path):
    param = "starting_declination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


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
    catch_invalid_uuid_generator(param, "test", str(uuid4()))


def test_validate_receivers_radar_drape(tmp_path):
    param = "receivers_radar_drape"
    newval = str(uuid4())
    invalidval = {}
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, invalidval, "type")
    catch_invalid_uuid_generator(param, "test", newval)


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
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, invalidval, "type")
    catch_invalid_uuid_generator(param, "test", newval)


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
    catch_invalid_generator(tmp_path, param, "", "value")


def test_validate_max_chunk_size(tmp_path):
    param = "max_chunk_size"
    newval = 256
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


def test_validate_chunk_by_rows(tmp_path):
    param = "chunk_by_rows"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


def test_validate_output_tile_files(tmp_path):
    param = "output_tile_files"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


def test_validate_mesh(tmp_path):
    param = "mesh"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_mesh_from_params(tmp_path):
    param = "mesh_from_params"
    newval = True
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


def test_validate_core_cell_size_x(tmp_path):
    param = "core_cell_size_x"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


def test_validate_core_cell_size_y(tmp_path):
    param = "core_cell_size_y"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


def test_validate_core_cell_size_z(tmp_path):
    param = "core_cell_size_z"
    newval = 9
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "", "type")


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
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_reference_inclination_object(tmp_path):
    param = "reference_inclination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_reference_declination_object(tmp_path):
    param = "reference_declination_object"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_reference_model(tmp_path):
    param = "reference_model"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_reference_inclination(tmp_path):
    param = "reference_inclination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


def test_validate_reference_declination(tmp_path):
    param = "reference_declination"
    newval = str(uuid4())
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval, wrkstr=wrkstr)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")
    catch_invalid_uuid_generator(param, "test", newval)


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


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "gravity"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "value")


def test_validate_workspace(tmp_path):
    param = "workspace"
    newval = "../assets/something.geoh5py"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_output_geoh5(tmp_path):
    param = "output_geoh5"
    newval = "../assets/something.geoh5py"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, newval)
    ### test ordinary behaviour ###
    # param_test_generator(tmp_path, param, newval)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, {}, "type")


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
