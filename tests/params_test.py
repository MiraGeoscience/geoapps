#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import numpy as np
import pytest

from geoapps.io import InputFile, Params
from geoapps.io.driver import create_relative_path, create_work_path

######################  Setup  ###########################


input_dict = {"inversion_type": "mvi", "core_cell_size": 2}
tmpfile = lambda path: os.path.join(path, "test.json")


def tmp_input_file(filepath, input_dict):
    with open(filepath, "w") as outfile:
        json.dump(input_dict, outfile)


def default_test_generator(tmp_path, param, default_value):
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, input_dict)
    params = Params.from_path(filepath)

    assert params.__getattribute__(param) == default_value


def catch_invalid_generator(tmp_path, param, invalid_value, validation_type):
    idict = input_dict.copy()
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    err = TypeError if validation_type == "type" else ValueError
    with pytest.raises(err) as excinfo:
        params.__setattr__(param, invalid_value)
    assert validation_type in str(excinfo.value)
    return params


def param_test_generator(tmp_path, tparams):
    idict = input_dict.copy()
    idict.update(tparams)
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    for k, v in tparams.items():
        assert getattr(params, k) == v


######################  Tests  ###########################


def test_create_work_path():
    wp = create_work_path("./inputfile.json")
    assert wp == os.path.abspath(".") + os.path.sep


def test_create_relative_path():
    dsep = os.path.sep
    outpath = create_work_path("../../some/project/file")
    path = create_relative_path("../assets/Inversion_.json", outpath)
    root = os.path.abspath("..")
    validate_path = os.path.join(root, "assets", "some", "project") + dsep
    assert path == validate_path


def test_params_constructors(tmp_path):
    idict = input_dict.copy()
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    assert params.inversion_type == "mvi"
    assert params.core_cell_size == 2
    assert params.inversion_style == "voxel"
    inputfile = InputFile(filepath)
    params = Params.from_ifile(inputfile)
    assert params.inversion_type == "mvi"
    assert params.core_cell_size == 2
    assert params.inversion_style == "voxel"


def test_override_default():
    params = Params("mvi", 2)
    params._override_default("forward_only", True)
    assert params.forward_only == True


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    ### test ordinary behaviour ###
    tparams = {param: "gravity", "core_cell_size": 4.0}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "em", "value")
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, {"core_cell_size": 2})
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "parameter(s): ('inversion_type',)." in str(excinfo.value)


def test_validate_core_cell_size(tmp_path):
    param = "core_cell_size"
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, {"inversion_type": "mvi"})
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "parameter(s): ('core_cell_size',)." in str(excinfo.value)


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, "voxel")
    ### test ordinary behaviour ###
    tparams = {param: "voxel"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "parametric", "value")


def test_validate_forward_only(tmp_path):
    param = "forward_only"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, False)
    ### test ordinary behaviour ###
    tparams = {param: True, "reference_model": {"model": "some_path"}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "true", "type")


def test_validate_result_folder(tmp_path):
    param = "result_folder"
    ### test default behaviour ###
    path = os.path.join(tmp_path, "SimPEG_PFInversion") + os.path.sep
    default_test_generator(tmp_path, param, path)
    ### test ordinary behaviour ###
    idict = input_dict.copy()
    idict.update({param: os.path.join(tmp_path, "some_path") + os.path.sep})
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    assert getattr(params, param) == idict[param]
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, True, "type")


def test_validate_inducing_field_aid(tmp_path):
    param = "inducing_field_aid"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    idict = input_dict.copy()
    idict.update({param: [1, 2, 3]})
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    assert all(getattr(params, param) == np.array(idict[param]))
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")
    catch_invalid_generator(tmp_path, param, [1.0, 2.0], "shape")
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.inducing_field_aid = [0, 1, 2]
    assert "greater than 0." in str(excinfo.value)


def test_validate_resolution(tmp_path):
    param = "resolution"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 0)
    ### test ordinary behaviour ###
    tparams = {param: 6}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_window(tmp_path):
    param = "window"
    test_dict = {
        "center_x": 2,
        "center_y": 2,
        "width": 2,
        "height": 2,
        "azimuth": 2,
    }
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: test_dict.update({"center": [2, 2], "size": [2, 2]})}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1, "type")
    test_dict["nogood"] = 2
    catch_invalid_generator(tmp_path, param, test_dict, "keys")
    test_dict.pop("nogood", None)
    test_dict.pop("center_x", None)
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.window = test_dict
    assert "Input parameter 'window'" in str(excinfo.value)


def test_validate_workspace(tmp_path):
    param = "workspace"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: "some_path"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 12234, "type")
    idict = input_dict.copy()
    idict["data"] = {"type": "GA_object", "name": "test"}
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "data type 'GA_object'." in str(excinfo.value)


def test_validate_data(tmp_path):
    param = "data"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"type": "GA_object", "name": "Trevor"}}
    tparams.update({"workspace": "."})
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1234, "type")
    idict = input_dict.copy()
    idict["data"] = {"type": "GA_object"}
    idict["workspace"] = "."
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "Data 'type' and 'name'" in str(excinfo.value)
    idict["data"]["name"] = "Nicole"
    idict.pop("workspace", None)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "'workspace' path for data" in str(excinfo.value)


def test_validate_ignore_values(tmp_path):
    param = "ignore_values"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: "-1e8"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1234, "type")


def test_validate_detrend(tmp_path):
    param = "detrend"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"corners": 2}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1, "type")
    idict = input_dict.copy()
    idict["detrend"] = {"corners": 3}
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "Detrend order must be 0," in str(excinfo.value)


def test_validate_data_file(tmp_path):
    param = "data_file"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: "some_path"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1234, "type")
    idict = input_dict.copy()
    idict["data"] = {"type": "ubc_mag", "name": "test"}
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "for data types 'ubc_grav' and" in str(excinfo.value)


def test_validate_new_uncert(tmp_path):
    param = "new_uncert"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: [0.1, 1e-4]}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, ["test", "me"], "type")
    catch_invalid_generator(
        tmp_path,
        param,
        [
            0.1,
        ],
        "shape",
    )
    idict = input_dict.copy()
    idict["new_uncert"] = [12, 1e-14]
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "percent (new_uncert[0])" in str(excinfo.value)
    idict = input_dict.copy()
    idict["new_uncert"] = [0.1, -1e-14]
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "floor (new_uncert[1])" in str(excinfo.value)


def test_validate_input_mesh(tmp_path):
    param = "input_mesh"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: "some_path"}
    tparams.update({"save_to_geoh5": "another_path"})
    tparams.update({"input_mesh_file": "yet_another_path"})
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1, "type")
    idict = input_dict.copy()
    idict["input_mesh"] = "some_path"
    idict["input_mesh_file"] = "yet_another_path"
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "'save_to_geoh5' path if 'input_mesh'" in str(excinfo.value)
    idict = input_dict.copy()
    idict["input_mesh"] = "some_path"
    idict["save_to_geoh5"] = "yet_another_path"
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "'input_mesh_file' path if 'input_mesh'" in str(excinfo.value)


def test_validate_save_to_geoh5(tmp_path):
    param = "save_to_geoh5"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: "some_path"}
    tparams.update({"out_group": "test"})
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 1, "type")
    idict = input_dict.copy()
    idict["save_to_geoh5"] = "some_path"
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    with pytest.raises(ValueError) as excinfo:
        params = Params.from_path(filepath)
    assert "must contain a 'out_group'" in str(excinfo.value)


def test_validate_inversion_mesh_type(tmp_path):
    param = "inversion_mesh_type"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, "TREE")
    ### test ordinary behaviour ###
    tparams = {param: "TREE"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "not_valid_name", "value")


def test_validate_shift_mesh_z0(tmp_path):
    param = "shift_mesh_z0"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: 20.0}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "bogus", "type")


def test_validate_topography(tmp_path):
    param = "topography"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"constant": 0}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "bogus", "type")
    catch_invalid_generator(tmp_path, param, {"badkey": "na"}, "keys")


def test_validate_receivers_offset(tmp_path):
    param = "receivers_offset"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"constant": 0}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "bogus", "type")
    catch_invalid_generator(tmp_path, param, {"badkey": "na"}, "keys")


def test_validate_chi_factor(tmp_path):
    param = "chi_factor"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 1)
    ### test ordinary behaviour ###
    tparams = {param: 1}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "bogus", "type")
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.chi_factor = 0
    assert "chi_factor. Must be between 0 and 1." in str(excinfo.value)


def test_validate_model_norms(tmp_path):
    param = "model_norms"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, [2, 2, 2, 2])
    ### test ordinary behaviour ###
    tparams = {param: [1.0, 1.0, 1.0, 1.0]}
    param_test_generator(tmp_path, tparams)
    idict = input_dict.copy()
    idict["model_norms"] = [1, 1, 1, 1]
    filepath = tmpfile(tmp_path)
    tmp_input_file(filepath, idict)
    params = Params.from_path(filepath)
    assert params.max_iterations == 40
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, ["a", "b", "c", "d"], "type")
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.model_norms = [2, 2]
    assert "Must be a multiple of 4." in str(excinfo.value)


def test_validate_max_iterations(tmp_path):
    param = "max_iterations"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 10)
    ### test ordinary behaviour ###
    tparams = {param: 15}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 4.5, "type")
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.max_iterations = -10
    assert "Must be > 0." in str(excinfo.value)


def test_validate_max_cg_iterations(tmp_path):
    param = "max_cg_iterations"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 30)
    ### test ordinary behaviour ###
    tparams = {param: 15}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 4.5, "type")
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.max_iterations = -10
    assert "Must be > 0." in str(excinfo.value)


def test_validate_tol_cg(tmp_path):
    param = "tol_cg"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 1e-4)
    ### test ordinary behaviour ###
    tparams = {param: 1e-6}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nogood", "type")


def test_validate_max_global_iterations(tmp_path):
    param = "max_global_iterations"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 100)
    ### test ordinary behaviour ###
    tparams = {param: 40}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 4.5, "type")
    params = Params("mvi", 2)
    with pytest.raises(ValueError) as excinfo:
        params.max_iterations = -10
    assert "Must be > 0." in str(excinfo.value)


def test_validate_gradient_type(tmp_path):
    param = "gradient_type"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, "total")
    ### test ordinary behaviour ###
    tparams = {param: "total"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nogood", "value")


def test_validate_initial_beta(tmp_path):
    param = "initial_beta"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: 5}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nogood", "type")


def test_validate_initial_beta_ratio(tmp_path):
    param = "initial_beta_ratio"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 1e2)
    ### test ordinary behaviour ###
    tparams = {param: 0.6}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 2, "type")


def test_validate_n_cpu(tmp_path):
    param = "n_cpu"
    ### test ordinary behaviour ###
    tparams = {param: 32}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_max_ram(tmp_path):
    param = "max_ram"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 2)
    ### test ordinary behaviour ###
    tparams = {param: 16}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_depth_core(tmp_path):
    param = "depth_core"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"value": 2}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_padding_distance(tmp_path):
    param = "padding_distance"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, [[0, 0], [0, 0], [0, 0]])
    ### test ordinary behaviour ###
    tparams = {param: [[1, 1], [1, 1], [1, 1]]}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")
    catch_invalid_generator(tmp_path, param, [[1.0, 2.0]], "shape")


def test_validate_octree_levels_topo(tmp_path):
    param = "octree_levels_topo"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, [0, 1])
    ### test ordinary behaviour ###
    tparams = {param: [2, 2, 4, 4]}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_octree_levels_obs(tmp_path):
    param = "octree_levels_obs"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, [5, 5])
    ### test ordinary behaviour ###
    tparams = {param: [2, 2, 4, 4]}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_octree_levels_padding(tmp_path):
    param = "octree_levels_padding"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, [2, 2])
    ### test ordinary behaviour ###
    tparams = {param: [2, 2, 4, 4]}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_alphas(tmp_path):
    param = "alphas"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, [1] * 12)
    ### test ordinary behaviour ###
    tparams = {param: [5] * 12}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")
    params = Params("mvi", 2)
    params.alphas = [1] * 4
    assert len(params.alphas) == 12
    with pytest.raises(ValueError) as excinfo:
        params.alphas = [1] * 5
    assert "'alphas' must be a list of" in str(excinfo.value)


def test_validate_reference_model(tmp_path):
    param = "reference_model"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"model": "some_path"}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")
    catch_invalid_generator(tmp_path, param, {"invalid": 23}, "keys")


def test_validate_starting_model(tmp_path):
    param = "starting_model"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: {"model": "some_path"}}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")
    catch_invalid_generator(tmp_path, param, {"invalid": 23}, "keys")


def test_validate_lower_bound(tmp_path):
    param = "lower_bound"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, -np.inf)
    ### test ordinary behaviour ###
    tparams = {param: -1e10}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_upper_bound(tmp_path):
    param = "upper_bound"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, np.inf)
    ### test ordinary behaviour ###
    tparams = {param: 1e9}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_max_distance(tmp_path):
    param = "max_distance"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, np.inf)
    ### test ordinary behaviour ###
    tparams = {param: 1e8}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_max_chunk_size(tmp_path):
    param = "max_chunk_size"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 128)
    ### test ordinary behaviour ###
    tparams = {param: 256}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_chunk_by_rows(tmp_path):
    param = "chunk_by_rows"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, False)
    ### test ordinary behaviour ###
    tparams = {param: True}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_output_tile_files(tmp_path):
    param = "output_tile_files"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, False)
    ### test ordinary behaviour ###
    tparams = {param: True}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_no_data_value(tmp_path):
    param = "no_data_value"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, 0)
    ### test ordinary behaviour ###
    tparams = {param: 1}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_parallelized(tmp_path):
    param = "parallelized"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, True)
    ### test ordinary behaviour ###
    tparams = {param: False}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, "nope", "type")


def test_validate_out_group(tmp_path):
    param = "out_group"
    ### test default behaviour ###
    default_test_generator(tmp_path, param, None)
    ### test ordinary behaviour ###
    tparams = {param: "test"}
    param_test_generator(tmp_path, tparams)
    ### test validation behaviour ###
    catch_invalid_generator(tmp_path, param, 2, "type")
