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

from geoapps.io import InputFile
from geoapps.io.Gravity import GravityParams
from geoapps.io.MVI import MVIParams
from geoapps.io.MVI.constants import default_ui_json as MVI_defaults
from geoapps.io.MVI.constants import validations as MVI_validations
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
#     geotest = setup_params(tmp_path, MVI_defaults, MVIParams)
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
    ifile.write_ui_json(MVI_defaults, default=True, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    if isinstance(ui[param], dict):
        ui[param]["value"] = invalid_value
        ui[param]["visible"] = True
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
        MVIParams.from_path(filepath, workspace=workspace)

    for a in assertions:
        assert a in str(excinfo.value)


def param_test_generator(tmp_path, param, value, workspace=workspace):
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    wrkstr = workspace.h5file
    ifile.write_ui_json(MVI_defaults, default=True, workspace=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    if isinstance(ui[param], dict):
        if "isValue" in ui[param].keys():
            if isinstance(value, UUID):
                ui[param]["isValue"] = False
            else:
                ui[param]["isValue"] = True
        ui[param]["value"] = value
        ui[param]["visible"] = True
        ui[param]["enabled"] = True
    else:
        ui[param] = value
    ui["geoh5"] = None
    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)
    params = MVIParams.from_path(filepath, workspace=workspace)

    try:
        value = UUID(str(value))
    except ValueError:
        pass

    pval = getattr(params, param)

    if param == "out_group":
        pval = pval.name

    assert pval == value


def test_params_initialize():
    for params in [MVIParams(), GravityParams(), OctreeParams(), PeakFinderParams()]:
        check = []
        for k, v in params.defaults.items():
            if " " in k:
                continue
                check.append(getattr(params, k) == v)
        assert all(check)

    params = MVIParams(u_cell_size=9999, validate=True, workspace=workspace)
    assert params.u_cell_size == 9999
    params = GravityParams(u_cell_size=9999, validate=True, workspace=workspace)
    assert params.u_cell_size == 9999
    params = OctreeParams(vertical_padding=500, validate=True, workspace=workspace)
    assert params.vertical_padding == 500
    params = PeakFinderParams(center=1000, validate=True, workspace=workspace)
    assert params.center == 1000


def test_params_constructors(tmp_path):
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    ui = deepcopy(MVI_defaults)
    ui["geoh5"] = wrkstr
    ifile.write_ui_json(ui, default=True)
    params1 = MVIParams.from_path(filepath, workspace=workspace)
    params2 = MVIParams.from_input_file(ifile, workspace=workspace)


def test_param_names():
    assert np.all(MVIParams.param_names == list(MVI_defaults.keys()))


def test_active_set():
    params = MVIParams(workspace=workspace, inversion_type="mvi", u_cell_size=2)
    params.active_set()


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "mvic"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "em", "value", workspace=workspace)
    # catch_invalid_generator(tmp_path, param, "mvi", "reqs", workspace=workspace)


def test_validate_forward_only(tmp_path):
    param = "forward_only"
    newval = False
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, "test", "type", workspace=workspace)
    # catch_invalid_generator(tmp_path, param, True, "reqs", workspace=workspace)


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
    catch_invalid_generator(tmp_path, param, "lsdkfj", "uuid", workspace=workspace)
    catch_invalid_generator(
        tmp_path, param, "", "uuid", workspace=workspace, parent="topography_object"
    )


def test_validate_data_object(tmp_path):
    param = "data_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_tmi_channel(tmp_path):
    param = "tmi_channel"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


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


def test_validate_gps_receivers_offset(tmp_path):
    param = "gps_receivers_offset"
    newval = str(uuid4())
    invalidval = {}
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, invalidval, "type", workspace=workspace)


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


def test_validate_detrend_data(tmp_path):
    param = "detrend_data"
    newval = True
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_detrend_order(tmp_path):
    param = "detrend_order"
    newval = 2
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, 9, "value", workspace=workspace)


def test_validate_detrend_type(tmp_path):
    param = "detrend_type"
    newval = "corners"
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


def test_validate_mesh_from_params(tmp_path):
    param = "mesh_from_params"
    newval = True
    # param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "type", workspace=workspace)


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


def test_validate_smallness_norm(tmp_path):
    param = "smallness_norm"
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
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_output_geoh5(tmp_path):
    param = "output_geoh5"
    newval = "../assets/something.geoh5py"
    # param_test_generator(tmp_path, param, newval)
    # catch_invalid_generator(tmp_path, param, 34, "type")


def test_validate_out_group(tmp_path):
    param = "out_group"
    newval = "test_"
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_validate_no_data_value(tmp_path):
    param = "no_data_value"
    newval = 5
    param_test_generator(tmp_path, param, newval, workspace=workspace)
    catch_invalid_generator(tmp_path, param, {}, "type", workspace=workspace)


def test_isValue(tmp_path):
    # "starting_model"
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath

    mesh = workspace.get_entity("O2O_Interp_25m")[0]

    params = MVIParams.from_input_file(ifile, workspace)
    params.starting_model_object = mesh.uid
    params.starting_model = 0.0

    params.write_input_file()

    with open(filepath) as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is True, "isValue should be True"

    params.starting_model = mesh.get_data("VTEM_model")[0].uid

    params.write_input_file()
    with open(filepath) as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is False, "isValue should be False"
