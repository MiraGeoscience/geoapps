#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
from uuid import UUID, uuid4

import pytest
from geoh5py.shared.exceptions import (
    AssociationValidationError,
    OptionalValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import optional_type
from geoh5py.workspace import Workspace

from geoapps.drivers.inversion.direct_current.params import DirectCurrentParams
from geoapps.drivers.inversion.gravity import GravityParams
from geoapps.drivers.inversion.gravity import app_initializer as grav_init
from geoapps.drivers.inversion.induced_polarization import InducedPolarizationParams
from geoapps.drivers.inversion.magnetic_scalar import MagneticScalarParams
from geoapps.drivers.inversion.magnetic_vector.constants import (
    app_initializer as mvi_init,
)
from geoapps.drivers.inversion.magnetic_vector.params import MagneticVectorParams
from geoapps.drivers.octree.params import OctreeParams
from geoapps.drivers.peak_finder.params import PeakFinderParams
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(tmp, ui, params_class):
    geotest = Geoh5Tester(geoh5, tmp, "test.geoh5", ui, params_class)
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("gz_channel", "{6de9177a-8277-4e17-b76c-2b8b05dcf23c}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    return geotest


######################  Setup  ###########################

tmpfile = lambda path: os.path.join(path, "test.ui.json")
wrkstr = "FlinFlon.geoh5"
geoh5 = Workspace(wrkstr)


def tmp_input_file(filepath, idict):
    with open(filepath, "w") as f:
        json.dump(idict, f)


mvi_init["geoh5"] = "./FlinFlon.geoh5"
mvi_params = MagneticVectorParams(**mvi_init)


def catch_invalid_generator(
    tmp_path, param, invalid_value, validation_type, geoh5=None, parent=None
):
    filepath = tmpfile(tmp_path)

    if validation_type == "value":
        err = ValueValidationError
    elif validation_type == "type":
        err = TypeValidationError
    elif validation_type == "shape":
        err = ShapeValidationError
    elif validation_type == "required":
        err = RequiredValidationError
    elif validation_type == "uuid":
        err = UUIDValidationError
    elif validation_type == "association":
        err = AssociationValidationError

    with pytest.raises(err):
        setattr(mvi_params, param, invalid_value)


def param_test_generator(tmp_path, param, value):
    filepath = tmpfile(tmp_path)
    setattr(mvi_params, param, value)
    pval = mvi_params.input_file.data[param]
    if hasattr(pval, "uid"):
        pval = pval.uid

    assert pval == value


def test_write_input_file_validation(tmp_path):

    grav_init["geoh5"] = "./FlinFlon.geoh5"
    params = GravityParams(validate=False, **grav_init)
    params.starting_model = None
    params.validate = True
    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)

    assert "starting_model" in str(excinfo.value)


def test_params_initialize():
    for params in [
        MagneticScalarParams(),
        MagneticVectorParams(),
        GravityParams(),
        DirectCurrentParams(),
        InducedPolarizationParams(),
        OctreeParams(),
        PeakFinderParams(),
    ]:
        check = []
        for k, v in params.defaults.items():
            if " " in k:
                continue
                check.append(getattr(params, k) == v)
        assert all(check)

    params = MagneticVectorParams(u_cell_size=9999.0)
    assert params.u_cell_size == 9999.0
    params = GravityParams(u_cell_size=9999.0)
    assert params.u_cell_size == 9999.0
    params = OctreeParams(vertical_padding=500.0)
    assert params.vertical_padding == 500.0
    params = PeakFinderParams(center=1000.0)
    assert params.center == 1000.0


def test_input_file_construction(tmp_path):

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
            params = params_class(forward_only=forward_only)
            params.write_input_file(name=filename, path=tmp_path, validate=False)
            ifile = InputFile.read_ui_json(
                os.path.join(tmp_path, filename), validation_options={"disabled": True}
            )
            params = params_class(input_file=ifile)

            check = []
            for k, v in params.defaults.items():
                if " " in k:
                    continue
                    check.append(getattr(params, k) == v)

            assert all(check)


def test_default_input_file(tmp_path):

    for params_class in [
        MagneticScalarParams,
        MagneticVectorParams,
        GravityParams,
        DirectCurrentParams,
        InducedPolarizationParams,
    ]:
        filename = os.path.join(tmp_path, "test.ui.json")
        params = params_class()
        params.write_input_file(name=filename, path=tmp_path, validate=False)
        ifile = InputFile.read_ui_json(filename, validation_options={"disabled": True})

        # check that reads back into input file with defaults
        check = []
        for k, v in ifile.data.items():
            if " " in k or not optional_type(ifile.ui_json, k):
                continue
            check.append(v == params.defaults[k])
        assert all(check)

        # check that params constructed from_path is defaulted
        params2 = params_class()
        check = []
        for k, v in params2.to_dict(ui_json_format=False).items():
            if " " in k or not optional_type(ifile.ui_json, k):
                continue
            check.append(v == ifile.data[k])
        assert all(check)


def test_update(tmp_path):
    new_params = {
        "u_cell_size": 5.0,
    }
    params = MagneticVectorParams()
    params.update(new_params)
    assert params.u_cell_size == 5.0


def test_chunk_validation(tmp_path):

    from geoapps.drivers.inversion.magnetic_vector.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    test_dict.pop("data_object")
    params = MagneticVectorParams(**test_dict)
    with pytest.raises(RequiredValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Missing required parameter", "data_object"]:
        assert a in str(excinfo.value)

    from geoapps.drivers.inversion.magnetic_scalar import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    test_dict["inducing_field_strength"] = None
    params = MagneticScalarParams(**test_dict)
    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Cannot set a None", "inducing_field_strength"]:
        assert a in str(excinfo.value)

    from geoapps.drivers.inversion.gravity import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    test_dict.pop("starting_model")
    params = GravityParams(**test_dict)

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Cannot set a None", "starting_model"]:
        assert a in str(excinfo.value)

    from geoapps.drivers.inversion.direct_current import app_initializer

    dc_geoh5 = Workspace("FlinFlon_dcip.geoh5")
    test_dict = dict(app_initializer, **{"geoh5": dc_geoh5})
    test_dict.pop("topography_object")
    params = DirectCurrentParams(**test_dict)

    with pytest.raises(RequiredValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Missing required parameter", "topography_object"]:
        assert a in str(excinfo.value)

    from geoapps.drivers.inversion.induced_polarization import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": dc_geoh5})
    test_dict.pop("conductivity_model")
    params = InducedPolarizationParams(**test_dict)

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Cannot set a None", "conductivity_model"]:
        assert a in str(excinfo.value)

    from geoapps.drivers.octree.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    test_dict.pop("objects")
    params = OctreeParams(**test_dict)

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["objects"]:
        assert a in str(excinfo.value)

    from geoapps.drivers.peak_finder.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    test_dict.pop("data")
    params = PeakFinderParams(**test_dict)

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["data"]:
        assert a in str(excinfo.value)


def test_active_set():
    from geoapps.drivers.inversion.magnetic_vector.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = MagneticVectorParams(**test_dict)
    assert "inversion_type" in params.active_set()
    assert "u_cell_size" in params.active_set()


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "magnetic vector"
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "em", "value")


def test_validate_inducing_field_strength(tmp_path):
    param = "inducing_field_strength"
    newval = 60000.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_inducing_field_inclination(tmp_path):
    param = "inducing_field_inclination"
    newval = 44.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_inducing_field_declination(tmp_path):
    param = "inducing_field_declination"
    newval = 9.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_topography_object(tmp_path):
    param = "topography_object"
    newval = UUID("{79b719bc-d996-4f52-9af0-10aa9c7bb941}")
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, True, "type")
    catch_invalid_generator(tmp_path, param, "lsdkfj", "uuid")
    catch_invalid_generator(tmp_path, param, "", "uuid")


def test_validate_topography(tmp_path):
    param = "topography"
    mvi_params.topography_object = UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    newval = UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    param_test_generator(tmp_path, param, newval)
    newval = 1234.0
    param_test_generator(tmp_path, param, newval)
    newval = UUID("{79b719bc-d996-4f52-9af0-10aa9c7bb941}")
    catch_invalid_generator(tmp_path, param, newval, "association")
    newval = "abc"
    catch_invalid_generator(tmp_path, param, newval, "uuid")


def test_validate_data_object(tmp_path):
    param = "data_object"
    newval = UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    param_test_generator(tmp_path, param, newval)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, 2, "type")


def test_validate_tmi_channel(tmp_path):
    param = "tmi_channel"
    newval = UUID("{44822654-b6ae-45b0-8886-2d845f80f422}")
    param_test_generator(tmp_path, param, newval)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, 4, "type")


def test_validate_tmi_uncertainty(tmp_path):
    param = "tmi_uncertainty"
    param_test_generator(tmp_path, param, 1.0)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_model_object(tmp_path):
    param = "starting_model_object"
    newval = UUID("{e334f687-df71-4538-ad28-264e420210b8}")
    param_test_generator(tmp_path, param, newval)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_inclination_object(tmp_path):
    param = "starting_inclination_object"
    newval = UUID("{e334f687-df71-4538-ad28-264e420210b8}")
    param_test_generator(tmp_path, param, newval)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_declination_object(tmp_path):
    param = "starting_declination_object"
    newval = UUID("{e334f687-df71-4538-ad28-264e420210b8}")
    param_test_generator(tmp_path, param, newval)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_model(tmp_path):
    param = "starting_model"
    mvi_params.starting_model_object = UUID("{e334f687-df71-4538-ad28-264e420210b8}")
    param_test_generator(tmp_path, param, 1.0)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_inclination(tmp_path):
    param = "starting_inclination"
    mvi_params.starting_model_object = UUID("{e334f687-df71-4538-ad28-264e420210b8}")
    param_test_generator(tmp_path, param, 1.0)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_starting_declination(tmp_path):
    param = "starting_declination"
    mvi_params.starting_model_object = UUID("{e334f687-df71-4538-ad28-264e420210b8}")
    param_test_generator(tmp_path, param, 1.0)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_tile_spatial(tmp_path):
    param = "tile_spatial"
    newval = 9
    invalidval = {}
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, invalidval, "type")


def test_validate_receivers_radar_drape(tmp_path):
    param = "receivers_radar_drape"
    newval = UUID("{44822654-b6ae-45b0-8886-2d845f80f422}")
    param_test_generator(tmp_path, param, newval)
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_receivers_offset_x(tmp_path):
    param = "receivers_offset_x"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_receivers_offset_y(tmp_path):
    param = "receivers_offset_x"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_receivers_offset_z(tmp_path):
    param = "receivers_offset_x"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_ignore_values(tmp_path):
    param = "ignore_values"
    newval = "12345"
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_resolution(tmp_path):
    param = "resolution"
    newval = 10.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_detrend_order(tmp_path):
    param = "detrend_order"
    newval = 2
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_detrend_type(tmp_path):
    param = "detrend_type"
    newval = "perimeter"
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "value")


def test_validate_max_chunk_size(tmp_path):
    param = "max_chunk_size"
    newval = 256
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "asdf", "type")


def test_validate_chunk_by_rows(tmp_path):
    param = "chunk_by_rows"
    newval = True
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_output_tile_files(tmp_path):
    param = "output_tile_files"
    newval = True
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_mesh(tmp_path):
    param = "mesh"
    newval = UUID("{c02e0470-0c3e-4119-8ac1-0aacba5334af}")
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_u_cell_size(tmp_path):
    param = "u_cell_size"
    newval = 9.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_v_cell_size(tmp_path):
    param = "v_cell_size"
    newval = 9.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_w_cell_size(tmp_path):
    param = "w_cell_size"
    newval = 9.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "sdf", "type")


def test_validate_octree_levels_topo(tmp_path):
    param = "octree_levels_topo"
    newval = [1, 2, 3]
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_octree_levels_obs(tmp_path):
    param = "octree_levels_obs"
    newval = [1, 2, 3]
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_depth_core(tmp_path):
    param = "depth_core"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_max_distance(tmp_path):
    param = "max_distance"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_horizontal_padding(tmp_path):
    param = "horizontal_padding"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_vertical_padding(tmp_path):
    param = "vertical_padding"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_center_x(tmp_path):
    param = "window_center_x"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_center_y(tmp_path):
    param = "window_center_y"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_width(tmp_path):
    param = "window_width"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_window_height(tmp_path):
    param = "window_height"
    newval = 99.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    newval = "voxel"
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, 123, "type")


def test_validate_chi_factor(tmp_path):
    param = "chi_factor"
    newval = 0.5
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_iterations(tmp_path):
    param = "max_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_cg_iterations(tmp_path):
    param = "max_cg_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_max_global_iterations(tmp_path):
    param = "max_global_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_initial_beta(tmp_path):
    param = "initial_beta"
    newval = 2.0
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_initial_beta_ratio(tmp_path):
    param = "initial_beta_ratio"
    newval = 0.5
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_tol_cg(tmp_path):
    param = "tol_cg"
    newval = 0.1
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_s(tmp_path):
    param = "alpha_s"
    newval = 0.1
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_x(tmp_path):
    param = "alpha_x"
    newval = 0.1
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_y(tmp_path):
    param = "alpha_y"
    newval = 0.1
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_alpha_z(tmp_path):
    param = "alpha_z"
    newval = 0.1
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_s_norm(tmp_path):
    param = "s_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_x_norm(tmp_path):
    param = "x_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_y_norm(tmp_path):
    param = "y_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_z_norm(tmp_path):
    param = "z_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_reference_model_object(tmp_path):
    param = "reference_model_object"
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_inclination_object(tmp_path):
    param = "reference_inclination_object"
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_declination_object(tmp_path):
    param = "reference_declination_object"
    newval = uuid4()
    catch_invalid_generator(tmp_path, param, newval, "association")
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_model(tmp_path):
    param = "reference_model"
    newval = uuid4()
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_inclination(tmp_path):
    param = "reference_inclination"
    newval = uuid4()
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_reference_declination(tmp_path):
    param = "reference_declination"
    newval = uuid4()
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_gradient_type(tmp_path):
    param = "gradient_type"
    newval = "components"
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "value")


def test_validate_lower_bound(tmp_path):
    param = "lower_bound"
    newval = -1000
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_upper_bound(tmp_path):
    param = "upper_bound"
    newval = 1000
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_validate_parallelized(tmp_path):
    param = "parallelized"
    newval = False
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


def test_validate_n_cpu(tmp_path):
    param = "n_cpu"
    newval = 12
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, "test", "type")


grav_params = GravityParams(
    **{
        "geoh5": "./FlinFlon.geoh5",
        "data_object": UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
    }
)


def test_validate_geoh5(tmp_path):
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.geoh5 = 4

    assert all(
        [k in str(excinfo.value) for k in ["geoh5", "Type", "int", "str", "Workspace"]]
    )


def test_validate_out_group(tmp_path):
    param = "out_group"
    newval = "test_"
    param_test_generator(tmp_path, param, newval)
    catch_invalid_generator(tmp_path, param, {}, "type")


def test_gravity_inversion_type():
    with pytest.raises(ValueValidationError) as excinfo:
        grav_params.inversion_type = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["inversion_type", "alskdj", "gravity"]]
    )


def test_gz_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gz_channel_bool", "Type", "str", "bool"]]
    )


def test_gz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gz_channel", "Type", "int", "str", "UUID"]]
    )


def test_gz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gz_uncertainty = uuid4()

    grav_params.gz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_guv_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.guv_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["guv_channel_bool", "Type", "str", "bool"]]
    )


def test_guv_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.guv_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.guv_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["guv_channel", "Type", "int", "str", "UUID"]]
    )


def test_guv_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.guv_uncertainty = uuid4()

    grav_params.guv_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.guv_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "guv_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gxy_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxy_channel_bool", "Type", "str", "bool"]]
    )


def test_gxy_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gxy_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxy_channel", "Type", "int", "str", "UUID"]]
    )


def test_gxy_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gxy_uncertainty = uuid4()

    grav_params.gxy_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gxy_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gxx_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxx_channel_bool", "Type", "str", "bool"]]
    )


def test_gxx_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gxx_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxx_channel", "Type", "int", "str", "UUID"]]
    )


def test_gxx_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gxx_uncertainty = uuid4()

    grav_params.gxx_uncertainty = 4
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxx_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gxx_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gyy_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gyy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gyy_channel_bool", "Type", "str", "bool"]]
    )


def test_gyy_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gyy_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gyy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gyy_channel", "Type", "int", "str", "UUID"]]
    )


def test_gyy_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gyy_uncertainty = uuid4()

    grav_params.gyy_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gyy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gyy_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gzz_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gzz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gzz_channel_bool", "Type", "str", "bool"]]
    )


def test_gzz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gzz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gzz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gzz_channel", "Type", "int", "str", "UUID"]]
    )


def test_gzz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gzz_uncertainty = uuid4()

    grav_params.gzz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gzz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gzz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gxz_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxz_channel_bool", "Type", "str", "bool"]]
    )


def test_gxz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gxz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxz_channel", "Type", "int", "str", "UUID"]]
    )


def test_gxz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gxz_uncertainty = uuid4()

    grav_params.gxz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gxz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gxz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gyz_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gyz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gyz_channel_bool", "Type", "str", "bool"]]
    )


def test_gyz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gyz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gyz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gyz_channel", "Type", "int", "str", "UUID"]]
    )


def test_gyz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gyz_uncertainty = uuid4()

    grav_params.gyz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gyz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gyz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gx_channel_bool():

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gx_channel_bool", "Type", "str", "bool"]]
    )


def test_gx_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gx_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gx_channel", "Type", "int", "str", "UUID"]]
    )


def test_gx_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gx_uncertainty = uuid4()

    grav_params.gx_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gx_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gx_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_gy_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gy_channel_bool", "Type", "str", "bool"]]
    )


def test_gy_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gy_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gy_channel", "Type", "int", "str", "UUID"]]
    )


def test_gy_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        grav_params.gy_uncertainty = uuid4()

    grav_params.gy_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.gy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "gy_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


mag_params = MagneticScalarParams(
    **{
        "geoh5": "./FlinFlon.geoh5",
        "data_object": UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
    }
)


def test_magnetic_scalar_inversion_type():
    with pytest.raises(ValueValidationError) as excinfo:
        mag_params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "magnetic scalar"]
        ]
    )


def test_inducing_field_strength():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.inducing_field_strength = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inducing_field_strength", "Type", "str", "float"]
        ]
    )


def test_inducing_field_inclination():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.inducing_field_inclination = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inducing_field_inclination", "Type", "str", "float"]
        ]
    )


def test_inducing_field_declination():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.inducing_field_declination = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inducing_field_declination", "Type", "str", "float"]
        ]
    )


def test_tmi_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.tmi_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["tmi_channel_bool", "Type", "str", "bool"]]
    )


def test_tmi_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.tmi_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.tmi_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["tmi_channel", "Type", "int", "str", "UUID"]]
    )


def test_tmi_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.tmi_uncertainty = uuid4()

    mag_params.tmi_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.tmi_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "tmi_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bxx_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bxx_channel_bool", "Type", "str", "bool"]]
    )


def test_bxx_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bxx_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bxx_channel", "Type", "int", "str", "UUID"]]
    )


def test_bxx_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bxx_uncertainty = uuid4()

    mag_params.bxx_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxx_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bxx_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bxy_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bxy_channel_bool", "Type", "str", "bool"]]
    )


def test_bxy_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bxy_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bxy_channel", "Type", "int", "str", "UUID"]]
    )


def test_bxy_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bxy_uncertainty = uuid4()

    mag_params.bxy_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bxy_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bxz_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bxz_channel_bool", "Type", "str", "bool"]]
    )


def test_bxz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bxz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bxz_channel", "Type", "int", "str", "UUID"]]
    )


def test_bxz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bxz_uncertainty = uuid4()

    mag_params.bxz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bxz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bxz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_byy_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.byy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["byy_channel_bool", "Type", "str", "bool"]]
    )


def test_byy_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.byy_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.byy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["byy_channel", "Type", "int", "str", "UUID"]]
    )


def test_byy_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.byy_uncertainty = uuid4()

    mag_params.byy_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.byy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "byy_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_byz_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.byz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["byz_channel_bool", "Type", "str", "bool"]]
    )


def test_byz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.byz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.byz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["byz_channel", "Type", "int", "str", "UUID"]]
    )


def test_byz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.byz_uncertainty = uuid4()

    mag_params.byz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.byz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "byz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bzz_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bzz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bzz_channel_bool", "Type", "str", "bool"]]
    )


def test_bzz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bzz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bzz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bzz_channel", "Type", "int", "str", "UUID"]]
    )


def test_bzz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bzz_uncertainty = uuid4()

    mag_params.bzz_uncertainty = 4.0

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bzz_uncertainty = geoh5
    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bzz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bx_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bx_channel_bool", "Type", "str", "bool"]]
    )


def test_bx_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bx_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bx_channel", "Type", "int", "str", "UUID"]]
    )


def test_bx_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bx_uncertainty = uuid4()
    mag_params.bx_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bx_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bx_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_by_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.by_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["by_channel_bool", "Type", "str", "bool"]]
    )


def test_by_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.by_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.by_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["by_channel", "Type", "int", "str", "UUID"]]
    )


def test_by_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.by_uncertainty = uuid4()

    mag_params.by_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.by_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "by_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bz_channel_bool():
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bz_channel_bool", "Type", "str", "bool"]]
    )


def test_bz_channel():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bz_channel = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bz_channel", "Type", "int", "str", "UUID"]]
    )


def test_bz_uncertainty():
    with pytest.raises(AssociationValidationError) as excinfo:
        mag_params.bz_uncertainty = uuid4()

    mag_params.bz_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        mag_params.bz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bz_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_direct_current_inversion_type():
    params = DirectCurrentParams()
    params.inversion_type = "direct current"
    with pytest.raises(ValueValidationError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "direct current"]
        ]
    )


def test_direct_current_data_object():
    params = DirectCurrentParams()
    params.data_object = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        params.data_object = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["data_object", "Type", "int", "UUID", "PotentialElectrode"]
        ]
    )


def test_potential_channel_bool():
    params = DirectCurrentParams()
    with pytest.raises(TypeValidationError) as excinfo:
        params.potential_channel_bool = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["potential_channel_bool", "Type", "str", "bool"]
        ]
    )


def test_potential_channel():
    params = DirectCurrentParams()
    params.potential_channel = uuid4()
    params.potential_channel = uuid4()
    with pytest.raises(TypeValidationError) as excinfo:
        params.potential_channel = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["potential_channel", "Type", "int", "str", "UUID"]
        ]
    )


def test_potential_uncertainty():
    params = DirectCurrentParams()
    params.potential_uncertainty = uuid4()
    params.potential_uncertainty = uuid4()
    params.potential_uncertainty = 4
    params.potential_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        params.potential_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "potential_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_induced_polarization_inversion_type():
    params = InducedPolarizationParams()
    params.inversion_type = "induced polarization"
    with pytest.raises(ValueValidationError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "induced polarization"]
        ]
    )


def test_direct_current_data_object():
    params = InducedPolarizationParams()
    params.data_object = uuid4()

    with pytest.raises(TypeValidationError) as excinfo:
        params.data_object = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["data_object", "Type", "int", "UUID", "PotentialElectrode"]
        ]
    )


def test_chargeability_channel_bool():
    params = InducedPolarizationParams()
    params.chargeability_channel_bool = True
    with pytest.raises(TypeValidationError) as excinfo:
        params.chargeability_channel_bool = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["chargeability_channel_bool", "Type", "str", "bool"]
        ]
    )


def test_chargeability_channel():
    params = InducedPolarizationParams()
    params.chargeability_channel = uuid4()
    params.chargeability_channel = uuid4()
    with pytest.raises(TypeValidationError) as excinfo:
        params.chargeability_channel = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["chargeability_channel", "Type", "int", "str", "UUID"]
        ]
    )


def test_chargeability_uncertainty():
    params = InducedPolarizationParams()
    params.chargeability_uncertainty = uuid4()
    params.chargeability_uncertainty = uuid4()
    params.chargeability_uncertainty = 4
    params.chargeability_uncertainty = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        params.chargeability_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "chargeability_uncertainty",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def conductivity_model_object():
    params = InducedPolarizationParams()
    params.conductivity_model_object = uuid4()
    params.conductivity_model_object = uuid4()
    with pytest.raises(TypeValidationError) as excinfo:
        params.conductivity_model_object = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["conductivity_model_object", "Type", "int", "str", "UUID"]
        ]
    )


def test_conductivity_model():
    params = InducedPolarizationParams()
    params.conductivity_model = uuid4()
    params.conductivity_model = uuid4()
    params.conductivity_model = 4
    params.conductivity_model = 4.0
    with pytest.raises(TypeValidationError) as excinfo:
        params.conductivity_model = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "conductivity_model",
                "Type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_isValue(tmp_path):
    file_name = "test.ui.json"
    mesh = geoh5.get_entity("O2O_Interp_25m")[0]
    mag_params.starting_model_object = mesh.uid
    mag_params.starting_model = 0.0
    mag_params.write_input_file(name=file_name, path=tmp_path, validate=False)

    with open(os.path.join(tmp_path, file_name)) as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is True, "isValue should be True"

    mag_params.starting_model = mesh.get_data("VTEM_model")[0].uid

    mag_params.write_input_file(name=file_name, path=tmp_path, validate=False)
    with open(os.path.join(tmp_path, file_name)) as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is False, "isValue should be False"
