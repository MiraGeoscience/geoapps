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
from geoh5py.ui_json.utils import requires_value
from geoh5py.workspace import Workspace

from geoapps.inversion.electricals.direct_current.three_dimensions import (
    DirectCurrent3DParams,
)
from geoapps.inversion.electricals.direct_current.three_dimensions.constants import (
    app_initializer as dc_initializer,
)
from geoapps.inversion.electricals.induced_polarization.three_dimensions import (
    InducedPolarization3DParams,
)
from geoapps.inversion.electricals.induced_polarization.three_dimensions.constants import (
    app_initializer as ip_initializer,
)
from geoapps.inversion.potential_fields import (
    GravityParams,
    MagneticScalarParams,
    MagneticVectorParams,
)
from geoapps.inversion.potential_fields.gravity.constants import (
    app_initializer as grav_init,
)
from geoapps.inversion.potential_fields.magnetic_scalar.constants import (
    app_initializer as mag_initializer,
)
from geoapps.inversion.potential_fields.magnetic_vector.constants import (
    app_initializer as mvi_init,
)
from geoapps.octree_creation.constants import app_initializer as octree_initializer
from geoapps.octree_creation.params import OctreeParams
from geoapps.peak_finder.constants import app_initializer as peak_initializer
from geoapps.peak_finder.params import PeakFinderParams

geoh5 = Workspace("./FlinFlon.geoh5")

# Setup
tmpfile = lambda path: os.path.join(path, "test.ui.json")
wrkstr = "FlinFlon.geoh5"
geoh5 = Workspace(wrkstr)


def tmp_input_file(filepath, idict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(idict, f)


mvi_init["geoh5"] = "./FlinFlon.geoh5"
mvi_params = MagneticVectorParams(**mvi_init)


def catch_invalid_generator(param, invalid_value, validation_type):
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


def param_test_generator(param, value):
    setattr(mvi_params, param, value)
    pval = mvi_params.input_file.data[param]
    if hasattr(pval, "uid"):
        pval = pval.uid

    assert pval == value


def test_write_input_file_validation(tmp_path):

    grav_init["geoh5"] = "./FlinFlon.geoh5"
    params = GravityParams(validate=False, **grav_init)
    params.validate = True
    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)

    assert "gz_channel" in str(excinfo.value)


def test_params_initialize():
    for params in [
        MagneticScalarParams(),
        MagneticVectorParams(),
        GravityParams(),
        DirectCurrent3DParams(),
        InducedPolarization3DParams(),
        OctreeParams(),
        PeakFinderParams(),
    ]:
        check = []
        for k, v in params.defaults.items():
            if " " in k or k in [
                "starting_model",
                "conductivity_model",
                "min_value",
            ]:
                continue
            check.append(getattr(params, k) == v)
        assert all(check)

    params = MagneticVectorParams(starting_model=1.0)
    assert params.starting_model == 1.0
    params = GravityParams(starting_model=1.0)
    assert params.starting_model == 1.0
    params = OctreeParams(vertical_padding=500.0)
    assert params.vertical_padding == 500.0
    params = PeakFinderParams(center=1000.0)
    assert params.center == 1000.0


def test_input_file_construction(tmp_path):

    params_classes = [
        GravityParams,
        MagneticScalarParams,
        MagneticVectorParams,
        DirectCurrent3DParams,
        InducedPolarization3DParams,
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
                # TODO Need to better handle defaults None to value
                if (" " in k) or k in [
                    "starting_model",
                    "reference_model",
                    "conductivity_model",
                    "min_value",
                ]:
                    continue
                check.append(getattr(params, k) == v)

            assert all(check)


def test_default_input_file(tmp_path):

    for params_class in [
        MagneticScalarParams,
        MagneticVectorParams,
        GravityParams,
        DirectCurrent3DParams,
        InducedPolarization3DParams,
    ]:
        filename = os.path.join(tmp_path, "test.ui.json")
        params = params_class()
        params.write_input_file(name=filename, path=tmp_path, validate=False)
        ifile = InputFile.read_ui_json(filename, validation_options={"disabled": True})

        # check that reads back into input file with defaults
        check = []
        for k, v in ifile.data.items():
            if " " in k or requires_value(ifile.ui_json, k):
                continue
            check.append(v == params.defaults[k])
        assert all(check)

        # check that params constructed from_path is defaulted
        params2 = params_class()
        check = []
        for k, v in params2.to_dict(ui_json_format=False).items():
            if " " in k or requires_value(ifile.ui_json, k):
                continue
            check.append(v == ifile.data[k])
        assert all(check)


def test_update():
    new_params = {
        "starting_model": 99.0,
    }
    params = MagneticVectorParams()
    params.update(new_params)
    assert params.starting_model == 99.0


def test_chunk_validation_mvi(tmp_path):
    test_dict = dict(mvi_init, **{"geoh5": geoh5})
    test_dict.pop("data_object")
    params = MagneticVectorParams(**test_dict)  # pylint: disable=repeated-keyword
    with pytest.raises(RequiredValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Missing required parameter", "data_object"]:
        assert a in str(excinfo.value)


def test_chunk_validation_mag(tmp_path):
    test_dict = dict(mag_initializer, **{"geoh5": geoh5})
    test_dict["inducing_field_strength"] = None
    params = MagneticScalarParams(**test_dict)  # pylint: disable=repeated-keyword
    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Cannot set a None", "inducing_field_strength"]:
        assert a in str(excinfo.value)


def test_chunk_validation_grav(tmp_path):
    test_dict = dict(grav_init, **{"geoh5": geoh5})
    params = GravityParams(**test_dict)  # pylint: disable=repeated-keyword
    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["Cannot set a None", "gz_channel"]:
        assert a in str(excinfo.value)


def test_chunk_validation_dc(tmp_path):
    with Workspace("./FlinFlon_dcip.geoh5") as dc_geoh5:
        test_dict = dc_initializer.copy()
        test_dict.update({"geoh5": dc_geoh5})
        test_dict.pop("topography_object")
        params = DirectCurrent3DParams(**test_dict)  # pylint: disable=repeated-keyword

        with pytest.raises(OptionalValidationError) as excinfo:
            params.write_input_file(name="test.ui.json", path=tmp_path)
        for a in ["Cannot set a None value", "topography_object"]:
            assert a in str(excinfo.value)


def test_chunk_validation_ip(tmp_path):
    with Workspace("./FlinFlon_dcip.geoh5") as dc_geoh5:
        test_dict = ip_initializer.copy()
        test_dict.update({"geoh5": dc_geoh5})
        test_dict.pop("topography_object")
        params = InducedPolarization3DParams(
            **test_dict
        )  # pylint: disable=repeated-keyword

        with pytest.raises(OptionalValidationError) as excinfo:
            params.write_input_file(name="test.ui.json", path=tmp_path)
        for a in ["Cannot set a None", "topography_object"]:
            assert a in str(excinfo.value)


def test_chunk_validation_octree(tmp_path):
    test_dict = dict(octree_initializer, **{"geoh5": geoh5})
    test_dict.pop("objects")
    params = OctreeParams(**test_dict)  # pylint: disable=repeated-keyword

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["objects"]:
        assert a in str(excinfo.value)


def test_chunk_validation_peakfinder(tmp_path):
    test_dict = dict(peak_initializer, **{"geoh5": geoh5})
    test_dict.pop("data")
    params = PeakFinderParams(**test_dict)  # pylint: disable=repeated-keyword

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["data"]:
        assert a in str(excinfo.value)


def test_active_set():
    test_dict = dict(mvi_init, **{"geoh5": geoh5})
    params = MagneticVectorParams(**test_dict)  # pylint: disable=repeated-keyword
    assert "inversion_type" in params.active_set()
    assert "mesh" in params.active_set()


def test_validate_inversion_type():
    param = "inversion_type"
    newval = "magnetic vector"
    param_test_generator(param, newval)
    catch_invalid_generator(param, "em", "value")


def test_validate_inducing_field_strength():
    param = "inducing_field_strength"
    newval = 60000.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_inducing_field_inclination():
    param = "inducing_field_inclination"
    newval = 44.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_inducing_field_declination():
    param = "inducing_field_declination"
    newval = 9.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_topography_object():
    param = "topography_object"
    newval = UUID("{79b719bc-d996-4f52-9af0-10aa9c7bb941}")
    param_test_generator(param, newval)
    catch_invalid_generator(param, True, "type")
    catch_invalid_generator(param, "lsdkfj", "uuid")
    catch_invalid_generator(param, "", "uuid")


def test_validate_topography():
    param = "topography"
    mvi_params.topography_object = UUID("{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    newval = UUID("{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    param_test_generator(param, newval)
    newval = 1234.0
    param_test_generator(param, newval)
    newval = UUID("{79b719bc-d996-4f52-9af0-10aa9c7bb941}")
    catch_invalid_generator(param, newval, "association")
    newval = "abc"
    catch_invalid_generator(param, newval, "uuid")


def test_validate_data_object():
    param = "data_object"
    newval = UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    param_test_generator(param, newval)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, 2, "type")


def test_validate_tmi_channel():
    param = "tmi_channel"
    newval = UUID("{44822654-b6ae-45b0-8886-2d845f80f422}")
    param_test_generator(param, newval)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, 4, "type")


def test_validate_tmi_uncertainty():
    param = "tmi_uncertainty"
    param_test_generator(param, 1.0)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, {}, "type")


def test_validate_starting_model():
    param = "starting_model"
    param_test_generator(param, 1.0)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, {}, "type")


def test_validate_starting_inclination():
    param = "starting_inclination"
    param_test_generator(param, 1.0)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, {}, "type")


def test_validate_starting_declination():
    param = "starting_declination"
    param_test_generator(param, 1.0)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, {}, "type")


def test_validate_tile_spatial():
    param = "tile_spatial"
    newval = 9
    invalidval = {}
    param_test_generator(param, newval)
    catch_invalid_generator(param, invalidval, "type")


def test_validate_receivers_radar_drape():
    param = "receivers_radar_drape"
    newval = UUID("{44822654-b6ae-45b0-8886-2d845f80f422}")
    param_test_generator(param, newval)
    newval = uuid4()
    catch_invalid_generator(param, newval, "association")
    catch_invalid_generator(param, {}, "type")


def test_validate_receivers_offset_x():
    param = "receivers_offset_x"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_receivers_offset_y():
    param = "receivers_offset_x"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_receivers_offset_z():
    param = "receivers_offset_x"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_ignore_values():
    param = "ignore_values"
    newval = "12345"
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_resolution():
    param = "resolution"
    newval = 10.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_detrend_order():
    param = "detrend_order"
    newval = 2
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_detrend_type():
    param = "detrend_type"
    newval = "perimeter"
    param_test_generator(param, newval)
    catch_invalid_generator(param, "sdf", "value")


def test_validate_max_chunk_size():
    param = "max_chunk_size"
    newval = 256
    param_test_generator(param, newval)
    catch_invalid_generator(param, "asdf", "type")


def test_validate_chunk_by_rows():
    param = "chunk_by_rows"
    newval = True
    param_test_generator(param, newval)
    catch_invalid_generator(param, "sdf", "type")


def test_validate_output_tile_files():
    param = "output_tile_files"
    newval = True
    param_test_generator(param, newval)
    catch_invalid_generator(param, "sdf", "type")


def test_validate_mesh():
    param = "mesh"
    newval = UUID("{c02e0470-0c3e-4119-8ac1-0aacba5334af}")
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_window_center_x():
    param = "window_center_x"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_window_center_y():
    param = "window_center_y"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_window_width():
    param = "window_width"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_window_height():
    param = "window_height"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_inversion_style():
    param = "inversion_style"
    newval = "voxel"
    param_test_generator(param, newval)
    catch_invalid_generator(param, 123, "type")


def test_validate_chi_factor():
    param = "chi_factor"
    newval = 0.5
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_sens_wts_threshold():
    param = "sens_wts_threshold"
    newval = 0.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_every_iteration_bool():
    param = "every_iteration_bool"
    newval = True
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_f_min_change():
    param = "f_min_change"
    newval = 1e-3
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_beta_tol():
    param = "beta_tol"
    newval = 0.2
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_prctile():
    param = "prctile"
    newval = 90
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_coolingRate():
    param = "coolingRate"
    newval = 3
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_coolingFactor():
    param = "coolingFactor"
    newval = 4.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_coolEps_q():
    param = "coolEps_q"
    newval = False
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_coolEpsFact():
    param = "coolEpsFact"
    newval = 1.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_beta_search():
    param = "beta_search"
    newval = True
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_starting_chi_factor():
    param = "starting_chi_factor"
    newval = 2.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_max_global_iterations():
    param = "max_global_iterations"
    newval = 2
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_max_irls_iterations():
    param = "max_irls_iterations"
    newval = 1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_max_cg_iterations():
    param = "max_cg_iterations"
    newval = 2
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_initial_beta():
    param = "initial_beta"
    newval = 2.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_initial_beta_ratio():
    param = "initial_beta_ratio"
    newval = 0.5
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_tol_cg():
    param = "tol_cg"
    newval = 0.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_alpha_s():
    param = "alpha_s"
    newval = 0.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_alpha_x():
    param = "alpha_x"
    newval = 0.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_alpha_y():
    param = "alpha_y"
    newval = 0.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_alpha_z():
    param = "alpha_z"
    newval = 0.1
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_s_norm():
    param = "s_norm"
    newval = 0.5
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_x_norm():
    param = "x_norm"
    newval = 0.5
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_y_norm():
    param = "y_norm"
    newval = 0.5
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_z_norm():
    param = "z_norm"
    newval = 0.5
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_reference_model():
    param = "reference_model"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_reference_inclination():
    param = "reference_inclination"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_reference_declination():
    param = "reference_declination"
    newval = 99.0
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_gradient_type():
    param = "gradient_type"
    newval = "components"
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "value")


def test_validate_lower_bound():
    param = "lower_bound"
    newval = -1000
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_upper_bound():
    param = "upper_bound"
    newval = 1000
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


def test_validate_parallelized():
    param = "parallelized"
    newval = False
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


def test_validate_n_cpu():
    param = "n_cpu"
    newval = 12
    param_test_generator(param, newval)
    catch_invalid_generator(param, "test", "type")


grav_params = GravityParams(
    **{
        "geoh5": "./FlinFlon.geoh5",
        "data_object": UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"),
    }
)


def test_validate_geoh5():
    with pytest.raises(TypeValidationError) as excinfo:
        grav_params.geoh5 = 4

    assert all(
        [k in str(excinfo.value) for k in ["geoh5", "Type", "int", "str", "Workspace"]]
    )


def test_validate_out_group():
    param = "out_group"
    newval = "test_"
    param_test_generator(param, newval)
    catch_invalid_generator(param, {}, "type")


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
    params = DirectCurrent3DParams()
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
    params = DirectCurrent3DParams()
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
    params = DirectCurrent3DParams()
    with pytest.raises(TypeValidationError) as excinfo:
        params.potential_channel_bool = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["potential_channel_bool", "Type", "str", "bool"]
        ]
    )


def test_potential_channel():
    params = DirectCurrent3DParams()
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
    params = DirectCurrent3DParams()
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
    params = InducedPolarization3DParams()
    params.inversion_type = "induced polarization"
    with pytest.raises(ValueValidationError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "induced polarization"]
        ]
    )


def test_chargeability_channel_bool():
    params = InducedPolarization3DParams()
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
    params = InducedPolarization3DParams()
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
    params = InducedPolarization3DParams()
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
    params = InducedPolarization3DParams()
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
    params = InducedPolarization3DParams()
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
    mag_params.starting_model = 0.0
    mag_params.write_input_file(name=file_name, path=tmp_path, validate=False)

    with open(os.path.join(tmp_path, file_name), encoding="utf-8") as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is True, "isValue should be True"

    mag_params.starting_model = mesh.get_data("VTEM_model")[0].uid

    mag_params.write_input_file(name=file_name, path=tmp_path, validate=False)
    with open(os.path.join(tmp_path, file_name), encoding="utf-8") as f:
        ui = json.load(f)

    assert ui["starting_model"]["isValue"] is False, "isValue should be False"
