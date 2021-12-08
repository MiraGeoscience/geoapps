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


# def test_inversion_type(tmp_path):
#     geotest = setup_params(tmp_path, MVI_defaults, MagneticVectorParams)
#     geotest.set_param("inversion_type", "nogood")
#     ws, params = geotest.make()
#     assert True


######################  Setup  ###########################

tmpfile = lambda path: os.path.join(path, "test.ui.json")
wrkstr = "FlinFlon.geoh5"
geoh5 = Workspace(wrkstr)


def tmp_input_file(filepath, idict):
    with open(filepath, "w") as f:
        json.dump(idict, f)


def catch_invalid_generator(
    tmp_path, param, invalid_value, validation_type, geoh5=None, parent=None
):

    key_map = {
        "value": "values",
        "type": "types",
        "shape": "shapes",
        "reqs": "reqs",
        "uuid": "uuid",
    }
    validator_opts = {"ignore_requirements": True}
    pvalidations = MVI_validations[param][key_map[validation_type]]
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    ifile.write_ui_json(MVI_defaults, geoh5=wrkstr)
    with open(filepath) as f:
        ui = json.load(f)
    if isinstance(ui[param], dict):
        ui[param]["value"] = invalid_value
        ui[param]["enabled"] = True
        ui[param]["isValue"] = True
    else:
        ui[param] = invalid_value
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
        validator_opts = {}
        assertions = ["Unsatisfied", param]
        req = pvalidations[0]
        hasval = len(req) > 1
        preq = req[1] if hasval else req[0]
        ui[preq]["value"] = None
        ui[preq]["enabled"] = False
        assertions += [str(k) for k in req]
    elif validation_type == "uuid":
        err = (ValueError, IndexError)
        if geoh5 is None:
            assertions = [param, "uuid", invalid_value, "valid uuid"]
        if geoh5 is not None and parent is None:
            uuid_str = str(uuid4())
            ui[param]["value"] = uuid_str
            assertions = [param, "uuid", uuid_str, "Address does"]
        if geoh5 is not None and parent is not None:
            ui[param]["value"] = "{c02e0470-0c3e-4119-8ac1-0aacba5334af}"
            ui[param]["parent"] = parent
            ui[parent]["value"] = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"

            assertions = [param, "uuid", invalid_value, "child of"]

    with open(filepath, "w") as f:
        json.dump(ui, f, indent=4)

    with pytest.raises(err) as excinfo:
        ifile = InputFile(filepath)
        MagneticVectorParams(
            ifile, geoh5=geoh5, validate=True, validator_opts=validator_opts
        )

    for a in assertions:
        assert a in str(excinfo.value)


def param_test_generator(tmp_path, param, value, geoh5=geoh5):
    filepath = tmpfile(tmp_path)
    ifile = InputFile()
    ifile.filepath = filepath
    wrkstr = geoh5.h5file
    ifile.write_ui_json(MVI_defaults, geoh5=wrkstr)
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
    params = MagneticVectorParams(ifile, validate=False, geoh5=geoh5)

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

    params = MagneticVectorParams(u_cell_size=9999, validate=False, geoh5=geoh5)
    assert params.u_cell_size == 9999
    params = GravityParams(u_cell_size=9999, validate=False, geoh5=geoh5)
    assert params.u_cell_size == 9999
    params = OctreeParams(vertical_padding=500, validate=False, geoh5=geoh5)
    assert params.vertical_padding == 500
    params = PeakFinderParams(center=1000, validate=False, geoh5=geoh5)
    assert params.center == 1000


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
        params2 = params_class(
            ifile, validate=True, validator_opts={"ignore_requirements": True}
        )
        check = []
        for k, v in params2.to_dict(ui_json_format=False).items():
            if " " in k:
                continue
            check.append(v == ifile.data[k])
        assert all(check)

        # check that params constructed from_input_file is defaulted
        params3 = params_class(
            ifile, validate=True, validator_opts={"ignore_requirements": True}
        )
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
        input_file=InputFile(filepath), validate=False, geoh5=geoh5
    )
    params2 = MagneticVectorParams(input_file=ifile, validate=False, geoh5=geoh5)


def test_chunk_validation():

    from geoapps.io.MagneticVector.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = MagneticVectorParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict.pop("data_object")
        params = MagneticVectorParams(**test_dict)
    for a in ["Missing required", "data_object"]:
        assert a in str(excinfo.value)

    from geoapps.io.MagneticScalar.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = MagneticScalarParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict["inducing_field_strength"] = None
        params = MagneticScalarParams(**test_dict)
    for a in ["Missing required", "inducing_field_strength"]:
        assert a in str(excinfo.value)

    from geoapps.io.Gravity.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = GravityParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict.pop("starting_model")
        params = GravityParams(**test_dict)
    for a in ["Missing required", "starting_model"]:
        assert a in str(excinfo.value)

    from geoapps.io.DirectCurrent.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = DirectCurrentParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict.pop("topography_object")
        params = DirectCurrentParams(**test_dict)
    for a in ["Missing required", "topography_object"]:
        assert a in str(excinfo.value)

    from geoapps.io.InducedPolarization.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = InducedPolarizationParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict.pop("conductivity_model")
        params = InducedPolarizationParams(**test_dict)
    for a in ["Missing required", "conductivity_model"]:
        assert a in str(excinfo.value)

    from geoapps.io.Octree.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = OctreeParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict.pop("objects")
        params = OctreeParams(**test_dict)
    for a in ["Missing required", "objects"]:
        assert a in str(excinfo.value)

    from geoapps.io.PeakFinder.constants import app_initializer

    test_dict = dict(app_initializer, **{"geoh5": geoh5})
    params = PeakFinderParams(**test_dict)
    with pytest.raises(ValueError) as excinfo:
        test_dict.pop("data")
        params = PeakFinderParams(**test_dict)
    for a in ["Missing required", "data"]:
        assert a in str(excinfo.value)


def test_active_set():
    params = MagneticVectorParams(
        default=False,
        validate=False,
        forward_only=True,
        geoh5=geoh5,
        inversion_type="magnetic vector",
        u_cell_size=2,
    )
    assert "inversion_type" in params.active_set()
    assert "u_cell_size" in params.active_set()


def test_validate_inversion_type(tmp_path):
    param = "inversion_type"
    newval = "magnetic scalar"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "em", "value", geoh5=geoh5)


def test_validate_inducing_field_strength(tmp_path):
    param = "inducing_field_strength"
    newval = 60000
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_inducing_field_inclination(tmp_path):
    param = "inducing_field_inclination"
    newval = 44
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_inducing_field_declination(tmp_path):
    param = "inducing_field_declination"
    newval = 9
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_topography_object(tmp_path):
    param = "topography_object"
    newval = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, True, "type", geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "lsdkfj", "uuid", geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "", "uuid", geoh5=geoh5)


def test_validate_topography(tmp_path):
    param = "topography"
    newval = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, True, "type", geoh5=geoh5)


def test_validate_data_object(tmp_path):
    param = "data_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, 2, "type", geoh5=geoh5)


def test_validate_tmi_channel(tmp_path):
    param = "tmi_channel"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, 4, "type", geoh5=geoh5)


def test_validate_tmi_uncertainty(tmp_path):
    param = "tmi_uncertainty"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_starting_model_object(tmp_path):
    param = "starting_model_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_starting_inclination_object(tmp_path):
    param = "starting_inclination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_starting_declination_object(tmp_path):
    param = "starting_declination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_starting_model(tmp_path):
    param = "starting_model"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_starting_inclination(tmp_path):
    param = "starting_inclination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_starting_declination(tmp_path):
    param = "starting_declination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_tile_spatial(tmp_path):
    param = "tile_spatial"
    newval = 9
    invalidval = {}
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, invalidval, "type", geoh5=geoh5)


def test_validate_receivers_radar_drape(tmp_path):
    param = "receivers_radar_drape"
    newval = str(uuid4())
    invalidval = {}
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, invalidval, "type", geoh5=geoh5)


def test_validate_receivers_offset_x(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_receivers_offset_y(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_receivers_offset_z(tmp_path):
    param = "receivers_offset_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_ignore_values(tmp_path):
    param = "ignore_values"
    newval = "12345"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_resolution(tmp_path):
    param = "resolution"
    newval = 10
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_detrend_order(tmp_path):
    param = "detrend_order"
    newval = 2
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_detrend_type(tmp_path):
    param = "detrend_type"
    newval = "perimeter"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "sdf", "value", geoh5=geoh5)


def test_validate_max_chunk_size(tmp_path):
    param = "max_chunk_size"
    newval = 256
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "asdf", "type", geoh5=geoh5)


def test_validate_chunk_by_rows(tmp_path):
    param = "chunk_by_rows"
    newval = True
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "sdf", "type", geoh5=geoh5)


def test_validate_output_tile_files(tmp_path):
    param = "output_tile_files"
    newval = True
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "sdf", "type", geoh5=geoh5)


def test_validate_mesh(tmp_path):
    param = "mesh"
    newval = "{c02e0470-0c3e-4119-8ac1-0aacba5334af}"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_u_cell_size(tmp_path):
    param = "u_cell_size"
    newval = 9
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "sdf", "type", geoh5=geoh5)


def test_validate_v_cell_size(tmp_path):
    param = "v_cell_size"
    newval = 9
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "sdf", "type", geoh5=geoh5)


def test_validate_w_cell_size(tmp_path):
    param = "w_cell_size"
    newval = 9
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "sdf", "type", geoh5=geoh5)


def test_validate_octree_levels_topo(tmp_path):
    param = "octree_levels_topo"
    newval = [1, 2, 3]
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_octree_levels_obs(tmp_path):
    param = "octree_levels_obs"
    newval = [1, 2, 3]
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_depth_core(tmp_path):
    param = "depth_core"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_max_distance(tmp_path):
    param = "max_distance"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_horizontal_padding(tmp_path):
    param = "horizontal_padding"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_vertical_padding(tmp_path):
    param = "vertical_padding"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_window_center_x(tmp_path):
    param = "window_center_x"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_window_center_y(tmp_path):
    param = "window_center_y"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_window_width(tmp_path):
    param = "window_width"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_window_height(tmp_path):
    param = "window_height"
    newval = 99
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_inversion_style(tmp_path):
    param = "inversion_style"
    newval = "voxel"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "value", geoh5=geoh5)


def test_validate_chi_factor(tmp_path):
    param = "chi_factor"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_max_iterations(tmp_path):
    param = "max_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_max_cg_iterations(tmp_path):
    param = "max_cg_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_max_global_iterations(tmp_path):
    param = "max_global_iterations"
    newval = 2
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_initial_beta(tmp_path):
    param = "initial_beta"
    newval = 2
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_initial_beta_ratio(tmp_path):
    param = "initial_beta_ratio"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_tol_cg(tmp_path):
    param = "tol_cg"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_alpha_s(tmp_path):
    param = "alpha_s"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_alpha_x(tmp_path):
    param = "alpha_x"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_alpha_y(tmp_path):
    param = "alpha_y"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_alpha_z(tmp_path):
    param = "alpha_z"
    newval = 0.1
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_s_norm(tmp_path):
    param = "s_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_x_norm(tmp_path):
    param = "x_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_y_norm(tmp_path):
    param = "y_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_z_norm(tmp_path):
    param = "z_norm"
    newval = 0.5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_reference_model_object(tmp_path):
    param = "reference_model_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_reference_inclination_object(tmp_path):
    param = "reference_inclination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_reference_declination_object(tmp_path):
    param = "reference_declination_object"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_reference_model(tmp_path):
    param = "reference_model"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_reference_inclination(tmp_path):
    param = "reference_inclination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_reference_declination(tmp_path):
    param = "reference_declination"
    newval = str(uuid4())
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_gradient_type(tmp_path):
    param = "gradient_type"
    newval = "components"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "value", geoh5=geoh5)


def test_validate_lower_bound(tmp_path):
    param = "lower_bound"
    newval = -1000
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_upper_bound(tmp_path):
    param = "upper_bound"
    newval = 1000
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_parallelized(tmp_path):
    param = "parallelized"
    newval = False
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_n_cpu(tmp_path):
    param = "n_cpu"
    newval = 12
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_max_ram(tmp_path):
    param = "max_ram"
    newval = 10
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "test", "type", geoh5=geoh5)


def test_validate_geoh5(tmp_path):
    param = "geoh5"
    newval = os.path.join(tmp_path, "something.geoh5py")
    params = MagneticVectorParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.geoh5 = newval
    with pytest.raises(TypeError) as excinfo:
        params.geoh5 = 4

    assert all(
        [k in str(excinfo.value) for k in ["geoh5", "type", "int", "str", "Workspace"]]
    )


def test_validate_out_group(tmp_path):
    param = "out_group"
    newval = "test_"
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, {}, "type", geoh5=geoh5)


def test_validate_no_data_value(tmp_path):
    param = "no_data_value"
    newval = 5
    param_test_generator(tmp_path, param, newval, geoh5=geoh5)
    catch_invalid_generator(tmp_path, param, "lskjdf", "type", geoh5=geoh5)


def test_gravity_inversion_type():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.inversion_type = "gravity"
    with pytest.raises(ValueError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["inversion_type", "alskdj", "gravity"]]
    )


def test_gz_channel_bool():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gz_channel_bool", "type", "str", "bool"]]
    )


def test_gz_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gz_channel = str(uuid4())
    params.gz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gz_channel", "type", "int", "str", "UUID"]]
    )


def test_gz_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gz_uncertainty = str(uuid4())
    params.gz_uncertainty = uuid4()
    params.gz_uncertainty = 4
    params.gz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gz_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.guv_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.guv_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["guv_channel_bool", "type", "str", "bool"]]
    )


def test_guv_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.guv_channel = str(uuid4())
    params.guv_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.guv_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["guv_channel", "type", "int", "str", "UUID"]]
    )


def test_guv_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.guv_uncertainty = str(uuid4())
    params.guv_uncertainty = uuid4()
    params.guv_uncertainty = 4
    params.guv_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.guv_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gxy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxy_channel_bool", "type", "str", "bool"]]
    )


def test_gxy_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxy_channel = str(uuid4())
    params.gxy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gxy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxy_channel", "type", "int", "str", "UUID"]]
    )


def test_gxy_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxy_uncertainty = str(uuid4())
    params.gxy_uncertainty = uuid4()
    params.gxy_uncertainty = 4
    params.gxy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gxy_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxx_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gxx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxx_channel_bool", "type", "str", "bool"]]
    )


def test_gxx_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxx_channel = str(uuid4())
    params.gxx_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gxx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxx_channel", "type", "int", "str", "UUID"]]
    )


def test_gxx_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxx_uncertainty = str(uuid4())
    params.gxx_uncertainty = uuid4()
    params.gxx_uncertainty = 4
    params.gxx_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gxx_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gyy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gyy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gyy_channel_bool", "type", "str", "bool"]]
    )


def test_gyy_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gyy_channel = str(uuid4())
    params.gyy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gyy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gyy_channel", "type", "int", "str", "UUID"]]
    )


def test_gyy_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gyy_uncertainty = str(uuid4())
    params.gyy_uncertainty = uuid4()
    params.gyy_uncertainty = 4
    params.gyy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gyy_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gzz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gzz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gzz_channel_bool", "type", "str", "bool"]]
    )


def test_gzz_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gzz_channel = str(uuid4())
    params.gzz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gzz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gzz_channel", "type", "int", "str", "UUID"]]
    )


def test_gzz_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gzz_uncertainty = str(uuid4())
    params.gzz_uncertainty = uuid4()
    params.gzz_uncertainty = 4
    params.gzz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gzz_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gxz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gxz_channel_bool", "type", "str", "bool"]]
    )


def test_gxz_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxz_channel = str(uuid4())
    params.gxz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gxz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gxz_channel", "type", "int", "str", "UUID"]]
    )


def test_gxz_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gxz_uncertainty = str(uuid4())
    params.gxz_uncertainty = uuid4()
    params.gxz_uncertainty = 4
    params.gxz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gxz_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gyz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gyz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gyz_channel_bool", "type", "str", "bool"]]
    )


def test_gyz_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gyz_channel = str(uuid4())
    params.gyz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gyz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gyz_channel", "type", "int", "str", "UUID"]]
    )


def test_gyz_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gyz_uncertainty = str(uuid4())
    params.gyz_uncertainty = uuid4()
    params.gyz_uncertainty = 4
    params.gyz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gyz_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gx_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gx_channel_bool", "type", "str", "bool"]]
    )


def test_gx_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gx_channel = str(uuid4())
    params.gx_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gx_channel", "type", "int", "str", "UUID"]]
    )


def test_gx_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gx_uncertainty = str(uuid4())
    params.gx_uncertainty = uuid4()
    params.gx_uncertainty = 4
    params.gx_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gx_uncertainty = geoh5

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
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.gy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["gy_channel_bool", "type", "str", "bool"]]
    )


def test_gy_channel():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gy_channel = str(uuid4())
    params.gy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.gy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["gy_channel", "type", "int", "str", "UUID"]]
    )


def test_gy_uncertainty():
    params = GravityParams(validate=True, validator_opts={"ignore_requirements": True})
    params.gy_uncertainty = str(uuid4())
    params.gy_uncertainty = uuid4()
    params.gy_uncertainty = 4
    params.gy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.gy_uncertainty = geoh5

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


def test_magnetic_scalar_inversion_type():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.inversion_type = "magnetic scalar"
    with pytest.raises(ValueError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "magnetic scalar"]
        ]
    )


def test_inducing_field_strength():
    params = MagneticScalarParams(validate=False)
    params.inducing_field_strength = 1.0
    params.inducing_field_strength = 1

    with pytest.raises(TypeError) as excinfo:
        params.inducing_field_strength = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inducing_field_strength", "type", "str", "float"]
        ]
    )


def test_inducing_field_inclination():
    params = MagneticScalarParams(validate=False)
    params.inducing_field_inclination = 1.0
    params.inducing_field_inclination = 1

    with pytest.raises(TypeError) as excinfo:
        params.inducing_field_inclination = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inducing_field_inclination", "type", "str", "float"]
        ]
    )


def test_inducing_field_declination():
    params = MagneticScalarParams(validate=False)
    params.inducing_field_declination = 1.0
    params.inducing_field_declination = 1

    with pytest.raises(TypeError) as excinfo:
        params.inducing_field_declination = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inducing_field_declination", "type", "str", "float"]
        ]
    )


def test_tmi_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.tmi_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.tmi_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["tmi_channel_bool", "type", "str", "bool"]]
    )


def test_tmi_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.tmi_channel = str(uuid4())
    params.tmi_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.tmi_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["tmi_channel", "type", "int", "str", "UUID"]]
    )


def test_tmi_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.tmi_uncertainty = str(uuid4())
    params.tmi_uncertainty = uuid4()
    params.tmi_uncertainty = 4
    params.tmi_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.tmi_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "tmi_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bxx_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxx_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.bxx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bxx_channel_bool", "type", "str", "bool"]]
    )


def test_bxx_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxx_channel = str(uuid4())
    params.bxx_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.bxx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bxx_channel", "type", "int", "str", "UUID"]]
    )


def test_bxx_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxx_uncertainty = str(uuid4())
    params.bxx_uncertainty = uuid4()
    params.bxx_uncertainty = 4
    params.bxx_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.bxx_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bxx_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bxy_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.bxy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bxy_channel_bool", "type", "str", "bool"]]
    )


def test_bxy_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxy_channel = str(uuid4())
    params.bxy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.bxy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bxy_channel", "type", "int", "str", "UUID"]]
    )


def test_bxy_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxy_uncertainty = str(uuid4())
    params.bxy_uncertainty = uuid4()
    params.bxy_uncertainty = 4
    params.bxy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.bxy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bxy_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bxz_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.bxz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bxz_channel_bool", "type", "str", "bool"]]
    )


def test_bxz_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxz_channel = str(uuid4())
    params.bxz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.bxz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bxz_channel", "type", "int", "str", "UUID"]]
    )


def test_bxz_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bxz_uncertainty = str(uuid4())
    params.bxz_uncertainty = uuid4()
    params.bxz_uncertainty = 4
    params.bxz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.bxz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bxz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_byy_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.byy_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.byy_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["byy_channel_bool", "type", "str", "bool"]]
    )


def test_byy_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.byy_channel = str(uuid4())
    params.byy_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.byy_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["byy_channel", "type", "int", "str", "UUID"]]
    )


def test_byy_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.byy_uncertainty = str(uuid4())
    params.byy_uncertainty = uuid4()
    params.byy_uncertainty = 4
    params.byy_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.byy_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "byy_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_byz_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.byz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.byz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["byz_channel_bool", "type", "str", "bool"]]
    )


def test_byz_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.byz_channel = str(uuid4())
    params.byz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.byz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["byz_channel", "type", "int", "str", "UUID"]]
    )


def test_byz_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.byz_uncertainty = str(uuid4())
    params.byz_uncertainty = uuid4()
    params.byz_uncertainty = 4
    params.byz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.byz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "byz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bzz_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bzz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.bzz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bzz_channel_bool", "type", "str", "bool"]]
    )


def test_bzz_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bzz_channel = str(uuid4())
    params.bzz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.bzz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bzz_channel", "type", "int", "str", "UUID"]]
    )


def test_bzz_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bzz_uncertainty = str(uuid4())
    params.bzz_uncertainty = uuid4()
    params.bzz_uncertainty = 4
    params.bzz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.bzz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bzz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bx_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bx_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.bx_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bx_channel_bool", "type", "str", "bool"]]
    )


def test_bx_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bx_channel = str(uuid4())
    params.bx_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.bx_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bx_channel", "type", "int", "str", "UUID"]]
    )


def test_bx_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bx_uncertainty = str(uuid4())
    params.bx_uncertainty = uuid4()
    params.bx_uncertainty = 4
    params.bx_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.bx_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bx_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_by_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.by_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.by_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["by_channel_bool", "type", "str", "bool"]]
    )


def test_by_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.by_channel = str(uuid4())
    params.by_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.by_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["by_channel", "type", "int", "str", "UUID"]]
    )


def test_by_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.by_uncertainty = str(uuid4())
    params.by_uncertainty = uuid4()
    params.by_uncertainty = 4
    params.by_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.by_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "by_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_bz_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bz_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.bz_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["bz_channel_bool", "type", "str", "bool"]]
    )


def test_bz_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bz_channel = str(uuid4())
    params.bz_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.bz_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["bz_channel", "type", "int", "str", "UUID"]]
    )


def test_bz_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.bz_uncertainty = str(uuid4())
    params.bz_uncertainty = uuid4()
    params.bz_uncertainty = 4
    params.bz_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.bz_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "bz_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_tmi_channel_bool():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.tmi_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.tmi_channel_bool = "alskdj"

    assert all(
        [s in str(excinfo.value) for s in ["tmi_channel_bool", "type", "str", "bool"]]
    )


def test_tmi_channel():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.tmi_channel = str(uuid4())
    params.tmi_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.tmi_channel = 4

    assert all(
        [s in str(excinfo.value) for s in ["tmi_channel", "type", "int", "str", "UUID"]]
    )


def test_tmi_uncertainty():
    params = MagneticScalarParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.tmi_uncertainty = str(uuid4())
    params.tmi_uncertainty = uuid4()
    params.tmi_uncertainty = 4
    params.tmi_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.tmi_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "tmi_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_direct_current_inversion_type():
    params = DirectCurrentParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.inversion_type = "direct current"
    with pytest.raises(ValueError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "direct current"]
        ]
    )


def test_direct_current_data_object():
    params = DirectCurrentParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.data_object = uuid4()

    with pytest.raises(TypeError) as excinfo:
        params.data_object = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["data_object", "type", "int", "UUID", "PotentialElectrode"]
        ]
    )


def test_potential_channel_bool():
    params = DirectCurrentParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.potential_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.potential_channel_bool = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["potential_channel_bool", "type", "str", "bool"]
        ]
    )


def test_potential_channel():
    params = DirectCurrentParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.potential_channel = str(uuid4())
    params.potential_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.potential_channel = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["potential_channel", "type", "int", "str", "UUID"]
        ]
    )


def test_potential_uncertainty():
    params = DirectCurrentParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.potential_uncertainty = str(uuid4())
    params.potential_uncertainty = uuid4()
    params.potential_uncertainty = 4
    params.potential_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.potential_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "potential_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def test_induced_polarization_inversion_type():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.inversion_type = "induced polarization"
    with pytest.raises(ValueError) as excinfo:
        params.inversion_type = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["inversion_type", "alskdj", "induced polarization"]
        ]
    )


def test_direct_current_data_object():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.data_object = uuid4()

    with pytest.raises(TypeError) as excinfo:
        params.data_object = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["data_object", "type", "int", "UUID", "PotentialElectrode"]
        ]
    )


def test_chargeability_channel_bool():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.chargeability_channel_bool = True
    with pytest.raises(TypeError) as excinfo:
        params.chargeability_channel_bool = "alskdj"

    assert all(
        [
            s in str(excinfo.value)
            for s in ["chargeability_channel_bool", "type", "str", "bool"]
        ]
    )


def test_chargeability_channel():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.chargeability_channel = str(uuid4())
    params.chargeability_channel = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.chargeability_channel = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["chargeability_channel", "type", "int", "str", "UUID"]
        ]
    )


def test_chargeability_uncertainty():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.chargeability_uncertainty = str(uuid4())
    params.chargeability_uncertainty = uuid4()
    params.chargeability_uncertainty = 4
    params.chargeability_uncertainty = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.chargeability_uncertainty = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "chargeability_uncertainty",
                "type",
                "Workspace",
                "str",
                "int",
                "float",
                "UUID",
            ]
        ]
    )


def conductivity_model_object():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.conductivity_model_object = str(uuid4())
    params.conductivity_model_object = uuid4()
    with pytest.raises(TypeError) as excinfo:
        params.conductivity_model_object = 4

    assert all(
        [
            s in str(excinfo.value)
            for s in ["conductivity_model_object", "type", "int", "str", "UUID"]
        ]
    )


def test_conductivity_model():
    params = InducedPolarizationParams(
        validate=True, validator_opts={"ignore_requirements": True}
    )
    params.conductivity_model = str(uuid4())
    params.conductivity_model = uuid4()
    params.conductivity_model = 4
    params.conductivity_model = 4.0
    with pytest.raises(TypeError) as excinfo:
        params.conductivity_model = geoh5

    assert all(
        [
            s in str(excinfo.value)
            for s in [
                "conductivity_model",
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

    mesh = geoh5.get_entity("O2O_Interp_25m")[0]

    params = MagneticVectorParams(input_file=ifile, validate=False, geoh5=geoh5)
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
