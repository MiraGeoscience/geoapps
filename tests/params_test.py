# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from pathlib import Path

import pytest
from geoh5py.shared.exceptions import OptionalValidationError
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps.peak_finder.constants import app_initializer as peak_initializer
from geoapps.peak_finder.params import PeakFinderParams

from . import PROJECT

# Setup
geoh5 = Workspace(PROJECT)


def test_params_initialize():
    for params in [
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

    params = PeakFinderParams(center=1000.0)
    assert params.center == 1000.0


def test_input_file_construction(tmp_path: Path):
    params_classes = [
        PeakFinderParams,
    ]

    for params_class in params_classes:
        filename = "test.ui.json"
        for forward_only in [True, False]:
            params = params_class(forward_only=forward_only)
            params.write_input_file(name=filename, path=tmp_path, validate=False)
            ifile = InputFile.read_ui_json(tmp_path / filename, validate=False)
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


def test_chunk_validation_peakfinder(tmp_path: Path):
    test_dict = dict(peak_initializer, **{"geoh5": geoh5})
    test_dict.pop("data")
    params = PeakFinderParams(**test_dict)  # pylint: disable=repeated-keyword

    with pytest.raises(OptionalValidationError) as excinfo:
        params.write_input_file(name="test.ui.json", path=tmp_path)
    for a in ["data"]:
        assert a in str(excinfo.value)
