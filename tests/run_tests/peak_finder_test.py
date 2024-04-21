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

import numpy as np
from geoh5py.objects import Curve
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace
from ipywidgets import Widget

from geoapps.peak_finder.application import PeakFinder, PeakFinderDriver

from .. import PROJECT

# pytest.skip("eliminating conflicting test.", allow_module_level=True)


def test_peak_finder_app(tmp_path: Path):
    app = PeakFinder(geoh5=str(PROJECT), plot_result=False)

    h5file_path = tmp_path / r"testPeakFinder.geoh5"

    # Create temp workspace
    ws = Workspace.create(h5file_path)

    x = np.arange(-2 * np.pi + np.pi / 4, 2 * np.pi, np.pi / 32)

    curve = Curve.create(ws, vertices=np.c_[x, np.zeros((x.shape[0], 2))])

    for ii in range(5):
        c = curve.add_data(
            {f"d{ii}": {"values": np.sin(x + np.pi / 8.0 * ii) - 0.1 * ii}}
        )
        curve.add_data_to_group(c, property_group="obs")

    line = curve.add_data(
        {
            "line_id": {
                "values": np.ones_like(x),
                "value_map": {1: "1", 2: "2", 3: "3"},
                "type": "referenced",
            }
        }
    )
    curve.add_data_to_group(line, property_group="Line")
    changes = {
        "objects": curve.uid,
        "data": curve.find_or_create_property_group(name="obs").uid,
        "line_field": line.uid,
        "line_id": 1,
        "width": 10.0,
        "center": 1.0,
        "min_amplitude": 1e-2,
        "min_width": 1e-2,
        "max_migration": 1.0,
        "group_auto": True,
    }
    app.geoh5 = ws

    for param, value in changes.items():
        if isinstance(getattr(app, param), Widget):
            getattr(app, param).value = value
        else:
            setattr(app, param, value)

    app.trigger_click(None)

    anomalies = app.lines.anomalies
    assert len(anomalies) == 3, f"Expected 3 groups. Found {len(anomalies)}"
    assert all(
        [aa["azimuth"] == 270 for aa in anomalies]
    ), "Anomalies with wrong azimuth found"
    assert [aa["channel_group"]["label"][0] for aa in anomalies] == [
        4,
        5,
        3,
    ], "Grouping different than expected"


def test_peak_finder_driver(tmp_path: Path):
    uijson_path = tmp_path.parent / "test_peak_finder_app0" / "Temp"
    json_file = next(uijson_path.glob("*.ui.json"))
    driver = PeakFinderDriver.start(str(uijson_path / json_file))

    with driver.params.geoh5.open(mode="r"):
        results = driver.params.geoh5.get_entity("PointMarkers")
        compare_entities(results[0], results[1], ignore=["_uid"])
