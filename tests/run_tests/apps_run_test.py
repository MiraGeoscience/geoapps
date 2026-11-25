# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# pylint: disable=protected-access

from __future__ import annotations

import base64
import pathlib
import uuid
from pathlib import Path

import numpy as np
import pytest
from curve_apps import assets_path as curve_apps_assets_path
from geoh5py.data import FilenameData
from geoh5py.objects import Curve, Surface
from geoh5py.ui_json import UIJson
from geoh5py.workspace import Workspace
from surface_apps import assets_path as surface_apps_assets_path

from geoapps import assets_path
from geoapps.block_model_creation.application import BlockModelCreation
from geoapps.calculator import Calculator
from geoapps.contours.application import ContourValues
from geoapps.coordinate_transformation import CoordinateTransformation
from geoapps.edge_detection.application import EdgeDetectionApp
from geoapps.export.application import Export
from geoapps.interpolation.application import DataInterpolation
from geoapps.iso_surfaces.application import IsoSurface
from geoapps.octree_creation.application import OctreeMesh
from geoapps.peak_finder.application import PeakFinder
from geoapps.scatter_plot.application import ScatterPlots
from geoapps.triangulated_surfaces.application import Surface2D
from geoapps.utils.testing import get_output_workspace
from tests import PROJECT, PROJECT_TEM


# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

GEOH5 = Workspace(PROJECT)


def test_block_model(tmp_path: Path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in ["{2e814779-c35f-4da0-ad6a-39a6912361f9}"]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    block_model = BlockModelCreation(geoh5=str(temp_workspace))
    # Test initialization
    object_options, objects_uid, ui_json_data, _, _ = block_model.update_object_options(
        None, None, param_name="objects", trigger=""
    )
    param_list = [
        "export_as",
        "cell_size_x",
        "cell_size_y",
        "cell_size_z",
        "depth_core",
        "horizontal_padding",
        "bottom_padding",
        "expansion_factor",
        "monitoring_directory",
    ]
    (  # pylint: disable=W0632
        new_name,
        cell_size_x,
        cell_size_y,
        cell_size_z,
        depth_core,
        horizontal_padding,
        bottom_padding,
        expansion_factor,
        _,
    ) = block_model.update_remainder_from_ui_json(
        ui_json_data, param_list, trigger="test"
    )

    assert new_name == block_model.params.output.export_as
    assert objects_uid == "{" + str(block_model.params.source.objects.uid) + "}"
    assert cell_size_x == block_model.params.creation.cell_size_x
    assert cell_size_y == block_model.params.creation.cell_size_y
    assert cell_size_z == block_model.params.creation.cell_size_z
    assert depth_core == block_model.params.creation.depth_core
    assert horizontal_padding == block_model.params.creation.horizontal_padding
    assert bottom_padding == block_model.params.creation.bottom_padding
    assert expansion_factor == block_model.params.creation.expansion_factor

    # Create a second workspace to test file uploads
    temp_workspace2 = tmp_path / "contour2.geoh5"
    with Workspace(temp_workspace2) as workspace:
        for uid in ["{2e814779-c35f-4da0-ad6a-39a6912361f9}"]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)
    # Reproduce output of dcc.Upload Component
    with open(temp_workspace2, "rb") as file:
        decoded = file.read()
    content_bytes = base64.b64encode(decoded)
    content_string = content_bytes.decode("utf-8")
    contents = "".join(["content_type", ",", content_string])
    object_options, _, _, _, _ = block_model.update_object_options(
        "ws.geoh5", contents, param_name="objects", trigger="upload"
    )

    # Test export
    block_model.trigger_click(
        n_clicks=0,
        new_grid=new_name,
        objects=object_options[0]["value"],
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        cell_size_z=cell_size_z,
        depth_core=depth_core,
        horizontal_padding=horizontal_padding,
        bottom_padding=bottom_padding,
        expansion_factor=expansion_factor,
        live_link=[],
        monitoring_directory=str(tmp_path),
        trigger="export",
    )

    filename = next(tmp_path.glob("BlockModel_*.geoh5"))
    with Workspace(filename) as workspace:
        ent = workspace.get_entity(new_name)
        assert (len(ent) == 1) and (ent[0] is not None)
        assert np.sum([isinstance(c, FilenameData) for c in ent[0].children]) == 1


def test_calculator(tmp_path: Path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace(temp_workspace) as workspace:
        GEOH5.get_entity("geochem")[0].copy(parent=workspace)

    app = Calculator(geoh5=str(temp_workspace))
    app.trigger.click()

    with Workspace(get_output_workspace(tmp_path / "Temp")) as workspace:
        output = workspace.get_entity("NewChannel")[0]
        assert output.values.shape[0] == 4438, "Change in output. Need to verify."


def test_coordinate_transformation(tmp_path: Path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(temp_workspace) as workspace:
        GEOH5.get_entity("Gravity_Magnetics_drape60m")[0].copy(parent=workspace)
        GEOH5.get_entity("Data_TEM_pseudo3D")[0].copy(parent=workspace)

    app = CoordinateTransformation(geoh5=str(temp_workspace))
    app.trigger.click()

    with Workspace(get_output_workspace(tmp_path / "Temp")) as workspace:
        assert len(workspace.objects) == 3, "Coordinate transform failed."


def test_geoh5_as_contour_argument(tmp_path):
    geoh5 = Workspace.create(tmp_path / "test.geoh5")
    ContourValues(geoh5=geoh5, plot_result=False)


def test_contour_values(tmp_path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace(temp_workspace) as workspace:
        objects = GEOH5.get_entity(uuid.UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[
            0
        ]
        grid = objects.copy(parent=workspace, copy_children=True)
        data = workspace.get_entity(
            uuid.UUID("{44822654-b6ae-45b0-8886-2d845f80f422}")
        )[0]
        ifile = UIJson.read(curve_apps_assets_path() / "uijson/contours.ui.json")
        ifile.set_values(
            geoh5=workspace,
            objects=grid.uid,
            data=data.uid,
            interval_min=-400.0,
            interval_max=2000.0,
            interval_spacing=100.0,
            fixed_contours="-240",
        )

    app = ContourValues(ui_json=ifile, plot_result=False)
    app.export_directory._set_form_values(str(tmp_path), "")
    app.export_directory._apply_selection()
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        output = next(k for k in workspace.objects if isinstance(k, Curve))
        assert output.n_vertices == 4984, "Change in output. Need to verify."
        # TODO the double value here is due to package conflicts locally and on github
        # Once we switch from matplotlib to scikit-image, this should be a single value.
        assert (
            len([child for child in output.children if isinstance(child, FilenameData)])
            == 1
        )


def test_create_surface(tmp_path: Path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{5fa66412-3a4c-440c-8b87-6f10cb5f1c7f}",
        ]:
            new_obj = GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = Surface2D(geoh5=str(temp_workspace))

    app.data.value = [p_g.uid for p_g in new_obj.property_groups if p_g.name == "COND"]
    app.elevations.data.value = [
        p_g.uid for p_g in new_obj.property_groups if p_g.name == "ELEV"
    ][0]
    app.export_directory._set_form_values(str(tmp_path), "")
    app.export_directory._apply_selection()
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        surf = workspace.get_entity("CDI_")[0]
        assert surf.n_vertices == 40000


def test_data_interpolation(tmp_path: Path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
            "{f3e36334-be0a-4210-b13e-06933279de25}",
            "{7450be38-1327-4336-a9e4-5cff587b6715}",
            "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
        ]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = DataInterpolation(geoh5=str(temp_workspace))
    app.export_directory._set_form_values(str(tmp_path), "")
    app.export_directory._apply_selection()
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path / "Temp")) as workspace:
        assert len(workspace.get_entity("Iteration_7_model_Interp")) == 1


def test_edge_detection(tmp_path: Path):
    temp_workspace = tmp_path / "edge_detection.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
        ]:
            new_copy = GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)
            grid = new_copy.copy(copy_children=False)
            new_data = grid.add_data(
                {
                    "copy_data": {
                        "values": new_copy.children[0].values,
                        "association": "CELL",
                    }
                }
            )

    app = EdgeDetectionApp(plot_result=False)
    app._file_browser.reset(
        path=tmp_path,
        filename="edge_detection.geoh5",
    )
    app._file_browser._apply_selection()
    app.file_browser_change(None)
    app.objects.value = grid.uid
    app.data.value = new_data.uid
    app.compute_trigger(None)
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path / "Temp")) as workspace:
        assert (
            len(
                [
                    child
                    for child in workspace.get_entity("copy_data")
                    if isinstance(child, Curve)
                ]
            )
            == 1
        )


def test_edge_detection_workspace_options():
    app = EdgeDetectionApp(plot_result=False)
    flin_flon = assets_path() / "FlinFlon.geoh5"
    assert app.workspace.h5file == flin_flon  # Uses flinflon
    app.workspace.close()

    h5file = str(assets_path() / "FlinFlon_dcip.geoh5")
    app = EdgeDetectionApp(geoh5=h5file, plot_result=False)
    assert str(app.workspace.h5file) == h5file  # Hardwires to dcip
    app.workspace.close()

    h5file = str(assets_path() / "FlinFlon_borked.geoh5")
    with pytest.warns(match="Path provided in 'geoh5' argument does not exist."):
        app = EdgeDetectionApp(geoh5=h5file, plot_result=False)
    assert app.workspace.h5file == flin_flon  # Falls back on flinflon
    app.workspace.close()


def test_export():
    app = Export(geoh5=PROJECT)

    # Test exporting mesh
    model_uid = uuid.UUID("eddc5be1-4753-4a41-99a0-c8f11a252e32")
    app.objects.value = uuid.UUID("7450be38-1327-4336-a9e4-5cff587b6715")
    app.data.value = [model_uid]
    app.file_type.value = "UBC format"
    app.trigger.click()

    assert pathlib.Path(
        app.export_directory.value + app.export_as.value + ".msh"
    ).exists()
    assert pathlib.Path(
        app.export_directory.value + app.data.uid_name_map[model_uid] + ".mod"
    ).exists()

    # TODO write all the files types and check that appropriate files are written


def test_iso_surface(tmp_path: Path):
    temp_workspace = tmp_path / f"{__name__}.geoh5"
    with Workspace(temp_workspace) as workspace:
        objects = GEOH5.get_entity(uuid.UUID("{2e814779-c35f-4da0-ad6a-39a6912361f9}"))[
            0
        ]
        objects.copy(parent=workspace, copy_children=True)
        data = workspace.get_entity(
            uuid.UUID("{f3e36334-be0a-4210-b13e-06933279de25}")
        )[0]
        ifile = UIJson.read(surface_apps_assets_path() / "uijson/iso_surfaces.ui.json")
        ifile.set_values(
            geoh5=workspace,
            objects=objects,
            data=data,
            max_distance=500.0,
            resolution=50.0,
            interval_min=0.005,
            interval_max=0.02,
            interval_spacing=0.005,
        )

    app = IsoSurface(ui_json=ifile)
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path / "Temp")) as workspace:
        group = workspace.get_entity("ISO")[0]
        assert len(group.children) == 5
        assert np.sum([isinstance(c, FilenameData) for c in group.children]) == 1
        assert np.sum([isinstance(c, Surface) for c in group.children]) == 4

    app.fixed_contours.value = "1000."

    with pytest.warns(UserWarning, match="The following levels were"):
        app.trigger_click(None)


def test_peak_finder():
    app = PeakFinder(geoh5=PROJECT_TEM)
    app()
    assert app.objects.value == uuid.UUID("{34698019-cde6-4b43-8d53-a040b25c989a}")
    assert app.data.value == uuid.UUID("{22a9cf91-5cff-42b5-8bbb-2f1c6a559204}")


def test_octree_creation(tmp_path):
    app = OctreeMesh()
    app.export_directory._selected_path = str(tmp_path)
    app.trigger_click(None)
    assert len([file for file in tmp_path.iterdir()]) == 2


def test_scatter_plot(tmp_path):
    app = ScatterPlots()
    figure = app.update_plot("", trigger="")
    app.trigger_click(1, str(tmp_path), figure, trigger="export")

    assert len([file for file in tmp_path.iterdir()]) == 3
