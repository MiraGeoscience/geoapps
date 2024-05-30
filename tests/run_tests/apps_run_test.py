# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
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
from dash._callback_context import context_value
from dash._utils import AttributeDict
from geoh5py.data import FilenameData
from geoh5py.objects import Curve, Surface
from geoh5py.workspace import Workspace

from geoapps.block_model_creation.application import BlockModelCreation
from geoapps.calculator import Calculator
from geoapps.clustering import ClusteringParams
from geoapps.clustering.application import Clustering
from geoapps.contours.application import ContourValues
from geoapps.coordinate_transformation import CoordinateTransformation
from geoapps.edge_detection.application import EdgeDetectionApp
from geoapps.export.application import Export
from geoapps.interpolation.application import DataInterpolation
from geoapps.iso_surfaces.application import IsoSurface
from geoapps.peak_finder.application import PeakFinder
from geoapps.triangulated_surfaces.application import Surface2D
from geoapps.utils.testing import get_output_workspace
from tests import PROJECT

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

GEOH5 = Workspace(PROJECT)


def test_block_model(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in ["{2e814779-c35f-4da0-ad6a-39a6912361f9}"]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    block_model = BlockModelCreation(geoh5=str(temp_workspace))
    # Test initialization
    object_options, objects_uid, ui_json_data, _, _ = block_model.update_object_options(
        None, None, param_name="objects", trigger=""
    )
    param_list = [
        "new_grid",
        "cell_size_x",
        "cell_size_y",
        "cell_size_z",
        "depth_core",
        "horizontal_padding",
        "bottom_padding",
        "expansion_fact",
        "monitoring_directory",
    ]
    (  # pylint: disable=W0632
        new_grid,
        cell_size_x,
        cell_size_y,
        cell_size_z,
        depth_core,
        horizontal_padding,
        bottom_padding,
        expansion_fact,
        _,
    ) = block_model.update_remainder_from_ui_json(
        ui_json_data, param_list, trigger="test"
    )

    assert new_grid == block_model.params.new_grid
    assert objects_uid == "{" + str(block_model.params.objects.uid) + "}"
    assert cell_size_x == block_model.params.cell_size_x
    assert cell_size_y == block_model.params.cell_size_y
    assert cell_size_z == block_model.params.cell_size_z
    assert depth_core == block_model.params.depth_core
    assert horizontal_padding == block_model.params.horizontal_padding
    assert bottom_padding == block_model.params.bottom_padding
    assert expansion_fact == block_model.params.expansion_fact

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
        new_grid=new_grid,
        objects=object_options[0]["value"],
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        cell_size_z=cell_size_z,
        depth_core=depth_core,
        horizontal_padding=horizontal_padding,
        bottom_padding=bottom_padding,
        expansion_fact=expansion_fact,
        live_link=[],
        monitoring_directory=str(tmp_path),
        trigger="export",
    )

    filename = next(tmp_path.glob("BlockModel_*.geoh5"))
    with Workspace(filename) as workspace:
        ent = workspace.get_entity("BlockModel")
        assert (len(ent) == 1) and (ent[0] is not None)
        assert np.sum([isinstance(c, FilenameData) for c in ent[0].children]) == 1


def test_calculator(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace(temp_workspace) as workspace:
        GEOH5.get_entity("geochem")[0].copy(parent=workspace)

    app = Calculator(geoh5=str(temp_workspace))
    app.trigger.click()

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        output = workspace.get_entity("NewChannel")[0]
        assert output.values.shape[0] == 4438, "Change in output. Need to verify."


def test_coordinate_transformation(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace.create(temp_workspace) as workspace:
        GEOH5.get_entity("Gravity_Magnetics_drape60m")[0].copy(parent=workspace)
        GEOH5.get_entity("Data_TEM_pseudo3D")[0].copy(parent=workspace)

    app = CoordinateTransformation(geoh5=str(temp_workspace))
    app.trigger.click()

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        assert len(workspace.objects) == 4, "Coordinate transform failed."


def test_contour_values(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace(temp_workspace) as workspace:
        GEOH5.get_entity(uuid.UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[0].copy(
            parent=workspace
        )

    app = ContourValues(geoh5=str(temp_workspace), plot_result=False)
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        output = workspace.get_entity("contours")[0]
        assert output.n_vertices == 2655, "Change in output. Need to verify."
        output = workspace.get_entity("Contours")[0]
        assert np.sum([isinstance(c, FilenameData) for c in output.children]) == 1


def test_create_surface(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
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

    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        group = workspace.get_entity("CDI")[0]
        assert len(group.children) == 1


def test_clustering(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in ["{79b719bc-d996-4f52-9af0-10aa9c7bb941}"]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    params = ClusteringParams(geoh5=str(temp_workspace), output_path=str(tmp_path))
    app = Clustering(params=params)

    # Set test variables
    n_clicks = 0
    live_link = []
    n_clusters = 3
    objects = "{79b719bc-d996-4f52-9af0-10aa9c7bb941}"
    data_subset = [
        "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}",
        "{18c2560c-6161-468a-8571-5d9d59649535}",
    ]
    color_pickers = ["#FF5733", "#33FF9D", "#E433FF"]
    downsampling = 80
    full_scales = {}
    full_lower_bounds = {}
    full_upper_bounds = {}
    x = "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}"
    x_log = [False]
    x_thresh = 0.1
    x_min = -17.0
    x_max = 25.5
    y = "{18c2560c-6161-468a-8571-5d9d59649535}"
    y_log = [True]
    y_thresh = 0.1
    y_min = -17.0
    y_max = 29.8
    z = None
    z_log = [True]
    z_thresh = 0.1
    z_min = 0
    z_max = 0
    color = None
    color_log = [True]
    color_thresh = 0.1
    color_min = 0
    color_max = 0
    size = None
    size_log = [True]
    size_thresh = 0.1
    size_min = 0
    size_max = 0
    size_markers = 20
    channel = "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}"
    channel_options = [
        {"label": "Al2O3", "value": "{cdd7668a-4b5b-49ac-9365-c9ce4fddf733}"},
        {"label": "CaO", "value": "{18c2560c-6161-468a-8571-5d9d59649535}"},
    ]
    ga_group_name = "Clusters"
    monitoring_directory = str(tmp_path)
    trigger = "export"
    clusters = {}

    with app.workspace.open():
        dataframe_dict, mapping, indices = app.update_dataframe(
            downsampling, data_subset
        )

        # Set a callback since callback_context is used by run_clustering
        context_value.set(
            AttributeDict(
                **{"triggered_inputs": [{"prop_id": "n_clusters.value", "value": 3}]}
            )
        )
        kmeans, clusters = app.run_clustering(
            dataframe_dict,
            n_clusters,
            full_scales,
            clusters,
            mapping,
        )
        color_maps = "kmeans"

        # Set a callback since callback_context is used by scatter plot
        context_value.set(
            AttributeDict(
                **{"triggered_inputs": [{"prop_id": "downsampling.value", "value": 80}]}
            )
        )

        # Test scatter plot output
        figure = app.make_scatter_plot(
            n_clusters,
            dataframe_dict,
            kmeans,
            indices,
            color_pickers,
            channel_options,
            x,
            x_log,
            x_thresh,
            x_min,
            x_max,
            y,
            y_log,
            y_thresh,
            y_min,
            y_max,
            z,
            z_log,
            z_thresh,
            z_min,
            z_max,
            color,
            color_log,
            color_thresh,
            color_min,
            color_max,
            color_maps,
            size,
            size_log,
            size_thresh,
            size_min,
            size_max,
            size_markers,
        )
        assert len(figure["data"]) != 0

        # Test inertia plot
        figure = app.make_inertia_plot(n_clusters, clusters)
        assert len(figure["data"]) != 0

        # Test histogram
        figure = app.make_hist_plot(
            dataframe_dict,
            channel,
            channel_options,
            lower_bounds=x_min,
            upper_bounds=x_max,
        )
        assert len(figure["data"]) != 0

        # Test boxplot
        figure = app.make_boxplot(
            n_clusters,
            channel,
            channel_options,
            color_pickers,
            kmeans,
            indices,
        )
        assert len(figure["data"]) != 0

        # Test stats table
        table = app.make_stats_table(dataframe_dict)
        assert table is not None

        # Test heatmap
        figure = app.make_heatmap(dataframe_dict)
        assert len(figure["data"]) != 0

        # Test export
        app.trigger_click(
            n_clicks,
            monitoring_directory,
            live_link,
            n_clusters,
            objects,
            data_subset,
            color_pickers,
            downsampling,
            full_scales,
            full_lower_bounds,
            full_upper_bounds,
            x,
            x_log,
            x_thresh,
            x_min,
            x_max,
            y,
            y_log,
            y_thresh,
            y_min,
            y_max,
            z,
            z_log,
            z_thresh,
            z_min,
            z_max,
            color,
            color_log,
            color_thresh,
            color_min,
            color_max,
            color_maps,
            size,
            size_log,
            size_thresh,
            size_min,
            size_max,
            size_markers,
            channel,
            ga_group_name,
            trigger,
        )

    filename = next(tmp_path.glob("Clustering_*.geoh5"))
    with Workspace(filename) as workspace:
        assert len(workspace.get_entity("Clusters")) == 1


def test_data_interpolation(tmp_path: Path):
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
            "{f3e36334-be0a-4210-b13e-06933279de25}",
            "{7450be38-1327-4336-a9e4-5cff587b6715}",
            "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
        ]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = DataInterpolation(geoh5=str(temp_workspace))
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
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

    with Workspace(get_output_workspace(tmp_path)) as workspace:
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
    temp_workspace = tmp_path / "contour.geoh5"
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
        ]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = IsoSurface(geoh5=str(temp_workspace))
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        group = workspace.get_entity("Isosurface")[0]
        assert len(group.children) == 5
        assert np.sum([isinstance(c, FilenameData) for c in group.children]) == 1
        assert np.sum([isinstance(c, Surface) for c in group.children]) == 4

    app.fixed_contours.value = "1000."

    with pytest.warns(UserWarning, match="The following levels were"):
        app.trigger_click(None)


def test_peak_finder():
    app = PeakFinder()
    app.line_field.value = uuid.UUID("{90b1d710-8a0f-4f69-bd38-6c06c7a977ed}")
    assert app.data.value == uuid.UUID("{b834a590-dea9-48cb-abe3-8c714bb0bb7c}")
