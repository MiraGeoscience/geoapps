#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import base64
import uuid
from os import listdir, path

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace

from geoapps.block_model_creation.application import BlockModelCreation
from geoapps.calculator import Calculator
from geoapps.clustering.application import Clustering
from geoapps.contours.application import ContourValues
from geoapps.coordinate_transformation import CoordinateTransformation
from geoapps.edge_detection.application import EdgeDetectionApp
from geoapps.export.application import Export
from geoapps.interpolation.application import DataInterpolation
from geoapps.iso_surfaces.application import IsoSurface
from geoapps.triangulated_surfaces.application import Surface2D
from geoapps.utils.testing import get_output_workspace

# import pytest
# pytest.skip("eliminating conflicting test.", allow_module_level=True)

PROJECT = "./FlinFlon.geoh5"
GEOH5 = Workspace(PROJECT)


def test_block_model(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in ["{2e814779-c35f-4da0-ad6a-39a6912361f9}"]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    block_model = BlockModelCreation(geoh5=temp_workspace)
    # Test initialization
    object_options, objects_uid, ui_json_data, _, _ = block_model.update_object_options(
        None, None, trigger=""
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
    temp_workspace2 = path.join(tmp_path, "contour2.geoh5")
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
        "ws.geoh5", contents, trigger="upload"
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

    filename = list(
        filter(lambda x: ("BlockModel_" in x) and ("geoh5" in x), listdir(tmp_path))
    )[0]
    with Workspace(path.join(tmp_path, filename)) as workspace:
        ent = workspace.get_entity("BlockModel")
        assert (len(ent) == 1) and (ent[0] is not None)


def test_calculator(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        GEOH5.get_entity("geochem")[0].copy(parent=workspace)

    app = Calculator(geoh5=temp_workspace)
    app.trigger.click()

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        output = workspace.get_entity("NewChannel")[0]
        assert output.values.shape[0] == 4438, "Change in output. Need to verify."


def test_coordinate_transformation(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        GEOH5.get_entity("Gravity_Magnetics_drape60m")[0].copy(parent=workspace)
        GEOH5.get_entity("Data_TEM_pseudo3D")[0].copy(parent=workspace)

    app = CoordinateTransformation(geoh5=temp_workspace)
    app.trigger.click()

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        assert len(workspace.objects) == 2, "Coordinate transform failed."


def test_contour_values(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        GEOH5.get_entity(uuid.UUID("{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}"))[0].copy(
            parent=workspace
        )

    app = ContourValues(geoh5=temp_workspace, plot_result=False)
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        output = workspace.get_entity("contours")[0]
        assert output.n_vertices == 2655, "Change in output. Need to verify."


def test_create_surface(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{5fa66412-3a4c-440c-8b87-6f10cb5f1c7f}",
        ]:
            new_obj = GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = Surface2D(geoh5=temp_workspace)

    app.data.value = [p_g.uid for p_g in new_obj.property_groups if p_g.name == "COND"]
    app.elevations.data.value = [
        p_g.uid for p_g in new_obj.property_groups if p_g.name == "ELEV"
    ][0]

    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        group = workspace.get_entity("CDI")[0]
        assert len(group.children) == 1


def test_clustering(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in ["{79b719bc-d996-4f52-9af0-10aa9c7bb941}"]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)
    app = Clustering(geoh5=temp_workspace, output_path=str(tmp_path))

    app.trigger_click(
        n_clicks=0,
        live_link=[],
        n_clusters=3,
        objects="{79b719bc-d996-4f52-9af0-10aa9c7bb941}",
        data_subset=[
            "{0e4833e3-74ad-4ca9-a98b-d8119069bc01}",
            "{18c2560c-6161-468a-8571-5d9d59649535}",
        ],
        color_pickers=[],
        downsampling=80,
        full_scales={},
        full_lower_bounds={},
        full_upper_bounds={},
        x="{0e4833e3-74ad-4ca9-a98b-d8119069bc01}",
        x_log=[True],
        x_thresh=0.1,
        x_min=0,
        x_max=0,
        y="{18c2560c-6161-468a-8571-5d9d59649535}",
        y_log=[True],
        y_thresh=0.1,
        y_min=0,
        y_max=0,
        z=None,
        z_log=[True],
        z_thresh=0.1,
        z_min=0,
        z_max=0,
        color=None,
        color_log=[True],
        color_thresh=0.1,
        color_min=0,
        color_max=0,
        color_maps=None,
        size=None,
        size_log=[True],
        size_thresh=0.1,
        size_min=0,
        size_max=0,
        size_markers=20,
        channel=None,
        ga_group_name="Clusters",
        monitoring_directory=str(tmp_path),
        trigger="export",
    )

    filename = list(
        filter(lambda x: ("Clustering_" in x) and ("geoh5" in x), listdir(tmp_path))
    )[0]
    with Workspace(path.join(tmp_path, filename)) as workspace:
        assert len(workspace.get_entity("Clusters")) == 1


def test_data_interpolation(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
            "{f3e36334-be0a-4210-b13e-06933279de25}",
            "{7450be38-1327-4336-a9e4-5cff587b6715}",
            "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}",
        ]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = DataInterpolation(geoh5=temp_workspace)
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        assert len(workspace.get_entity("Iteration_7_model_Interp")) == 1


def test_edge_detection(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}",
        ]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = EdgeDetectionApp(geoh5=temp_workspace, plot_result=False)

    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        assert (
            len(
                [
                    child
                    for child in workspace.get_entity("Airborne_Gxx")
                    if isinstance(child, Curve)
                ]
            )
            == 1
        )


def test_export():
    app = Export(geoh5=PROJECT)
    app.trigger.click()
    # TODO write all the files types and check that appropriate files are written


def test_iso_surface(tmp_path):
    temp_workspace = path.join(tmp_path, "contour.geoh5")
    with Workspace(temp_workspace) as workspace:
        for uid in [
            "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
        ]:
            GEOH5.get_entity(uuid.UUID(uid))[0].copy(parent=workspace)

    app = IsoSurface(geoh5=temp_workspace)
    app.trigger_click(None)

    with Workspace(get_output_workspace(tmp_path)) as workspace:
        group = workspace.get_entity("Isosurface")[0]
        assert len(group.children) == 5
