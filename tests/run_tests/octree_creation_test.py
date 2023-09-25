#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path

import numpy as np
from discretize.utils import mesh_builder_xyz
from geoh5py.objects import Curve, Points
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json.utils import str2list
from geoh5py.workspace import Workspace

from geoapps.driver_base.utils import treemesh_2_octree
from geoapps.octree_creation.application import OctreeDriver, OctreeMesh
from geoapps.octree_creation.params import OctreeParams
from geoapps.utils.testing import get_output_workspace

# pytest.skip("eliminating conflicting test.", allow_module_level=True)


def setup_test_octree():
    """
    Create a circle of points and treemesh from extent.
    """
    h = [5.0, 5.0, 5.0]
    n_data = 16
    degree = np.linspace(0, 2 * np.pi, n_data)
    xyz = np.c_[
        np.cos(degree) * 200.0, np.sin(degree) * 200.0, np.sin(degree * 2.0) * 40.0
    ]
    # Add point at origin
    xyz = np.r_[xyz, np.zeros((1, 3))]
    depth_core = 400.0
    horizontal_padding = 500.0
    vertical_padding = 200.0
    p_d = [
        [horizontal_padding, horizontal_padding],
        [horizontal_padding, horizontal_padding],
        [vertical_padding, vertical_padding],
    ]
    # Create a tree mesh from discretize
    treemesh = mesh_builder_xyz(
        xyz,
        h,
        padding_distance=p_d,
        mesh_type="tree",
        depth_core=depth_core,
    )

    return xyz, treemesh, h, depth_core, horizontal_padding, vertical_padding


def test_create_octree_app_radial(tmp_path: Path):
    # Create temp workspace
    refine_a = "4, 4"
    minimum_level = 4
    (
        xyz,
        treemesh,
        h,
        depth_core,
        horizontal_padding,
        vertical_padding,
    ) = setup_test_octree()

    with Workspace.create(tmp_path / "testOctree.geoh5") as workspace:
        points = Points.create(workspace, vertices=xyz)
        treemesh.refine(treemesh.max_level - minimum_level + 1, finalize=False)
        treemesh = OctreeDriver.refine_tree_from_points(
            treemesh,
            points,
            str2list(refine_a),
            finalize=True,
        )
        octree = treemesh_2_octree(workspace, treemesh, name="Octree_Mesh")

        # Repeat the creation using the app
        refinements = {
            "Refinement A object": points.uid,
            "Refinement A levels": refine_a,
            "Refinement A type": "radial",
            "Refinement B object": None,
            "minimum_level": minimum_level,
            # "Refinement B levels": refine_b,
            # "Refinement B type": "surface",
            # "Refinement B distance": max_distance,
            # "Refinement C object": remote.uid,
            # "Refinement C levels": refine_a,
            # "Refinement C type": "radial",
            # "Refinement C distance": max_distance,
        }
        app = OctreeMesh(
            geoh5=workspace,
            objects=str(points.uid),
            u_cell_size=h[0],
            v_cell_size=h[1],
            w_cell_size=h[2],
            horizontal_padding=horizontal_padding,
            vertical_padding=vertical_padding,
            depth_core=depth_core,
            **refinements,
        )
        app.trigger_click(None)

        # Re-load the new mesh and compare
        with Workspace(get_output_workspace(tmp_path)) as workspace:
            rec_octree = workspace.get_entity("Octree_Mesh")[0]
            compare_entities(octree, rec_octree, ignore=["_uid"])


def test_create_octree_driver(tmp_path: Path):
    uijson_path = tmp_path.parent / "test_create_octree_app0" / "Temp"
    json_file = next(uijson_path.glob("*.ui.json"))
    driver = OctreeDriver.start(str(json_file))

    with driver.params.geoh5.open(mode="r"):
        results = driver.params.geoh5.get_entity("Octree_Mesh")
        compare_entities(results[0], results[1], ignore=["_uid"])


def test_create_octree_curve(tmp_path: Path):
    # Create temp workspace
    with Workspace.create(tmp_path / "testOctree.geoh5") as workspace:
        n_data = 12
        xyz = np.random.randn(n_data, 3) * 100
        points = Curve.create(workspace, vertices=xyz)
        refine_a = "4, 4, 4"
        h = [5.0, 5.0, 5.0]
        depth_core = 400.0
        horizontal_padding = 500.0
        vertical_padding = 200.0
        minimum_level = 4
        params = OctreeParams(
            geoh5=workspace,
            objects=str(points.uid),
            u_cell_size=h[0],
            v_cell_size=h[1],
            w_cell_size=h[2],
            horizontal_padding=horizontal_padding,
            vertical_padding=vertical_padding,
            depth_core=depth_core,
            minimum_level=minimum_level,
        )  # pylint: disable=W0212

        setattr(params, "Refinement A object", points)
        setattr(params, "Refinement A levels", refine_a)
        setattr(params, "Refinement A type", "radial")
        setattr(params, "Refinement A distance", 0.0)
        params.write_input_file("octree Mesh Creator.ui.json", tmp_path)
        # print("Initializing application . . .")
        driver = OctreeDriver(params)
        print("Running application . . .")
        octree = driver.run()

        assert octree.octree_cells["NCells"].max() == 2 ** (minimum_level - 1)
