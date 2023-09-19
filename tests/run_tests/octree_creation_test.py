#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path

import numpy as np
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Curve, Points, Surface
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json.utils import str2list
from geoh5py.workspace import Workspace
from scipy import spatial

from geoapps.driver_base.utils import treemesh_2_octree
from geoapps.octree_creation.application import OctreeDriver, OctreeMesh
from geoapps.octree_creation.params import OctreeParams
from geoapps.utils.testing import get_output_workspace

# pytest.skip("eliminating conflicting test.", allow_module_level=True)


def test_create_octree_app(tmp_path: Path):
    # Create temp workspace
    with Workspace(tmp_path / "testOctree.geoh5") as workspace:
        n_data = 12
        xyz = np.random.randn(n_data, 3) * 100
        points = Points.create(workspace, vertices=xyz)
        remote = Points.create(
            workspace, vertices=np.array([[800, 800, np.mean(xyz[:, 2])]])
        )
        x, y = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))
        x, y = x.ravel() * 100, y.ravel() * 100
        z = np.random.randn(x.shape[0]) * 10
        surf = spatial.Delaunay(np.c_[x, y])
        simplices = getattr(surf, "simplices")
        # Create a geoh5 surface
        topo = Surface.create(workspace, vertices=np.c_[x, y, z], cells=simplices)
        h = [5.0, 10.0, 15.0]
        depth_core = 400.0
        horizontal_padding = 500.0
        vertical_padding = 200.0
        p_d = [
            [horizontal_padding, horizontal_padding],
            [horizontal_padding, horizontal_padding],
            [vertical_padding, vertical_padding],
        ]
        max_distance = 200
        refine_a = "4, 4, 4"
        refine_b = "0, 0, 4"

        # Create a tree mesh from discretize
        treemesh = mesh_builder_xyz(
            points.vertices,
            h,
            padding_distance=p_d,
            mesh_type="tree",
            depth_core=depth_core,
        )
        treemesh.refine(treemesh.max_level - 3, finalize=False)
        treemesh = refine_tree_xyz(
            treemesh,
            points.vertices,
            method="radial",
            octree_levels=str2list(refine_a),
            max_distance=max_distance,
            finalize=False,
        )
        treemesh = refine_tree_xyz(
            treemesh,
            topo.vertices,
            method="surface",
            octree_levels=str2list(refine_b),
            max_distance=max_distance,
            finalize=False,
        )
        treemesh = refine_tree_xyz(
            treemesh,
            remote.vertices,
            method="radial",
            octree_levels=str2list(refine_a),
            max_distance=max_distance,
            finalize=True,
        )

        octree = treemesh_2_octree(workspace, treemesh, name="Octree_Mesh")

        # Repeat the creation using the app
        refinements = {
            "Refinement A object": points.uid,
            "Refinement A levels": refine_a,
            "Refinement A type": "radial",
            "Refinement A distance": max_distance,
            "Refinement B object": topo.uid,
            "Refinement B levels": refine_b,
            "Refinement B type": "surface",
            "Refinement B distance": max_distance,
            "Refinement C object": remote.uid,
            "Refinement C levels": refine_a,
            "Refinement C type": "radial",
            "Refinement C distance": max_distance,
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
    with Workspace(tmp_path / "testOctree.geoh5") as workspace:
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
