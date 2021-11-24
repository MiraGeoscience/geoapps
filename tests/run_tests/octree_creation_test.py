#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from os import path

import numpy as np
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from geoh5py.objects import Points, Surface
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace
from scipy import spatial

from geoapps.create.octree_mesh import OctreeMesh
from geoapps.utils.utils import treemesh_2_octree


def test_create_octree_app(tmp_path):
    project = path.join(tmp_path, "testOctree.geoh5")
    # Create temp workspace
    ws = Workspace(project)
    n_data = 12
    xyz = np.random.randn(n_data, 3) * 100
    points = Points.create(ws, vertices=xyz)
    x, y = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))
    x, y = x.ravel() * 100, y.ravel() * 100
    z = np.random.randn(x.shape[0]) * 10
    surf = spatial.Delaunay(np.c_[x, y])
    simplices = getattr(surf, "simplices")
    # Create a geoh5 surface
    topo = Surface.create(ws, vertices=np.c_[x, y, z], cells=simplices)
    h = [5.0, 10.0, 15.0]
    depth_core = 400
    horizontal_padding = 500
    vertical_padding = 200
    p_d = [
        [horizontal_padding, horizontal_padding],
        [horizontal_padding, horizontal_padding],
        [vertical_padding, vertical_padding],
    ]
    max_distance = 200
    refine_A = [4, 4, 4]
    refine_B = [0, 0, 4]
    # Create a tree mesh from discretize
    treemesh = mesh_builder_xyz(
        points.vertices,
        h,
        padding_distance=p_d,
        mesh_type="tree",
        depth_core=depth_core,
    )
    treemesh = refine_tree_xyz(
        treemesh,
        points.vertices,
        method="radial",
        octree_levels=refine_A,
        max_distance=max_distance,
        finalize=False,
    )
    treemesh = refine_tree_xyz(
        treemesh,
        topo.vertices,
        method="surface",
        octree_levels=refine_B,
        max_distance=max_distance,
        finalize=True,
    )

    octree = treemesh_2_octree(ws, treemesh, name="Octree_Mesh")

    # Repeat the creation using the app
    refinements = {
        "Refinement A object": points.uid,
        "Refinement A levels": refine_A,
        "Refinement A type": "radial",
        "Refinement A distance": max_distance,
        "Refinement B object": topo.uid,
        "Refinement B levels": refine_B,
        "Refinement B type": "surface",
        "Refinement B distance": max_distance,
    }
    app = OctreeMesh(
        geoh5=str(ws.h5file),
        objects=str(points.uid),
        u_cell_size=h[0],
        v_cell_size=h[1],
        w_cell_size=h[2],
        horizontal_padding=horizontal_padding,
        vertical_padding=vertical_padding,
        depth_core=depth_core,
        **refinements,
    )
    app.trigger.click()
    # Re-load the new mesh and compare
    ws_B = Workspace(app.h5file)
    rec_octree = ws_B.get_entity("Octree_Mesh")[0]
    compare_entities(octree, rec_octree, ignore=["_name", "_uid"])
