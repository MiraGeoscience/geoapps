# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from warnings import warn

import numpy as np
from discretize import TreeMesh
from discretize.utils import mesh_utils
from geoh5py.objects import BlockModel, Octree
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.workspace import Workspace
from scipy.spatial import cKDTree


def truncate_locs_depths(locs: np.ndarray, depth_core: int) -> np.ndarray:
    """
    Sets locations below core to core bottom.

    :param locs: Location points.
    :param depth_core: Depth of core mesh below locs.
    :return locs: locs with depths truncated.
    """
    zmax = locs[:, 2].max()  # top of locs
    below_core_ind = (zmax - locs[:, 2]) > depth_core
    core_bottom_elev = zmax - depth_core
    locs[below_core_ind, 2] = (
        core_bottom_elev  # sets locations below core to core bottom
    )
    return locs


def minimum_depth_core(
    locs: np.ndarray, depth_core: int, core_z_cell_size: int
) -> float:
    """
    Get minimum depth core.

    :param locs: Location points.
    :param depth_core: Depth of core mesh below locs.
    :param core_z_cell_size: Cell size in z direction.
    :return depth_core: Minimum depth core.
    """
    zrange = locs[:, 2].max() - locs[:, 2].min()  # locs z range
    if depth_core >= zrange:
        return depth_core - zrange + core_z_cell_size
    else:
        return depth_core


def find_top_padding(obj: BlockModel, core_z_cell_size: int) -> float:
    """
    Loop through cell spacing and sum until core_z_cell_size is reached.

    :param obj: Block model.
    :param core_z_cell_size: Cell size in z direction.
    :return pad_sum: Top padding.
    """
    pad_sum = 0.0
    for h in np.abs(np.diff(obj.z_cell_delimiters)):
        if h != core_z_cell_size:
            pad_sum += h
        else:
            return pad_sum


def get_block_model(
    workspace: Workspace,
    name: str,
    locs: np.ndarray,
    h: list,
    depth_core: int,
    pads: list,
    expansion_factor: float,
) -> BlockModel:
    """
    Create a BlockModel object from parameters.

    :param workspace: Workspace.
    :param name: Block model name.
    :param locs: Location points.
    :param h: Cell size(s) for the core mesh.
    :param depth_core: Depth of core mesh below locs.
    :param pads: len(6) Padding distances [W, E, N, S, Down, Up]
    :param expansion_factor: Expansion factor for padding cells.
    :return object_out: Output block model.
    """

    locs = truncate_locs_depths(locs, depth_core)
    depth_core = minimum_depth_core(locs, depth_core, h[2])
    mesh = mesh_utils.mesh_builder_xyz(
        locs,
        h,
        padding_distance=[
            [pads[0], pads[1]],
            [pads[2], pads[3]],
            [pads[4], pads[5]],
        ],
        depth_core=depth_core,
        expansion_factor=expansion_factor,
    )

    object_out = BlockModel.create(
        workspace,
        origin=[mesh.x0[0], mesh.x0[1], locs[:, 2].max()],
        u_cell_delimiters=mesh.nodes_x - mesh.x0[0],
        v_cell_delimiters=mesh.nodes_y - mesh.x0[1],
        z_cell_delimiters=-(mesh.x0[2] + mesh.h[2].sum() - mesh.nodes_z[::-1]),
        name=name,
    )

    top_padding = find_top_padding(object_out, h[2])
    object_out.origin["z"] += top_padding

    return object_out


def create_octree_from_octrees(meshes: list[Octree | TreeMesh]) -> TreeMesh:
    """
    Create an all encompassing octree mesh from a list of meshes.

    :param meshes: List of Octree or TreeMesh meshes.

    :return octree: A global Octree.
    """
    cell_size = []
    dimensions = None
    origin = None

    for mesh in meshes:
        attributes = get_octree_attributes(mesh)

        if dimensions is None:
            dimensions = attributes["dimensions"]
            origin = attributes["origin"]
        else:
            if not np.allclose(dimensions, attributes["dimensions"]):
                raise ValueError("Meshes must have same dimensions")

            if not np.allclose(origin, attributes["origin"]):
                raise ValueError("Meshes must have same origin")

        cell_size.append(attributes["cell_size"])

    cell_size = np.min(np.vstack(cell_size), axis=0)
    cells = []
    for ind in range(3):
        extent = dimensions[ind]
        maxLevel = int(np.ceil(np.log2(extent / cell_size[ind])))
        cells += [np.ones(2**maxLevel) * cell_size[ind]]

    # Define the mesh and origin
    treemesh = TreeMesh(cells, origin=origin)

    for mesh in meshes:
        if isinstance(mesh, Octree):
            centers = mesh.centroids
            levels = treemesh.max_level - np.log2(mesh.octree_cells["NCells"])
        else:
            centers = mesh.cell_centers
            levels = (
                treemesh.max_level
                - mesh.max_level
                + mesh.cell_levels_by_index(np.arange(mesh.nC))
            )

        treemesh.insert_cells(centers, levels, finalize=False)

    treemesh.finalize()

    return treemesh


def collocate_octrees(global_mesh: Octree, local_meshes: list[Octree]):
    """
    Collocate a list of octree meshes into a global octree mesh.

    :param global_mesh: Global octree mesh.
    :param local_meshes: List of local octree meshes.
    """
    attributes = get_octree_attributes(global_mesh)
    cell_size = attributes["cell_size"]

    u_grid = global_mesh.octree_cells["I"] * global_mesh.u_cell_size
    v_grid = global_mesh.octree_cells["J"] * global_mesh.v_cell_size
    w_grid = global_mesh.octree_cells["K"] * global_mesh.w_cell_size

    xyz = np.c_[u_grid, v_grid, w_grid] + attributes["origin"]
    tree = cKDTree(xyz)

    for local_mesh in local_meshes:
        attributes = get_octree_attributes(local_mesh)

        if cell_size and not cell_size == attributes["cell_size"]:
            raise ValueError(
                f"Cell size mismatch in dimension {cell_size} != {attributes['cell_size']}"
            )

        _, closest = tree.query(attributes["origin"])
        shift = xyz[closest, :] - attributes["origin"]

        if np.any(shift != 0.0):
            with fetch_active_workspace(local_mesh.workspace) as workspace:
                warn(
                    f"Shifting {local_mesh.name} mesh origin by {shift} m to match inversion mesh."
                )
                local_mesh.origin = attributes["origin"] + shift
                workspace.update_attribute(local_mesh, "attributes")


def get_octree_attributes(mesh: Octree | TreeMesh) -> dict[str, list]:
    """
    Get mesh attributes.

    :param mesh: Input Octree or TreeMesh object.
    :return mesh_attributes: Dictionary of mesh attributes.
    """
    if not isinstance(mesh, (Octree, TreeMesh)):
        raise TypeError(f"All meshes must be Octree or TreeMesh, not {type(mesh)}")

    cell_size = []
    cell_count = []
    dimensions = []
    if isinstance(mesh, TreeMesh):
        for dim in range(3):
            cell_size.append(mesh.h[dim][0])
            cell_count.append(mesh.h[dim].size)
            dimensions.append(mesh.h[dim].sum())
        origin = mesh.origin
    else:
        with fetch_active_workspace(mesh.workspace):
            for dim in "uvw":
                cell_size.append(np.abs(getattr(mesh, f"{dim}_cell_size")))
                cell_count.append(getattr(mesh, f"{dim}_count"))
                dimensions.append(
                    getattr(mesh, f"{dim}_cell_size") * getattr(mesh, f"{dim}_count")
                )
            origin = np.r_[mesh.origin["x"], mesh.origin["y"], mesh.origin["z"]]

    extent = np.r_[origin, origin + np.r_[dimensions]]

    return {
        "cell_count": cell_count,
        "cell_size": cell_size,
        "dimensions": dimensions,
        "extent": extent,
        "origin": origin,
    }
