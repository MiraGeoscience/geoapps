# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import sys

import numpy as np
from discretize import TreeMesh
from discretize.utils import mesh_builder_xyz
from geoh5py.objects import Curve, ObjectBase, Octree, Surface
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import utils
from scipy import interpolate
from scipy.spatial import Delaunay, cKDTree

from geoapps.driver_base.driver import BaseDriver
from geoapps.driver_base.utils import treemesh_2_octree
from geoapps.octree_creation.constants import validations
from geoapps.octree_creation.params import OctreeParams
from geoapps.shared_utils.utils import densify_curve, get_locations


class OctreeDriver(BaseDriver):
    _params_class = OctreeParams
    _validations = validations

    def __init__(self, params: OctreeParams):
        super().__init__(params)
        self.params: OctreeParams = params

    def run(self) -> Octree:
        """
        Create an octree mesh from input values
        """
        with fetch_active_workspace(self.params.geoh5, mode="r+"):
            octree = self.octree_from_params(self.params)
            self.update_monitoring_directory(octree)

        return octree

    @staticmethod
    def minimum_level(mesh: TreeMesh, level: int):
        """Computes the minimum level of refinement for a given tree mesh."""
        return max([1, mesh.max_level - level + 1])

    @staticmethod
    def octree_from_params(params: OctreeParams):
        print("Setting the mesh extent")
        entity = params.objects
        mesh: TreeMesh = mesh_builder_xyz(
            entity.vertices,
            [
                params.u_cell_size,
                params.v_cell_size,
                params.w_cell_size,
            ],
            padding_distance=params.get_padding(),
            mesh_type="tree",
            depth_core=params.depth_core,
        )
        minimum_level = OctreeDriver.minimum_level(mesh, params.minimum_level)
        mesh.refine(
            minimum_level, finalize=False, diagonal_balance=params.diagonal_balance
        )

        for label, value in params.free_parameter_dict.items():
            refinement_object = getattr(params, value["object"])
            levels = utils.str2list(getattr(params, value["levels"]))
            if not isinstance(refinement_object, ObjectBase):
                continue

            print(f"Applying {label} on: {getattr(params, value['object']).name}")

            if isinstance(refinement_object, Curve):
                mesh = OctreeDriver.refine_tree_from_curve(
                    mesh, refinement_object, levels, params.diagonal_balance
                )

            elif isinstance(refinement_object, Surface):
                mesh = OctreeDriver.refine_tree_from_triangulation(
                    mesh, refinement_object, levels, params.diagonal_balance
                )

            elif getattr(params, value["type"]) == "surface":
                mesh = OctreeDriver.refine_tree_from_surface(
                    mesh,
                    refinement_object,
                    levels,
                    params.diagonal_balance,
                    max_distance=getattr(params, value["distance"]),
                )

            elif getattr(params, value["type"]) == "radial":
                mesh = OctreeDriver.refine_tree_from_points(
                    mesh,
                    refinement_object,
                    levels,
                    diagonal_balance=params.diagonal_balance,
                )

            else:
                raise NotImplementedError(
                    f"Refinement type {value['type']} is not implemented."
                )

        print("Finalizing . . .")
        mesh.finalize()
        octree = treemesh_2_octree(params.geoh5, mesh, name=params.ga_group_name)

        return octree

    @staticmethod
    def refine_tree_from_curve(
        mesh: TreeMesh,
        curve: Curve,
        levels: list[int] | np.ndarray,
        diagonal_balance: bool = True,
        finalize: bool = False,
    ) -> TreeMesh:
        """
        Refine a tree mesh along the segments of a curve densified by the
        mesh cell size.

        :param mesh: Tree mesh to refine.
        :param curve: Curve object to use for refinement.
        :param levels: Number of cells requested at each refinement level.
            Defined in reversed order from the highest octree to lowest.
        :param diagonal_balance: Whether to balance cells along the diagonal of the tree during construction.
        :param finalize: Finalize the tree mesh after refinement.

        """
        if not isinstance(curve, Curve):
            raise TypeError("Refinement object must be a Curve.")

        if isinstance(levels, list):
            levels = np.array(levels)

        locations = densify_curve(curve, mesh.h[0][0])
        mesh = OctreeDriver.refine_tree_from_points(
            mesh, locations, levels, diagonal_balance=diagonal_balance, finalize=False
        )

        if finalize:
            mesh.finalize()

        return mesh

    @staticmethod
    def refine_tree_from_points(
        mesh: TreeMesh,
        points: ObjectBase | np.ndarray,
        levels: list[int] | np.ndarray,
        diagonal_balance: bool = True,
        finalize: bool = False,
    ) -> TreeMesh:
        """
        Refine a tree mesh along the vertices of an object.

        :param mesh: Tree mesh to refine.
        :param points: Object to use for refinement.
        :param levels: Number of cells requested at each refinement level.
            Defined in reversed order from the highest octree to lowest.
        :param diagonal_balance: Whether to balance cells along the diagonal of the tree during construction.
        :param finalize: Finalize the tree mesh after refinement.

        :return: Refined tree mesh.
        """
        if isinstance(points, ObjectBase):
            locations = get_locations(points.workspace, points)
        else:
            locations = points

        if locations is None:
            raise ValueError("Could not find locations for refinement.")

        if isinstance(levels, list):
            levels = np.array(levels)

        distance = 0
        for ii, n_cells in enumerate(levels):
            distance += n_cells * OctreeDriver.cell_size_from_level(mesh, ii)
            mesh.refine_ball(
                locations,
                distance,
                mesh.max_level - ii,
                diagonal_balance=diagonal_balance,
                finalize=False,
            )

        if finalize:
            mesh.finalize()

        return mesh

    @staticmethod
    def refine_tree_from_surface(
        mesh: TreeMesh,
        surface: ObjectBase,
        levels: list[int] | np.ndarray,
        diagonal_balance: bool = True,
        max_distance: float = np.inf,
        finalize: bool = False,
    ) -> TreeMesh:
        """
        Refine a tree mesh along the simplicies of a surface.

        :param mesh: Tree mesh to refine.
        :param surface: Surface object to use for refinement.
        :param levels: Number of cells requested at each refinement level.
            Defined in reversed order from the highest octree to lowest.
        :param max_distance: Maximum distance from the surface to refine.
        :param diagonal_balance: Whether to balance cells along the diagonal of the tree during construction.
        :param finalize: Finalize the tree mesh after refinement.

        :return: Refined tree mesh.
        """
        if isinstance(levels, list):
            levels = np.array(levels)

        xyz = get_locations(surface.workspace, surface)
        tri2D = Delaunay(xyz[:, :2])
        tree = cKDTree(xyz[:, :2])

        if isinstance(surface, Surface):
            tri2D.simplices = surface.cells

        F = interpolate.LinearNDInterpolator(tri2D, xyz[:, -1])
        levels = np.array(levels)

        depth = 0
        # Cycle through the Tree levels backward
        for ind, n_cells in enumerate(levels):
            if n_cells == 0:
                continue

            dx = OctreeDriver.cell_size_from_level(mesh, ind, 0)
            dy = OctreeDriver.cell_size_from_level(mesh, ind, 1)
            dz = OctreeDriver.cell_size_from_level(mesh, ind, 2)

            # Create a grid at the octree level in xy
            cell_center_x, cell_center_y = np.meshgrid(
                np.arange(surface.extent[0, 0], surface.extent[1, 0], dx),
                np.arange(surface.extent[0, 1], surface.extent[1, 1], dy),
            )
            xy = np.c_[cell_center_x.reshape(-1), cell_center_y.reshape(-1)]

            # Only keep points within triangulation
            inside = tri2D.find_simplex(xy) != -1
            r, _ = tree.query(xy)
            keeper = np.logical_and(r < max_distance, inside)
            nnz = keeper.sum()
            elevation = F(xy[keeper])

            # Apply vertical padding for current octree level
            for _ in range(int(n_cells)):
                depth += dz
                mesh.insert_cells(
                    np.c_[xy[keeper], elevation - depth],
                    np.ones(nnz) * mesh.max_level - ind,
                    diagonal_balance=diagonal_balance,
                    finalize=False,
                )

        if finalize:
            mesh.finalize()

        return mesh

    @staticmethod
    def refine_tree_from_triangulation(
        mesh: TreeMesh,
        surface,
        levels: list[int] | np.ndarray,
        diagonal_balance: bool = True,
        finalize=False,
    ) -> TreeMesh:
        """
        Refine a tree mesh along the simplicies of a surface.

        :param mesh: Tree mesh to refine.
        :param surface: Surface object to use for refinement.
        :param levels: Number of cells requested at each refinement level.
            Defined in reversed order from highest octree to lowest.
        :param diagonal_balance: Whether to balance cells along the diagonal of the tree during construction.
        :param finalize: Finalize the tree mesh after refinement.

        :return: Refined tree mesh.
        """
        if not isinstance(surface, Surface):
            raise TypeError("Refinement object must be a Surface.")

        if isinstance(levels, list):
            levels = np.array(levels)

        ind = np.where(np.r_[levels] > 0)[0]

        if any(ind):
            paddings = []
            for n_cells in levels[ind[0] :]:
                if n_cells == 0:
                    continue

                paddings.append([n_cells] * 3)

            mesh.refine_surface(
                (surface.vertices, surface.cells),
                -ind[0] - 1,
                paddings,
                diagonal_balance=diagonal_balance,
                finalize=finalize,
            )
        return mesh

    @staticmethod
    def cell_size_from_level(octree, level: int, axis: int = 0):
        """
        Computes the cell size at a given level of refinement for a given tree mesh.

        :param octree: Tree mesh to refine.
        :param level: Level of refinement.
        :param axis: Axis of refinement.

        :return: Cell size at the given level of refinement.
        """
        return octree.h[axis][0] * 2**level


if __name__ == "__main__":
    file = sys.argv[1]
    OctreeDriver.start(file)
