#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
import sys
from os import path

import numpy as np
from discretize.utils import mesh_utils
from geoh5py.objects import BlockModel
from geoh5py.ui_json import InputFile, monitored_directory_copy
from geoh5py.workspace import Workspace
from scipy.spatial import cKDTree

from geoapps.block_model_creation.params import BlockModelParams
from geoapps.shared_utils.utils import get_locations


class BlockModelDriver:
    """
    Create BlockModel from BlockModelParams.
    """

    def __init__(self, params: BlockModelParams):
        self.params: BlockModelParams = params

    @staticmethod
    def truncate_locs_depths(locs: np.ndarray, depth_core: float) -> np.ndarray:
        """
        Sets locations below core to core bottom.

        :param locs: Location points.
        :param depth_core: Depth of core mesh below locs.

        :return locs: locs with depths truncated.
        """
        zmax = locs[:, 2].max()  # top of locs
        below_core_ind = (zmax - locs[:, 2]) > depth_core
        core_bottom_elev = zmax - depth_core
        locs[
            below_core_ind, 2
        ] = core_bottom_elev  # sets locations below core to core bottom
        return locs

    @staticmethod
    def minimum_depth_core(
        locs: np.ndarray, depth_core: float, core_z_cell_size: int
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

    @staticmethod
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

    @staticmethod
    def get_block_model(
        workspace: Workspace,
        name: str,
        locs: np.ndarray,
        h: list,
        depth_core: float,
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

        locs = BlockModelDriver.truncate_locs_depths(locs, depth_core)
        depth_core = BlockModelDriver.minimum_depth_core(locs, depth_core, h[2])
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
            u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
            v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
            z_cell_delimiters=-(mesh.x0[2] + mesh.hz.sum() - mesh.vectorNz[::-1]),
            name=name,
        )

        top_padding = BlockModelDriver.find_top_padding(object_out, h[2])
        object_out.origin["z"] += top_padding

        return object_out

    def run(self):
        """
        Create block model and add to self.params.geoh5.
        """
        xyz = get_locations(self.params.geoh5, self.params.objects)
        if xyz is None:
            raise ValueError("Input object has no centroids or vertices.")

        tree = cKDTree(xyz)

        # Find extent of grid
        h = [
            self.params.cell_size_x,
            self.params.cell_size_y,
            self.params.cell_size_z,
        ]
        # pads: W, E, S, N, D, U
        pads = [
            self.params.horizontal_padding,
            self.params.horizontal_padding,
            self.params.horizontal_padding,
            self.params.horizontal_padding,
            self.params.bottom_padding,
            0.0,
        ]

        object_out = BlockModelDriver.get_block_model(
            self.params.geoh5,
            self.params.new_grid,
            xyz,
            h,
            self.params.depth_core,
            pads,
            self.params.expansion_fact,
        )

        # Try to recenter on nearest
        # Find nearest cells
        rad, ind = tree.query(object_out.centroids)
        ind_nn = np.argmin(rad)

        d_xyz = object_out.centroids[ind_nn, :] - xyz[ind[ind_nn], :]

        object_out.origin = np.r_[object_out.origin.tolist()] - d_xyz

        if self.params.monitoring_directory is not None and path.exists(
            os.path.abspath(self.params.monitoring_directory)
        ):
            monitored_directory_copy(
                os.path.abspath(self.params.monitoring_directory), object_out
            )


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params_class = BlockModelParams(ifile)

    driver = BlockModelDriver(params_class)
    print("Loaded. Creating block model . . .")
    with params_class.geoh5.open("r+"):
        driver.run()
    print("Saved to " + params_class.geoh5.h5file)
