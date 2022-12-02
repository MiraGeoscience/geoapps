#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discretize import TreeMesh

import numpy as np
from discretize.utils import mesh_utils
from geoh5py.groups import Group
from geoh5py.objects import BlockModel, DrapeModel
from geoh5py.workspace import Workspace
from scipy.interpolate import interp1d

from geoapps.block_model_creation.driver import BlockModelDriver
from geoapps.shared_utils.utils import rotate_xyz
from geoapps.utils.surveys import compute_alongline_distance


def face_average(mesh: TreeMesh, model: np.ndarray) -> np.ndarray:
    """
    Compute the average face values of a model

    :param mesh: Tree mesh object
    :param model: A vector of cell centered property values
    """
    return mesh.stencil_cell_gradient.T * (mesh.stencil_cell_gradient * model)


def floating_active(mesh: TreeMesh, active: np.ndarray):
    """
    True if there are any active cells in the air

    :param mesh: Tree mesh object
    :param active: active cells array
    """
    return True if any(face_average(mesh, active) >= 6) else False


def get_drape_model(
    workspace: Workspace,
    name: str,
    locations: np.ndarray,
    h: list,
    depth_core: float,
    pads: list,
    expansion_factor: float,
    parent: Group = None,
    return_colocated_mesh: bool = False,
    return_sorting: bool = False,
) -> tuple:
    """
    Create a BlockModel object from parameters.

    :param workspace: Workspace.
    :param parent: Group to contain the result.
    :param name: Block model name.
    :param locations: Location points.
    :param h: Cell size(s) for the core mesh.
    :param depth_core: Depth of core mesh below locs.
    :param pads: len(6) Padding distances [W, E, N, S, Down, Up]
    :param expansion_factor: Expansion factor for padding cells.
    :param return_colocated_mesh: If true return TensorMesh.
    :param return_sorting: If true, return the indices required to map
        values stored in the TensorMesh to the drape model.

    :return object_out: Output block model.
    """

    locations = BlockModelDriver.truncate_locs_depths(locations, depth_core)
    depth_core = BlockModelDriver.minimum_depth_core(locations, depth_core, h[1])
    locs = compute_alongline_distance(locations)
    x_interp = interp1d(locs[:, 0], locations[:, 0], fill_value="extrapolate")
    y_interp = interp1d(locs[:, 0], locations[:, 1], fill_value="extrapolate")

    mesh = mesh_utils.mesh_builder_xyz(
        locs,
        h,
        padding_distance=[
            [pads[0], pads[1]],
            [pads[2], pads[3]],
        ],
        depth_core=depth_core,
        expansion_factor=expansion_factor,
        mesh_type="tensor",
    )

    cc = mesh.cell_centers
    hz = mesh.h[1]
    top = np.max(cc[:, 1].reshape(len(hz), -1)[:, 0] + (hz / 2))
    bottoms = cc[:, 1].reshape(len(hz), -1)[:, 0] - (hz / 2)
    n_layers = len(bottoms)

    prisms = []
    layers = []
    indices = []
    index = 0
    for i, d in enumerate(np.unique(mesh.cell_centers[:, 0])):
        prisms.append(
            [float(x_interp(d)), float(y_interp(d)), top, i * n_layers, n_layers]
        )
        for k, b in enumerate(bottoms):
            layers.append([i, k, b])
            indices.append(index)
            index += 1

    model = DrapeModel.create(workspace, name=name, parent=parent)
    model.prisms = np.vstack(prisms)
    layers = np.vstack(layers)
    layers[:, 2] = layers[:, 2][::-1]
    model.layers = layers

    model.add_data(
        {
            "indices": {
                "values": np.array(indices, dtype=np.int32),
                "association": "CELL",
            }
        }
    )
    val = [model]
    if return_colocated_mesh:
        val.append(mesh)
    if return_sorting:
        sorting = np.arange(mesh.n_cells)
        sorting = sorting.reshape(mesh.nCy, mesh.nCx, order="C")
        sorting = sorting[::-1].T.flatten()
        val.append(sorting)
    return val


class RectangularBlock:
    """
    Define a rotated rectangular block in 3D space
    :param length: U-size of the block
    :param width:  V-size of the block
    :param depth:  W-size of the block
    :param center: Position of the prism center
    :param dip: Orientation of the u-axis in degree from horizontal
    :param azimuth: Orientation of the u axis in degree from north
    :param reference: Point of rotation to be 'center' or 'top'
    """

    def __init__(self, **kwargs):
        self._center: list[float] = [0.0, 0.0, 0.0]
        self._length: float = 1.0
        self._width: float = 1.0
        self._depth: float = 1.0
        self._dip: float = 0.0
        self._azimuth: float = 0.0
        self._vertices: np.ndarray | None = None
        self._reference: str = "center"
        self.triangles: np.ndarray = np.vstack(
            [
                [0, 2, 1],
                [1, 2, 3],
                [0, 1, 4],
                [4, 1, 5],
                [1, 3, 5],
                [5, 3, 7],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [4, 5, 6],
                [6, 5, 7],
            ]
        )

        for attr, item in kwargs.items():
            try:
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def center(self) -> list[float]:
        """Prism center"""
        return self._center

    @center.setter
    def center(self, value: list[float]):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(
                "Input value for 'center' must be a list of floats len(3)."
            )
        self._center = value
        self._vertices = None

    @property
    def length(self) -> float:
        """U-size of the block"""
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError("Input value for 'length' must be a float >0.")

        self._length = value
        self._vertices = None

    @property
    def width(self) -> float:
        """V-size of the block"""
        return self._width

    @width.setter
    def width(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError("Input value for 'width' must be a float >0.")

        self._width = value
        self._vertices = None

    @property
    def depth(self) -> float:
        """W-size of the block"""
        return self._depth

    @depth.setter
    def depth(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError("Input value for 'depth' must be a float >0.")

        self._depth = value
        self._vertices = None

    @property
    def dip(self) -> float:
        """Orientation of the u-axis in degree from horizontal"""
        return self._dip

    @dip.setter
    def dip(self, value):
        if not isinstance(value, float) or value < -90.0 or value > 90.0:
            raise ValueError(
                "Input value for 'dip' must be a float on the interval [-90, 90] degrees."
            )

        self._dip = value
        self._vertices = None

    @property
    def azimuth(self) -> float:
        """Orientation of the u axis in degree from north"""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        if not isinstance(value, float) or value < -360.0 or value > 360.0:
            raise ValueError(
                "Input value for 'azimuth' must be a float on the interval [-360, 360] degrees."
            )

        self._azimuth = value
        self._vertices = None

    @property
    def reference(self) -> str:
        """Point of rotation to be 'center' or 'top'"""
        return self._reference

    @reference.setter
    def reference(self, value: str):
        if not isinstance(value, str) or value not in ["center", "top"]:
            raise ValueError(
                "Input value for 'reference' point should be a str from ['center', 'top']."
            )
        self._reference = value
        self._vertices = None

    @property
    def vertices(self) -> np.ndarray | None:
        """
        Prism eight corners in 3D space
        """

        if getattr(self, "_vertices", None) is None:
            x1, x2 = [
                -self.length / 2.0 + self.center[0],
                self.length / 2.0 + self.center[0],
            ]
            y1, y2 = [
                -self.width / 2.0 + self.center[1],
                self.width / 2.0 + self.center[1],
            ]
            z1, z2 = [
                -self.depth / 2.0 + self.center[2],
                self.depth / 2.0 + self.center[2],
            ]

            block_xyz = np.asarray(
                [
                    [x1, x2, x1, x2, x1, x2, x1, x2],
                    [y1, y1, y2, y2, y1, y1, y2, y2],
                    [z1, z1, z1, z1, z2, z2, z2, z2],
                ]
            )

            theta = (450.0 - np.asarray(self.azimuth)) % 360.0
            phi = -self.dip
            xyz = rotate_xyz(block_xyz.T, self.center, theta, phi)

            if self.reference == "top":
                offset = np.mean(xyz[4:, :], axis=0) - self._center
                self._center -= offset
                xyz -= offset

            self._vertices = xyz

        return self._vertices


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
    locs[
        below_core_ind, 2
    ] = core_bottom_elev  # sets locations below core to core bottom
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
        u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
        v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
        z_cell_delimiters=-(mesh.x0[2] + mesh.hz.sum() - mesh.vectorNz[::-1]),
        name=name,
    )

    top_padding = find_top_padding(object_out, h[2])
    object_out.origin["z"] += top_padding

    return object_out
