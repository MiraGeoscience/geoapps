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
from discretize import TensorMesh, TreeMesh
from discretize.utils import mesh_utils
from geoh5py.groups import Group
from geoh5py.objects import BlockModel, DrapeModel, Octree
from geoh5py.shared import INTEGER_NDV
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.workspace import Workspace
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from geoapps.block_model_creation.driver import BlockModelDriver
from geoapps.driver_base.utils import running_mean
from geoapps.shared_utils.utils import octree_2_treemesh, rotate_xyz
from geoapps.utils.surveys import compute_alongline_distance, traveling_salesman


def drape_to_octree(
    octree: Octree,
    drape_model: DrapeModel | list[DrapeModel],
    children: dict[str, list[str]],
    active: np.ndarray,
    method: str = "lookup",
) -> Octree:
    """
    Interpolate drape model(s) into octree mesh.

    :param octree: Octree mesh to transfer values into
    :param drape_model: Drape model(s) whose values will be transferred
        into 'octree'.
    :param children: Dictionary containing a label and the associated
        names of the children in 'drape_model' to transfer into 'octree'.
    :param active: Active cell array for 'octree' model.
    :param method: Use 'lookup' to for a containing cell lookup method, or
        'nearest' for a nearest neighbor search method to transfer values

    :returns octree: Input octree mesh augmented with 'children' data from
        'drape_model' transferred onto cells using the prescribed 'method'.

    """
    if method not in ["nearest", "lookup"]:
        raise ValueError(f"Method must be 'nearest' or 'lookup'.  Provided {method}.")

    if isinstance(drape_model, DrapeModel):
        drape_model = [drape_model]

    if any(len(v) != len(drape_model) for v in children.values()):
        raise ValueError(
            f"Number of names and drape models must match.  "
            f"Provided {len(children)} names and {len(drape_model)} models."
        )

    if method == "nearest":
        # create tree to search nearest neighbors in stacked drape model
        tree = cKDTree(np.vstack([d.centroids for d in drape_model]))
        _, lookup_inds = tree.query(octree.centroids)
    else:
        mesh = octree_2_treemesh(octree)

    # perform interpolation using nearest neighbor or lookup method
    for label, names in children.items():
        octree_model = (
            [] if method == "nearest" else np.array([np.nan] * octree.n_cells)
        )
        for ind, model in enumerate(drape_model):
            datum = [k for k in model.children if k.name == names[ind]]
            if len(datum) > 1:
                raise ValueError(
                    f"Found more than one data set with name {names[ind]} in"
                    f"model {model.name}."
                )
            if method == "nearest":
                octree_model.append(datum[0].values)
            else:
                lookup_inds = (
                    mesh._get_containing_cell_indexes(  # pylint: disable=W0212
                        model.centroids
                    )
                )
                octree_model[lookup_inds] = datum[0].values

        if method == "nearest":
            octree_model = np.hstack(octree_model)[lookup_inds]
        else:
            octree_model = octree_model[mesh._ubc_order]  # pylint: disable=W0212

        if np.issubdtype(octree_model.dtype, np.integer):
            octree_model[~active] = INTEGER_NDV
        else:
            octree_model[~active] = np.nan  # apply active cells

        octree.add_data({label: {"values": octree_model}})

    return octree


def floating_active(mesh: TensorMesh | TreeMesh, active: np.ndarray):
    """
    True if there are any active cells in the air

    :param mesh: Tree mesh object
    :param active: active cells array
    """
    if not isinstance(mesh, (TreeMesh, TensorMesh)):
        raise TypeError("Input mesh must be of type TreeMesh or TensorMesh.")

    if mesh.dim == 2:
        gradient = mesh.stencil_cell_gradient_y
    else:
        gradient = mesh.stencil_cell_gradient_z

    return any(gradient * active > 0)


def get_drape_model(
    workspace: Workspace,
    name: str,
    locations: np.ndarray,
    h: list,
    depth_core: float,
    pads: list,
    expansion_factor: float,
    parent: Group | None = None,
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
    order = traveling_salesman(locations)

    # Smooth the locations
    xy_smooth = np.vstack(
        [
            np.c_[locations[order[0], :]].T,
            np.c_[
                running_mean(locations[order, 0], 2),
                running_mean(locations[order, 1], 2),
                running_mean(locations[order, 2], 2),
            ],
            np.c_[locations[order[-1], :]].T,
        ]
    )
    distances = compute_alongline_distance(xy_smooth)
    distances[:, -1] += locations[:, 2].max() - distances[:, -1].max() + h[1]
    x_interp = interp1d(distances[:, 0], xy_smooth[:, 0], fill_value="extrapolate")
    y_interp = interp1d(distances[:, 0], xy_smooth[:, 1], fill_value="extrapolate")

    mesh = mesh_utils.mesh_builder_xyz(
        distances,
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
    center_xy = np.c_[x_interp(mesh.cell_centers_x), y_interp(mesh.cell_centers_x)]
    for i, (x_center, y_center) in enumerate(center_xy):
        prisms.append([float(x_center), float(y_center), top, i * n_layers, n_layers])
        for k, b in enumerate(bottoms):
            layers.append([i, k, b])
            indices.append(index)
            index += 1

    prisms = np.vstack(prisms)
    layers = np.vstack(layers)
    layers[:, 2] = layers[:, 2][::-1]

    model = DrapeModel.create(
        workspace, layers=layers, name=name, prisms=prisms, parent=parent
    )
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
        sorting = sorting.reshape(mesh.shape_cells[1], mesh.shape_cells[0], order="C")
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
