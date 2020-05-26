#  Copyright (c) 2020 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

import uuid
from typing import Optional, Tuple

import numpy as np

from .object_base import ObjectBase, ObjectType


class BlockModel(ObjectBase):
    """
    Rectilinear 3D tensor mesh defined by three perpendicular axes.
    Each axis is divided into discrete intervals that define the cell dimensions.
    Nodal coordinates are determined relative to the origin and the sign of cell delimiters.
    Negative and positive cell delimiters
    are accepted to denote relative offsets from the origin.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0xB020A277, 0x90E2, 0x4CD7, 0x84, 0xD6, 0x612EE3F25051)
    )
    _attribute_map = ObjectBase._attribute_map.copy()
    _attribute_map.update({"Origin": "origin", "Rotation": "rotation"})

    def __init__(self, object_type: ObjectType, **kwargs):
        self._origin = [0, 0, 0]
        self._rotation = 0
        self._u_cell_delimiters = None
        self._v_cell_delimiters = None
        self._z_cell_delimiters = None
        self._centroids = None
        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @property
    def centroids(self):
        """
        :obj:`numpy.array`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.n_cells`, 3):
        Cell center locations in world coordinates.

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:

            cell_center_u = np.cumsum(self.u_cells) - self.u_cells / 2.0
            cell_center_v = np.cumsum(self.v_cells) - self.v_cells / 2.0
            cell_center_z = np.cumsum(self.z_cells) - self.z_cells / 2.0

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

            u_grid, v_grid, z_grid = np.meshgrid(
                cell_center_u, cell_center_v, cell_center_z
            )

            xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.ravel(z_grid)]

            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def n_cells(self) -> Optional[int]:
        """
        :obj:`int`: Total number of cells
        """
        if self.shape is not None:
            return np.prod(self.shape)
        return None

    @property
    def origin(self) -> np.array:
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the origin
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert (
                len(value) == 3
            ), "Origin must be a list or numpy array of shape (3, )"

            self.modified_attributes = "attributes"
            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._origin = value

    @property
    def rotation(self) -> float:
        """
        :obj:`float`: Clockwise rotation angle (degree) about the vertical axis
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "Rotation angle must be a float of shape (1,)"
            self.modified_attributes = "attributes"
            self._centroids = None

            self._rotation = value.astype(float)

    @property
    def shape(self) -> Optional[Tuple]:
        """
        :obj:`list` of :obj:`int`, len (3, ): Number of cells along the u, v and z-axis
        """
        if (
            self.u_cells is not None
            and self.v_cells is not None
            and self.z_cells is not None
        ):
            return tuple(
                [self.u_cells.shape[0], self.v_cells.shape[0], self.z_cells.shape[0]]
            )
        return None

    @property
    def u_cell_delimiters(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array` of :obj:`float`:
        Nodal offsets along the u-axis relative to the origin.
        """
        if (
            getattr(self, "_u_cell_delimiters", None) is None
        ) and self.existing_h5_entity:
            delimiters = self.workspace.fetch_delimiters(self.uid)
            self._u_cell_delimiters = delimiters[0]
            self._v_cell_delimiters = delimiters[1]
            self._z_cell_delimiters = delimiters[2]

        return self._u_cell_delimiters

    @u_cell_delimiters.setter
    def u_cell_delimiters(self, value):
        if value is not None:
            value = np.r_[value]
            self.modified_attributes = "cell_delimiters"
            self._centroids = None

            self._u_cell_delimiters = value.astype(float)

    @property
    def u_cells(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.shape` [0], ):
        Cell size along the u-axis
        """
        if self.u_cell_delimiters is not None:
            return self.u_cell_delimiters[1:] - self.u_cell_delimiters[:-1]
        return None

    @property
    def v_cell_delimiters(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array` of :obj:`float`:
        Nodal offsets along the v-axis relative to the origin.
        """
        if (
            getattr(self, "_v_cell_delimiters", None) is None
        ) and self.existing_h5_entity:
            delimiters = self.workspace.fetch_delimiters(self.uid)
            self._u_cell_delimiters = delimiters[0]
            self._v_cell_delimiters = delimiters[1]
            self._z_cell_delimiters = delimiters[2]

        return self._v_cell_delimiters

    @v_cell_delimiters.setter
    def v_cell_delimiters(self, value):
        if value is not None:
            value = np.r_[value]
            self.modified_attributes = "cell_delimiters"
            self._centroids = None

            self._v_cell_delimiters = value.astype(float)

    @property
    def v_cells(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.shape` [1], ):
        Cell size along the v-axis
        """
        if self.v_cell_delimiters is not None:
            return self.v_cell_delimiters[1:] - self.v_cell_delimiters[:-1]
        return None

    @property
    def z_cell_delimiters(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array` of :obj:`float`:
        Nodal offsets along the z-axis relative to the origin (positive up).
        """
        if (
            getattr(self, "_z_cell_delimiters", None) is None
        ) and self.existing_h5_entity:
            delimiters = self.workspace.fetch_delimiters(self.uid)
            self._u_cell_delimiters = delimiters[0]
            self._v_cell_delimiters = delimiters[1]
            self._z_cell_delimiters = delimiters[2]

        return self._z_cell_delimiters

    @z_cell_delimiters.setter
    def z_cell_delimiters(self, value):
        if value is not None:
            value = np.r_[value]
            self.modified_attributes = "cell_delimiters"
            self._centroids = None

            self._z_cell_delimiters = value.astype(float)

    @property
    def z_cells(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.shape` [2], ):
        Cell size along the z-axis
        """
        if self.z_cell_delimiters is not None:
            return self.z_cell_delimiters[1:] - self.z_cell_delimiters[:-1]
        return None
