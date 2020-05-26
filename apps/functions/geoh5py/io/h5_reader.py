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
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np


class H5Reader:
    """
    Class to read information from a geoh5 file.
    """

    @classmethod
    def fetch_attributes(
        cls, h5file: str, uid: uuid.UUID, entity_type: str
    ) -> Tuple[dict, dict, dict]:
        """
        Get attributes of an :obj:`~geoh5py.shared.entity.Entity`

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'

        Returns
        -------
        attributes: :obj:`dict` of attributes for the :obj:`~geoh5py.shared.entity.Entity`
        type_attributes: :obj:`dict` of attributes for the :obj:`~geoh5py.shared.entity.EntityType`
        property_groups: :obj:`dict` of data :obj:`uuid.UUID`
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        attributes: Dict = {"entity": {}}
        type_attributes: Dict = {"entity_type": {}}
        property_groups: Dict = {}
        if "type" in entity_type:
            entity_type = entity_type.replace("_", " ").capitalize() + "s"
            entity = project[name]["Types"][entity_type][cls.uuid_str(uid)]
        elif entity_type == "Root":
            entity = project[name][entity_type]
        else:
            entity_type = entity_type.capitalize()
            if entity_type in ["Group", "Object"]:
                entity_type += "s"
            entity = project[name][entity_type][cls.uuid_str(uid)]

        for key, value in entity.attrs.items():
            attributes["entity"][key] = value

        for key, value in entity["Type"].attrs.items():
            type_attributes["entity_type"][key] = value

        if "Color map" in entity["Type"].keys():
            type_attributes["entity_type"]["color_map"] = {}
            for key, value in entity["Type"]["Color map"].attrs.items():
                type_attributes["entity_type"]["color_map"][key] = value
            type_attributes["entity_type"]["color_map"]["values"] = entity["Type"][
                "Color map"
            ][:]

        # Check if the entity has property_group
        if "PropertyGroups" in entity.keys():
            for pg_id in entity["PropertyGroups"].keys():
                property_groups[pg_id] = {"uid": pg_id}
                for key, value in entity["PropertyGroups"][pg_id].attrs.items():
                    property_groups[pg_id][key] = value

        project.close()

        attributes["entity"]["existing_h5_entity"] = True
        return attributes, type_attributes, property_groups

    @classmethod
    def fetch_cells(cls, h5file: Optional[str], uid: uuid.UUID) -> np.ndarray:
        """
        Get an object's :obj:`~geoh5py.objects.object_base.ObjectBase.cells`

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier of the target object

        :return cells: :obj:`numpy.ndarray` of :obj:`int`
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        indices = project[name]["Objects"][cls.uuid_str(uid)]["Cells"][:]

        project.close()

        return indices

    @classmethod
    def fetch_children(cls, h5file: str, uid: uuid.UUID, entity_type: str) -> dict:
        """
        Get :obj:`~geoh5py.shared.entity.Entity.children` of an
        :obj:`~geoh5py.shared.entity.Entity`

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'

        :return children: [{uuid: type}, ... ]
            List of dictionaries for the children uid and type
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        children = {}
        entity_type = entity_type.capitalize()
        if entity_type in ["Group", "Object"]:
            entity_type += "s"
        entity = project[name][entity_type][cls.uuid_str(uid)]

        for child_type, child_list in entity.items():
            if child_type in ["Type", "PropertyGroups"]:
                continue

            if isinstance(child_list, h5py.Group):
                for uid_str in child_list.keys():
                    children[cls.uuid_value(uid_str)] = child_type.replace(
                        "s", ""
                    ).lower()

        project.close()

        return children

    @classmethod
    def fetch_delimiters(
        cls, h5file: Optional[str], uid: uuid.UUID
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the delimiters of a :obj:`~geoh5py.objects.block_model.BlockModel`

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        Returns
        -------
        u_delimiters: :obj:`numpy.ndarray` of u_delimiters
        v_delimiters: :obj:`numpy.ndarray` of v_delimiters
        z_delimiters: :obj:`numpy.ndarray` of z_delimiters
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        u_delimiters = np.r_[
            project[name]["Objects"][cls.uuid_str(uid)]["U cell delimiters"]
        ]
        v_delimiters = np.r_[
            project[name]["Objects"][cls.uuid_str(uid)]["V cell delimiters"]
        ]
        z_delimiters = np.r_[
            project[name]["Objects"][cls.uuid_str(uid)]["Z cell delimiters"]
        ]

        project.close()

        return u_delimiters, v_delimiters, z_delimiters

    @classmethod
    def fetch_octree_cells(cls, h5file: Optional[str], uid: uuid.UUID) -> np.ndarray:
        """
        Get :obj:`~geoh5py.objects.octree.Octree`
        :obj:`~geoh5py.objects.object_base.ObjectBase.cells`.

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return octree_cells: :obj:`numpy.ndarray` of :obj:`int`
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        octree_cells = np.r_[
            project[name]["Objects"][cls.uuid_str(uid)]["Octree Cells"]
        ]

        project.close()

        return octree_cells

    @classmethod
    def fetch_project_attributes(cls, h5file: str) -> Dict[Any, Any]:
        """
        Get attributes of an :obj:`~geoh5py.shared.entity.Entity`

        :param h5file: Name of the target geoh5 file

        :return attributes: :obj:`dict` of attributes
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        attributes = {}

        for key, value in project[name].attrs.items():
            attributes[key] = value

        project.close()

        return attributes

    @classmethod
    def fetch_property_groups(
        cls, h5file: Optional[str], uid: uuid.UUID
    ) -> Dict[str, Dict[str, str]]:
        r"""
        Get the property groups

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return property_group_attributes: :obj:`dict` of property groups
            and respective attributes

        .. code-block:: python

            property_group = {
                "group_1": {"attribute": value, ...},
                ...,
                "group_N": {"attribute": value, ...},
            }
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        pg_handle = project[name]["Objects"][cls.uuid_str(uid)]["PropertyGroups"]

        property_groups: Dict[str, Dict[str, str]] = {}
        for pg_uid in pg_handle.keys():

            property_groups[pg_uid] = {}
            for attr, value in pg_handle[pg_uid].attrs.items():
                property_groups[pg_uid][attr] = value

        return property_groups

    @classmethod
    def fetch_values(cls, h5file: Optional[str], uid: uuid.UUID) -> Optional[float]:
        """
        Get data :obj:`~geoh5py.data.data.Data.values`

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return values: :obj:`numpy.array` of :obj:`float`
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        if "Data" in list(project[name]["Data"][cls.uuid_str(uid)].keys()):
            values = np.r_[project[name]["Data"][cls.uuid_str(uid)]["Data"]]
        else:
            values = None

        project.close()

        return values

    @classmethod
    def fetch_vertices(cls, h5file: Optional[str], uid: uuid.UUID) -> np.ndarray:
        """
        Get an object :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`

        :param h5file: Name of the target geoh5 file
        :param uid: Unique identifier of the target object

        :return vertices: :obj:`numpy.ndarray` of [x, y, z] coordinates

        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        x = project[name]["Objects"][cls.uuid_str(uid)]["Vertices"]["x"]
        y = project[name]["Objects"][cls.uuid_str(uid)]["Vertices"]["y"]
        z = project[name]["Objects"][cls.uuid_str(uid)]["Vertices"]["z"]

        project.close()

        return np.c_[x, y, z]

    @staticmethod
    def bool_value(value: np.int8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)

    @staticmethod
    def uuid_str(value: uuid.UUID) -> str:
        return "{" + str(value) + "}"
