import uuid
from typing import Any, Dict, Optional, Tuple

import h5py
from numpy import c_, int8, ndarray, r_


class H5Reader:
    """
        Class to read information from a geoh5 file.
    """

    @classmethod
    def fetch_project_attributes(cls, h5file: str) -> Dict[Any, Any]:
        """
        Get attributes og object from geoh5

        :param h5file: Name of the project h5file

        :return attributes: Dictionary of attributes recovered from geoh5
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        attributes = {}

        for key, value in project[name].attrs.items():
            attributes[key] = value

        project.close()

        return attributes

    @classmethod
    def fetch_attributes(
        cls, h5file: str, uid: uuid.UUID, entity_type: str
    ) -> Tuple[dict, dict, dict]:
        """
        Get attributes of object from geoh5

        :param h5file: Name of the project h5file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            "group", "data", "object", "group_type", "data_type", "object_type"

        :return attributes: Dictionary of attributes of an entity
        :return type_attributes: Dictionary of attributes of the entity type
        :return property_groups: Dictionary of property groups
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
    def fetch_children(cls, h5file: str, uid: uuid.UUID, entity_type: str) -> dict:
        """
        Get children of object from geoh5

        :param h5file: Name of the project h5file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            "group", "data", "object", "group_type", "data_type", "object_type"

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
    def fetch_vertices(cls, h5file: Optional[str], uid: uuid.UUID) -> ndarray:
        """
        Get the vertices of an object.

        :param h5file: Name of the project h5file
        :param uid: Unique identifier of the target object

        :return vertices: numpy.ndarray of float, shape ("*", 3)
            Array of coordinates [x, y, z]
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        x = project[name]["Objects"][cls.uuid_str(uid)]["Vertices"]["x"]
        y = project[name]["Objects"][cls.uuid_str(uid)]["Vertices"]["y"]
        z = project[name]["Objects"][cls.uuid_str(uid)]["Vertices"]["z"]

        project.close()

        return c_[x, y, z]

    @classmethod
    def fetch_cells(cls, h5file: Optional[str], uid: uuid.UUID) -> ndarray:
        """
        Get the cells of an object

        :param h5file: Name of the project h5file
        :param uid: Unique identifier of the target object

        :return cells: numpy.ndarray of int, shape ("*", 3)
            Array of vertex indices defining each cell
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        indices = project[name]["Objects"][cls.uuid_str(uid)]["Cells"][:]

        project.close()

        return indices

    @classmethod
    def fetch_values(cls, h5file: Optional[str], uid: uuid.UUID) -> Optional[float]:
        """
        Get the values of an entity

        :param h5file: Name of the project h5file
        :param uid: Unique identifier of the target entity

        :return values: array of float
            Array of values
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        if "Data" in list(project[name]["Data"][cls.uuid_str(uid)].keys()):
            values = r_[project[name]["Data"][cls.uuid_str(uid)]["Data"]]
        else:
            values = None

        project.close()

        return values

    @classmethod
    def fetch_delimiters(
        cls, h5file: Optional[str], uid: uuid.UUID
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the delimiters of an entity

        :param h5file: Name of the project h5file
        :param uid: Unique identifier of the target entity

        :return u_delimiters: Array of u_delimiters
        :return v_delimiters: Array of v_delimiters
        :return z_delimiters: Array of z_delimiters
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        u_delimiters = r_[
            project[name]["Objects"][cls.uuid_str(uid)]["U cell delimiters"]
        ]
        v_delimiters = r_[
            project[name]["Objects"][cls.uuid_str(uid)]["V cell delimiters"]
        ]
        z_delimiters = r_[
            project[name]["Objects"][cls.uuid_str(uid)]["Z cell delimiters"]
        ]

        project.close()

        return u_delimiters, v_delimiters, z_delimiters

    @classmethod
    def fetch_octree_cells(cls, h5file: Optional[str], uid: uuid.UUID) -> ndarray:
        """
        Get the cells of an octree mesh

        :param h5file: Name of the project h5file
        :param uid: Unique identifier of the target entity

        :return octree_cells: numpy.ndarray of int, shape ("*", 3)
            Array of octree_cells
        """
        project = h5py.File(h5file, "r")
        name = list(project.keys())[0]
        octree_cells = r_[project[name]["Objects"][cls.uuid_str(uid)]["Octree Cells"]]

        project.close()

        return octree_cells

    @classmethod
    def fetch_property_groups(
        cls, h5file: Optional[str], uid: uuid.UUID
    ) -> Dict[str, Dict[str, str]]:
        """
        Get the property groups of an object.

        :param h5file: Name of the project h5file
        :param uid: Unique identifier of the target entity

        :return property_group_attributes: dict
            Dictionary of property_groups and respective attributes
            {"prop_group1": {"attribute": value, ...}, ...}
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

    @staticmethod
    def bool_value(value: int8) -> bool:
        return bool(value)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        return uuid.UUID(value)

    @staticmethod
    def uuid_str(value: uuid.UUID) -> str:
        return "{" + str(value) + "}"
