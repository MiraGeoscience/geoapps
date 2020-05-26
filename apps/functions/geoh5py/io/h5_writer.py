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

# pylint: disable=R0904

import json
import uuid
from typing import TYPE_CHECKING, Optional, Union

import h5py
import numpy as np

from ..data import CommentsData, Data, DataType
from ..groups import Group, GroupType, RootGroup
from ..objects import ObjectBase, ObjectType
from ..shared import Entity

if TYPE_CHECKING:
    from .. import shared
    from .. import workspace


class H5Writer:
    """
    Writing class to a geoh5 file.
    """

    str_type = h5py.special_dtype(vlen=str)

    @staticmethod
    def bool_value(value: np.int8) -> bool:
        """Convert integer to bool
        """
        return bool(value)

    @classmethod
    def create_geoh5(
        cls,
        workspace: "workspace.Workspace",
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
            Create a geoh5 file and add the core structure

            :param workspace: :obj:`~geoh5py.workspace.workspace.Workspace` object
                defining the project structure.
            :param file: Name or handle to a geoh5 file
            :param close_file: Close file after write

            :return h5file: Pointer to a geoh5 file
            """
        # Take default name
        if file is None:
            file = workspace.h5file

        # Returns default error if already exists
        h5file = h5py.File(file, "w-")

        # Write the workspace group
        project = h5file.create_group(workspace.name)

        cls.write_attributes(workspace, file=file, close_file=False)

        # Create base entity structure for geoh5
        project.create_group("Data")
        project.create_group("Groups")
        project.create_group("Objects")
        types = project.create_group("Types")
        types.create_group("Data types")
        types.create_group("Group types")
        types.create_group("Object types")

        if close_file:
            h5file.close()

        return h5file

    @staticmethod
    def fetch_h5_handle(
        file: Optional[Union[str, h5py.File]],
        entity: Union[Entity, "shared.EntityType"],
    ) -> h5py.File:
        """
        Open in read+ mode a geoh5 file

        :param file: Name or handle to a geoh5 file
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`

        :return h5py.File: Handle to an opened h5py file
        """
        if file is None:
            h5file = h5py.File(entity.workspace.h5file, "r+")

        else:
            if not isinstance(file, h5py.File):
                h5file = h5py.File(file, "r+")
            else:
                h5file = file

        return h5file

    @classmethod
    def fetch_handle(cls, file: Optional[Union[str, h5py.File]], entity):
        """
        Get a pointer to an :obj:`~geoh5py.shared.entity.Entity` in geoh5.

        :param file: Name or handle to a geoh5 file
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`

        :return entity_handle: Pointer to an existing entity or None if not found
        """
        cls.str_type = h5py.special_dtype(vlen=str)

        h5file = cls.fetch_h5_handle(file, entity)

        base = list(h5file.keys())[0]

        if entity.name == base:
            return h5file[base]

        uid = entity.uid

        if isinstance(entity, Data):
            base_handle = h5file[base]["Data"]
        elif isinstance(entity, ObjectBase):
            base_handle = h5file[base]["Objects"]
        elif isinstance(entity, Group):
            base_handle = h5file[base]["Groups"]
        elif isinstance(entity, DataType):
            base_handle = h5file[base]["Types"]["Data types"]
        elif isinstance(entity, ObjectType):
            base_handle = h5file[base]["Types"]["Object types"]
        elif isinstance(entity, GroupType):
            base_handle = h5file[base]["Types"]["Group types"]
        else:
            raise RuntimeError(f"Cannot add object '{entity}' to geoh5.")

        # Check if already in the project
        if cls.uuid_str(uid) in list(base_handle.keys()):
            return base_handle[cls.uuid_str(uid)]

        return None

    @classmethod
    def finalize(cls, workspace, close_file=False):
        """
        Add/replace the :obj:`~geoh5py.groups.root_group.RootGroup` in geoh5

        :param workspace: Workspace object defining the project structure
        :param close_file: Close file after write [False]
        """
        h5file = cls.fetch_h5_handle(workspace.h5file, workspace)
        workspace_group: Entity = workspace.get_entity("Workspace")[0]
        root_handle = H5Writer.fetch_handle(h5file, workspace_group)

        if "Root" in h5file[workspace.name].keys():
            del h5file[workspace.name]["Root"]

        h5file[workspace.name]["Root"] = root_handle

        if close_file:
            h5file.close()

    @classmethod
    def save_entity(
        cls,
        entity: Entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
        add_children: bool = True,
    ):
        """
        Write an :obj:`~geoh5py.shared.entity.Entity` to geoh5 with its
        :obj:`~geoh5py.shared.entity.Entity.children`

        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param file: Name or handle to a geoh5 file
        :param close_file: Close file after write
        :param add_children: Add :obj:`~geoh5py.shared.entity.Entity.children`
        """
        h5file = cls.fetch_h5_handle(file, entity)

        # Add itself to the project
        new_entity = H5Writer.write_entity(entity, file=h5file, close_file=False)

        if add_children:
            # Write children entities and add to current parent
            for child in entity.children:
                H5Writer.write_entity(child, file=h5file, close_file=False)
                H5Writer.write_to_parent(
                    child, file=h5file, close_file=False, recursively=False
                )

        H5Writer.write_to_parent(
            entity, file=h5file, close_file=False, recursively=True
        )

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        return new_entity

    @classmethod
    def update_attributes(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Update the attributes of an :obj:`~geoh5py.shared.entity.Entity`

        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param file: Name or handle to a geoh5 file
        :param close_file: Close file after write
        """
        file = cls.fetch_h5_handle(file, entity)
        entity_handle = H5Writer.fetch_handle(file, entity)

        for attr in entity.modified_attributes:
            if attr == "values":
                del entity_handle["Data"]
                cls.write_data_values(
                    entity, entity.values, file=file, close_file=close_file
                )

            elif attr == "cells":
                del entity_handle["Cells"]
                cls.write_cells(entity, file=file, close_file=close_file)

            elif attr == "vertices":
                del entity_handle["Vertices"]
                cls.write_vertices(entity, file=file, close_file=close_file)

            elif attr == "octree_cells":
                del entity_handle["Octree Cells"]
                cls.write_octree_cells(entity, file=file, close_file=close_file)

            elif attr == "property_groups":
                del entity_handle["PropertyGroups"]
                cls.write_property_groups(entity, file=file, close_file=close_file)

            elif attr == "cell_delimiters":
                cls.write_cell_delimiters(entity, file=file, close_file=close_file)

            elif attr == "Color map":
                del entity_handle["Color map"]
                cls.write_color_map(entity, file=file, close_file=close_file)

            else:
                cls.write_attributes(entity, file=file, close_file=close_file)

    @staticmethod
    def uuid_value(value: str) -> uuid.UUID:
        """Convert string to uuid.UUID
        """
        return uuid.UUID(value)

    @staticmethod
    def uuid_str(value: uuid.UUID) -> str:
        """Convert uuid.UUID to string used in geoh5
        """
        return "{" + str(value) + "}"

    @classmethod
    def write_attributes(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Write attributes of an :obj:`~geoh5py.shared.entity.Entity`

        :param file: Name or handle to a geoh5 file
        :param entity: Entity with attributes to be added to the geoh5 file
        :param close_file: Close file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)
        entity_handle = H5Writer.fetch_handle(file, entity)
        str_type = h5py.special_dtype(vlen=str)

        for key, attr in entity.attribute_map.items():

            try:
                value = getattr(entity, attr)
            except AttributeError:
                continue

            if key == "ID":
                value = "{" + str(value) + "}"

            if key == "PropertyGroups":
                continue

            if key in ["Association", "Primitive type"]:
                value = value.name.lower().capitalize()

            if isinstance(value, (np.int8, bool)):
                entity_handle.attrs.create(key, int(value), dtype="int8")

            elif isinstance(value, str):
                entity_handle.attrs.create(key, value, dtype=str_type)

            elif value is None:
                entity_handle.attrs.create(key, "None", dtype=str_type)

            else:
                entity_handle.attrs.create(key, value, dtype=np.asarray(value).dtype)
        if close_file:
            h5file.close()

    @classmethod
    def write_cell_delimiters(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add cell delimiters (u, v, z)  to a :obj:`~geoh5py.objects.block_model.BlockModel`.

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        :param close_file: Close geoh5 file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        if hasattr(entity, "u_cell_delimiters") and (
            entity.u_cell_delimiters is not None
        ):
            u_cell_delimiters = entity.u_cell_delimiters
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "U cell delimiters",
                data=u_cell_delimiters,
                dtype=u_cell_delimiters.dtype,
            )

        if hasattr(entity, "v_cell_delimiters") and (
            entity.v_cell_delimiters is not None
        ):
            v_cell_delimiters = entity.v_cell_delimiters
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "V cell delimiters",
                data=v_cell_delimiters,
                dtype=v_cell_delimiters.dtype,
            )

        if hasattr(entity, "z_cell_delimiters") and (
            entity.z_cell_delimiters is not None
        ):
            z_cell_delimiters = entity.z_cell_delimiters
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "Z cell delimiters",
                data=z_cell_delimiters,
                dtype=z_cell_delimiters.dtype,
            )

        if close_file:
            h5file.close()

    @classmethod
    def write_cells(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add :obj:`~geoh5py.objects.object_base.ObjectBase.cells`

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        :param close_file: Close geoh5 file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        if hasattr(entity, "cells") and (entity.cells is not None):
            indices = entity.cells
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "Cells", indices.shape, data=indices, dtype=indices.dtype
            )

        if close_file:
            h5file.close()

    @classmethod
    def write_color_map(
        cls,
        entity_type: "shared.EntityType",
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add :obj:`~geoh5py.data.color_map.ColorMap` to a
        :obj:`~geoh5py.data.data_type.DataType`

        :param file: Name or handle to a geoh5 file
        :param entity_type: Target entity_type with color_map
        :param close_file: Close geoh5 file after write
        """
        h5file = cls.fetch_h5_handle(file, entity_type)
        color_map = getattr(entity_type, "color_map", None)

        if color_map is not None and color_map.values is not None:
            entity_type_handle = H5Writer.fetch_handle(h5file, entity_type)
            map_values = color_map.values
            entity_type_handle.create_dataset(
                "Color map", map_values.shape, data=map_values, dtype=map_values.dtype
            )

        if close_file:
            h5file.close()

    @classmethod
    def write_data_values(
        cls,
        entity,
        values,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add data :obj:`~geoh5py.data.data.Data.values`

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        :param values: Array of values to be added to the geoh5 file
        :param close_file: Close geoh5 file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        entity_handle = H5Writer.fetch_handle(h5file, entity)

        # Adding an array of values
        if isinstance(entity, CommentsData):

            comments = {"Comments": values}
            entity_handle.create_dataset(
                "Data",
                data=json.dumps(comments, indent=4),
                dtype=h5py.special_dtype(vlen=str),
                shape=(1,),
            )

        else:
            entity_handle.create_dataset("Data", data=values)

        if close_file:
            h5file.close()

    @classmethod
    def write_entity(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        values=None,
        close_file: bool = True,
    ):
        """
        Add an :obj:`~geoh5py.shared.entity.Entity` and its attributes to geoh5.
        The function returns a pointer to the entity if already present on file.

        :param file: Name or handle to a geoh5 file
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param values: Array of values to be added to Data entity
        :param close_file: Close file after write

        :return entity: Pointer to the written entity. Active link if "close_file" is False
        """
        cls.str_type = h5py.special_dtype(vlen=str)

        h5file = cls.fetch_h5_handle(file, entity)

        base = list(h5file.keys())[0]

        if isinstance(entity, Data):
            entity_type = "Data"
        elif isinstance(entity, ObjectBase):
            entity_type = "Objects"
        else:
            entity_type = "Groups"

        uid = entity.uid

        # Check if already in the project
        if cls.uuid_str(uid) in list(h5file[base][entity_type].keys()):

            if any([entity.modified_attributes]):

                if "entity_type" in entity.modified_attributes:
                    entity_handle = cls.fetch_handle(h5file, entity)
                    new_type = H5Writer.write_entity_type(
                        entity.entity_type, file=h5file, close_file=False
                    )
                    del entity_handle["Type"]
                    entity_handle["Type"] = new_type

                else:
                    cls.update_attributes(entity, file=h5file, close_file=False)

                entity.modified_attributes = []
                entity.existing_h5_entity = True

            else:
                # Check if file reference to a hdf5
                if close_file:
                    h5file.close()
                    return None
                entity.existing_h5_entity = True
            return h5file[base][entity_type][cls.uuid_str(uid)]

        entity_handle = h5file[base][entity_type].create_group(cls.uuid_str(uid))

        if entity_type == "Groups":
            entity_handle.create_group("Data")
            entity_handle.create_group("Groups")
            entity_handle.create_group("Objects")
        elif entity_type == "Objects":
            entity_handle.create_group("Data")

        H5Writer.write_attributes(entity, file=h5file, close_file=False)

        # Add the type and return a pointer
        new_type = H5Writer.write_entity_type(
            entity.entity_type, file=h5file, close_file=False
        )
        entity_handle["Type"] = new_type

        entity.entity_type.modified_attributes = []
        entity.entity_type.existing_h5_entity = True

        cls.write_properties(entity, file=h5file, values=values, close_file=False)

        # Check if file reference to a hdf5
        if close_file:
            h5file.close()

        entity.modified_attributes = []
        entity.existing_h5_entity = True

        return entity_handle

    @classmethod
    def write_entity_type(
        cls,
        entity_type: "shared.EntityType",
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add an :obj:`~geoh5py.shared.entity_type.EntityType` to geoh5.

        :param entity_type: Entity with type to be added
        :param file: Name or handle to a geoh5 file
        :param close_file: Close file after write

        :return type: Pointer to :obj:`~geoh5py.shared.entity_type.EntityType` in geoh5
        """
        h5file = cls.fetch_h5_handle(file, entity_type)

        base = list(h5file.keys())[0]

        # tree = entity_type.workspace.tree
        uid = entity_type.uid

        # entity_type_str = tree[uid]["entity_type"].replace("_", " ").capitalize() + "s"
        if isinstance(entity_type, DataType):
            entity_type_str = "Data types"
        elif isinstance(entity_type, ObjectType):
            entity_type_str = "Object types"
        elif isinstance(entity_type, GroupType):
            entity_type_str = "Group types"
        else:
            return None

        # Check if already in the project
        if cls.uuid_str(uid) in list(h5file[base]["Types"][entity_type_str].keys()):

            if any([entity_type.modified_attributes]):
                cls.update_attributes(entity_type, file=h5file, close_file=False)
                entity_type.modified_attributes = []
                entity_type.existing_h5_entity = False

            else:
                entity_type.existing_h5_entity = True
            return h5file[base]["Types"][entity_type_str][cls.uuid_str(uid)]

        new_type = h5file[base]["Types"][entity_type_str].create_group(
            cls.uuid_str(uid)
        )
        H5Writer.write_attributes(entity_type, file=h5file, close_file=False)

        if hasattr(entity_type, "color_map"):
            H5Writer.write_color_map(entity_type, file=h5file, close_file=False)

        if close_file:
            h5file.close()

        entity_type.modified_attributes = False
        entity_type.existing_h5_entity = True

        return new_type

    @classmethod
    def write_octree_cells(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add :obj:`~geoh5py.object.object_base.ObjectBase.cells` of an
        :obj:`~geoh5py.object.octree.Octree` object to geoh5.

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity_type with color_map
        :param close_file: Close geoh5 file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        if hasattr(entity, "octree_cells") and (entity.octree_cells is not None):
            octree_cells = entity.octree_cells
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding cells
            entity_handle.create_dataset(
                "Octree Cells",
                octree_cells.shape,
                data=octree_cells,
                dtype=octree_cells.dtype,
            )

        if close_file:
            h5file.close()

    @classmethod
    def write_properties(
        cls,
        entity: Entity,
        file: Optional[Union[str, h5py.File]] = None,
        values: Optional[np.ndarray] = None,
        close_file: bool = True,
    ):
        """
        Add properties of an :obj:`~geoh5py.shared.entity.Entity`

        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param file: Name or handle to a geoh5 file
        :param values: Array of values to be added
        :param close_file: Close file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        if hasattr(entity, "values"):
            if values is not None:
                H5Writer.write_data_values(
                    entity, values, file=h5file, close_file=False
                )

            if isinstance(entity, Data):
                H5Writer.write_data_values(
                    entity, entity.values, file=h5file, close_file=False
                )

        if isinstance(entity, ObjectBase) and isinstance(entity.property_groups, list):
            H5Writer.write_property_groups(entity, file=h5file, close_file=False)

        if getattr(entity, "vertices", None) is not None:
            H5Writer.write_vertices(entity, file=h5file, close_file=False)

        if getattr(entity, "u_cell_delimiters", None) is not None:
            H5Writer.write_cell_delimiters(entity, file=h5file, close_file=False)

        if getattr(entity, "cells", None) is not None:
            H5Writer.write_cells(entity, file=h5file, close_file=False)

        if getattr(entity, "octree_cells", None) is not None:
            H5Writer.write_octree_cells(entity, file=h5file, close_file=False)

        if close_file:
            h5file.close()

    @classmethod
    def write_property_groups(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Write :obj:`~geoh5py.groups.property_group.PropertyGroup` associated with
        an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param close_file: Close file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        if hasattr(entity, "property_groups") and isinstance(
            entity.property_groups, list
        ):

            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Check if a group already exists, then remove and write
            if "PropertyGroups" in entity_handle.keys():
                del entity_handle["PropertyGroups"]

            entity_handle.create_group("PropertyGroups")
            for p_g in entity.property_groups:

                uid = cls.uuid_str(p_g.uid)
                if uid in entity_handle["PropertyGroups"].keys():
                    del entity_handle["PropertyGroups"][uid]

                entity_handle["PropertyGroups"].create_group(uid)

                group_handle = entity_handle["PropertyGroups"][uid]

                for key, attr in p_g.attribute_map.items():

                    try:
                        value = getattr(p_g, attr)
                    except AttributeError:
                        continue

                    if key == "Association":
                        value = value.name.capitalize()

                    elif key == "Properties":
                        value = np.asarray([cls.uuid_str(val) for val in value])

                    elif key == "ID":
                        value = cls.uuid_str(value)

                    group_handle.attrs.create(
                        key, value, dtype=h5py.special_dtype(vlen=str)
                    )
        if close_file:
            h5file.close()

    @classmethod
    def write_to_parent(
        cls,
        entity: Entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
        recursively=False,
    ):
        """
        Add/create an :obj:`~geoh5py.shared.entity.Entity` and add it to its parent.

        :param file: Name or handle to a geoh5 file
        :param entity: Entity to be added or linked to a parent in geoh5
        :param close_file: Close file after write: [True] or False
        :param recursively: Add parents recursively until reaching the
            :obj:`~geoh5py.groups.root_group.RootGroup`
        """

        h5file = cls.fetch_h5_handle(file, entity)

        # If RootGroup than no parent to be added
        if isinstance(entity, RootGroup):
            return

        # cls.str_type = h5py.special_dtype(vlen=str)

        uid = entity.uid

        # Get the h5 handle
        entity_handle = H5Writer.write_entity(entity, file=h5file, close_file=False)

        parent_handle = H5Writer.write_entity(
            entity.parent, file=h5file, close_file=False
        )

        if isinstance(entity, Data):
            entity_type = "Data"
        elif isinstance(entity, ObjectBase):
            entity_type = "Objects"
        elif isinstance(entity, Group):
            entity_type = "Groups"
        else:
            if close_file:
                h5file.close()
            return

        # Check if child h5py.Group already exists
        if entity_type not in parent_handle.keys():
            parent_handle.create_group(entity_type)

        # Check if child uuid not already in h5
        if cls.uuid_str(uid) not in list(parent_handle[entity_type].keys()):
            parent_handle[entity_type][cls.uuid_str(uid)] = entity_handle

        if recursively:
            H5Writer.write_to_parent(
                entity.parent, file=h5file, close_file=False, recursively=recursively
            )

        # Close file if requested
        if close_file:
            h5file.close()

    @classmethod
    def write_vertices(
        cls,
        entity,
        file: Optional[Union[str, h5py.File]] = None,
        close_file: bool = True,
    ):
        """
        Add :obj:`~geoh5py.objects.object_base.ObjectBase.vertices` of an object.

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        :param close_file: Close geoh5 file after write
        """
        h5file = cls.fetch_h5_handle(file, entity)

        if getattr(entity, "vertices", None) is not None:
            xyz = entity.vertices
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            # Adding vertices
            loc_type = np.dtype(
                [("x", np.float64), ("y", np.float64), ("z", np.float64)]
            )

            vertices = entity_handle.create_dataset(
                "Vertices", (xyz.shape[0],), dtype=loc_type
            )
            vertices["x"] = xyz[:, 0]
            vertices["y"] = xyz[:, 1]
            vertices["z"] = xyz[:, 2]

        if close_file:
            h5file.close()
