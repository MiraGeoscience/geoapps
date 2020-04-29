# pylint: disable=R0904
# pylint: disable=R0912

from __future__ import annotations

import inspect
import uuid
import weakref
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from weakref import ReferenceType

import h5py
import numpy as np

from .. import data, groups, objects
from ..data import Data, DataType
from ..groups import CustomGroup, Group, PropertyGroup, RootGroup
from ..io import H5Reader, H5Writer
from ..objects import Cell, ObjectBase
from ..shared import weakref_utils
from ..shared.entity import Entity

if TYPE_CHECKING:
    from ..groups import group
    from ..objects import object_base
    from ..shared import EntityType


class Workspace:
    """
    The Workspace class manages all Entities created or imported from the *geoh5* structure.

    The basic requirements needed to create a Workspace are:

    :param h5file: File name of the target *geoh5* file.
        A new project is created if the target file cannot by found on disk.
    """

    _active_ref: ClassVar[ReferenceType[Workspace]] = type(None)  # type: ignore

    _attribute_map = {
        "Contributors": "contributors",
        "Distance unit": "distance_unit",
        "GA Version": "ga_version",
        "Version": "version",
    }

    def __init__(self, h5file: str = "Analyst.geoh5", **kwargs):

        self._contributors = np.asarray(
            ["UserName"], dtype=h5py.special_dtype(vlen=str)
        )
        self._root: Optional[Entity] = None
        self._distance_unit = "meter"
        self._ga_version = "1"
        self._version = 1.0
        self._name = "GEOSCIENCE"
        self._types: Dict[uuid.UUID, ReferenceType[EntityType]] = {}
        self._groups: Dict[uuid.UUID, ReferenceType[group.Group]] = {}
        self._objects: Dict[uuid.UUID, ReferenceType[object_base.ObjectBase]] = {}
        self._data: Dict[uuid.UUID, ReferenceType[data.Data]] = {}
        self._modified_attributes = False
        self._h5file = h5file

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

        try:
            open(self.h5file)
            proj_attributes = H5Reader.fetch_project_attributes(self.h5file)
            for key, attr in proj_attributes.items():
                setattr(self, self._attribute_map[key], attr)

            # Get the Root attributes
            attributes, type_attributes, _ = H5Reader.fetch_attributes(
                self.h5file, uuid.uuid4(), "Root"
            )
            self._root = self.create_entity(
                RootGroup, save_on_creation=False, **{**attributes, **type_attributes}
            )

            if self._root is not None:
                self._root.existing_h5_entity = True
                self._root.entity_type.existing_h5_entity = True

                self.fetch_children(self._root, recursively=True)

        except FileNotFoundError:
            H5Writer.create_geoh5(self)
            self._root = self.create_entity(RootGroup)
            self.finalize()

    @property
    def attribute_map(self) -> dict:
        """
        Mapping between names used in the geoh5 database.
        """
        return self._attribute_map

    @property
    def ga_version(self) -> str:
        """
        Version of Geoscience Analyst software
        """
        return self._ga_version

    @ga_version.setter
    def ga_version(self, value: str):
        self._ga_version = value

    @property
    def version(self) -> float:
        """
        Project version
        """
        return self._version

    @version.setter
    def version(self, value: float):
        self._version = value

    @property
    def distance_unit(self) -> str:
        """
        Distance unit used in the project
        """
        return self._distance_unit

    @distance_unit.setter
    def distance_unit(self, value: str):
        self._distance_unit = value

    @property
    def contributors(self) -> np.ndarray:
        """
        List of contributors name
        """
        return self._contributors

    @contributors.setter
    def contributors(self, value: List[str]):
        self._contributors = np.asarray(value, dtype=h5py.special_dtype(vlen=str))

    @property
    def name(self) -> str:
        """
         Name of the project
        """
        return self._name

    @property
    def list_groups_name(self) -> Dict[uuid.UUID, str]:
        """
        List all registered groups with name
        """
        groups_name = {}
        for key, val in self._groups.items():
            entity = val.__call__()
            if entity is not None:
                groups_name[key] = entity.name
        return groups_name

    @property
    def list_objects_name(self) -> Dict[uuid.UUID, str]:
        """
        List all registered objects with name
        """
        objects_name = {}
        for key, val in self._objects.items():
            entity = val.__call__()
            if entity is not None:
                objects_name[key] = entity.name
        return objects_name

    @property
    def list_data_name(self) -> Dict[uuid.UUID, str]:
        """
        List all registered data with name
        """
        data_name = {}
        for key, val in self._data.items():
            entity = val.__call__()
            if entity is not None:
                data_name[key] = entity.name
        return data_name

    @property
    def list_entities_name(self) -> Dict[uuid.UUID, str]:
        """
        List all registered entities with name
        """
        entities_name = self.list_groups_name
        entities_name.update(self.list_objects_name)
        entities_name.update(self.list_data_name)
        return entities_name

    @property
    def root(self) -> Optional[Entity]:
        """
        RootGroup entity with no parent
        """
        return self._root

    @root.setter
    def root(self, entity) -> Optional[Entity]:
        if self._root is None:
            assert isinstance(
                entity, RootGroup
            ), f"The given root entity must be of type {RootGroup}"
            self._root = entity

        return self._root

    @property
    def h5file(self) -> str:
        """
        Target *geoh5* file name with path
        """
        return self._h5file

    @property
    def modified_attributes(self) -> bool:
        """
        Flag to update the workspace attributes on file
        """
        return self._modified_attributes

    @modified_attributes.setter
    def modified_attributes(self, value: bool):
        self._modified_attributes = value

    @property
    def workspace(self):
        """
        :return self: The Workspace
        """
        return self

    @staticmethod
    def active() -> Workspace:
        """ Get the active workspace. """
        active_one = Workspace._active_ref()
        if active_one is None:
            raise RuntimeError("No active workspace.")

        # so that type check does not complain of possible returned None
        return cast(Workspace, active_one)

    @classmethod
    def create(cls, entity: Entity, **kwargs) -> Entity:
        """
        Create and register a new Entity.

        :param entity: Entity to be created
        :param kwargs: List of attributes to set on new entity

        :return entity: The new entity
        """
        return entity.create(cls, **kwargs)

    @staticmethod
    def save_entity(entity: Entity, close_file: bool = True, add_children: bool = True):
        """
        Save or update an entity to geoh5

        :param entity: Entity to be written to geoh5
        :param close_file: Close the geoh5 database after writing is completed
        :param add_children: Add children entities to geoh5
        """
        H5Writer.save_entity(entity, close_file=close_file, add_children=add_children)

    def finalize(self):
        """ Finalize the h5file by checking for updated entities and re-building the Root"""
        for entity in self.all_objects() + self.all_groups() + self.all_data():
            if len(entity.modified_attributes) > 0:
                self.save_entity(entity)

        for entity_type in self.all_types():
            if len(entity_type.modified_attributes) > 0:
                H5Writer.add_entity_type(entity_type)

        H5Writer.finalize(self)

    def get_entity(self, name: Union[str, uuid.UUID]) -> List[Optional[Entity]]:
        """
        Retrieve an entity from one of its identifier, either by name or uuid

        :param name: Object identifier, either name or uuid.

        :return: object_list: List of entities with the same given name
        """
        if isinstance(name, uuid.UUID):
            list_entity_uid = [name]

        else:  # Extract all objects uuid with matching name
            list_entity_uid = [
                key for key, val in self.list_entities_name.items() if val == name
            ]

        entity_list: List[Optional[Entity]] = []
        for uid in list_entity_uid:
            entity_list.append(self.find_entity(uid))

        return entity_list

    def create_entity(
        self, entity_class, save_on_creation=True, **kwargs
    ) -> Optional[Entity]:
        """
        create_entity(entity_class, name, uuid.UUID, type_uuid)

        Function to create and register a new entity and its entity_type.

        :param entity_class: Type of entity to be created

        :return entity: Newly created entity registered to the workspace
        """
        created_entity: Optional[Entity] = None

        # Assume that entity is being created from its class
        entity_kwargs: Dict = dict()
        if "entity" in kwargs.keys():
            entity_kwargs = kwargs["entity"]

        entity_type_kwargs: Dict = dict()
        if "entity_type" in kwargs.keys():
            entity_type_kwargs = kwargs["entity_type"]

        if entity_class is RootGroup:
            created_entity = RootGroup(
                RootGroup.find_or_create_type(self, **entity_type_kwargs),
                **entity_kwargs,
            )
        elif entity_class is Data:
            created_entity = self.create_data(
                entity_class, entity_kwargs, entity_type_kwargs
            )
        else:
            created_entity = self.create_object_or_group(
                entity_class, entity_kwargs, entity_type_kwargs
            )

        if created_entity is not None:
            if save_on_creation:
                self.save_entity(created_entity)
            return created_entity

        return None

    def create_data(
        self, entity_class, entity_kwargs, entity_type_kwargs
    ) -> Optional[Entity]:
        """

        :param entity_class:
        :param entity_kwargs:
        :param entity_type_kwargs:
        :returns entity:
        """
        if isinstance(entity_type_kwargs, DataType):
            data_type = entity_type_kwargs
        else:
            data_type = data.data_type.DataType.find_or_create(
                self, **entity_type_kwargs
            )

        for _, member in inspect.getmembers(data):
            if (
                inspect.isclass(member)
                and issubclass(member, entity_class)
                and member is not entity_class
                and hasattr(member, "primitive_type")
                and inspect.ismethod(member.primitive_type)
                and data_type.primitive_type is member.primitive_type()
            ):
                created_entity = member(data_type, **entity_kwargs)

                return created_entity

        return None

    def create_object_or_group(
        self, entity_class, entity_kwargs, entity_type_kwargs
    ) -> Optional[Entity]:
        """

        :param entity_class:
        :param entity_kwargs:
        :param entity_type_kwargs:
        :return entity:
        """
        entity_type_uid = None
        for key, val in entity_type_kwargs.items():
            if key.lower() in ["id", "uid"]:
                entity_type_uid = uuid.UUID(str(val))

        if entity_type_uid is None:
            if hasattr(entity_class, "default_type_uid"):
                entity_type_uid = entity_class.default_type_uid()
            else:
                entity_type_uid = uuid.uuid4()

        entity_class = entity_class.__bases__
        for _, member in inspect.getmembers(groups) + inspect.getmembers(objects):
            if (
                inspect.isclass(member)
                and issubclass(member, entity_class)
                and member is not entity_class
                and hasattr(member, "default_type_uid")
                and not member == CustomGroup
                and member.default_type_uid() == entity_type_uid
            ):
                entity_type = member.find_or_create_type(self, **entity_type_kwargs)
                created_entity = member(entity_type, **entity_kwargs)

                return created_entity

        # Special case for CustomGroup without uuid
        if entity_class == Group:
            entity_type = groups.custom_group.CustomGroup.find_or_create_type(
                self, **entity_type_kwargs
            )
            created_entity = groups.custom_group.CustomGroup(
                entity_type, **entity_kwargs
            )

            return created_entity

        return None

    def copy_to_parent(self, entity, parent, copy_children: bool = True):
        """
        Function to copy an entity to a different parent entity

        :param entity: Entity to be copied
        :param parent: Target parent to copy the entity under
        :param copy_children: Copy all children of the entity

        :return: Entity: Registered to the workspace.
        """

        entity_kwargs: dict = {"entity": {}}
        for key in entity.__dict__.keys():
            if key not in ["_uid", "_entity_type"]:
                entity_kwargs["entity"][key[1:]] = getattr(entity, key[1:])

        entity_type_kwargs: dict = {"entity_type": {}}
        for key in entity.entity_type.__dict__.keys():
            if key not in ["_workspace"]:
                entity_type_kwargs["entity_type"][key[1:]] = getattr(
                    entity.entity_type, key[1:]
                )

        if parent is None:
            parent = entity.parent
        elif isinstance(parent, Workspace):
            parent = parent.root

        entity_kwargs["entity"]["parent"] = parent

        if isinstance(entity, Data):
            entity_type = Data
        else:
            entity_type = type(entity)

        new_object = parent.workspace.create_entity(
            entity_type, **{**entity_kwargs, **entity_type_kwargs}
        )

        if copy_children:
            for child in entity.children:
                new_object.add_children(
                    [self.copy_to_parent(child, parent=new_object, copy_children=True)]
                )

        new_object.workspace.finalize()

        return new_object

    def fetch_children(self, entity: Entity, recursively: bool = False):
        """
        Recover and register children entities from the h5file

        :param entity: Parental entity
        :param recursively: Recover all children down the project tree
        """
        base_classes = {"group": Group, "object": ObjectBase, "data": Data}

        if isinstance(entity, Group):
            entity_type = "group"
        elif isinstance(entity, ObjectBase):
            entity_type = "object"
        else:
            entity_type = "data"

        children_list = H5Reader.fetch_children(self.h5file, entity.uid, entity_type)

        for uid, child_type in children_list.items():
            attributes, type_attributes, property_groups = H5Reader.fetch_attributes(
                self.h5file, uid, child_type
            )

            if self.get_entity(uid)[0] is not None:
                recovered_object = self.get_entity(uid)[0]
            else:
                recovered_object = self.create_entity(
                    base_classes[child_type],
                    save_on_creation=False,
                    **{**attributes, **type_attributes},
                )

            if recovered_object is not None:

                # Assumes the object was pulled from h5
                recovered_object.existing_h5_entity = True
                recovered_object.entity_type.existing_h5_entity = True

                # Add parent-child relationship
                recovered_object.parent = entity

                if recursively:
                    self.fetch_children(recovered_object, recursively=True)

            if isinstance(recovered_object, ObjectBase) and len(property_groups) > 0:
                for kwargs in property_groups.values():
                    recovered_object.find_or_create_property_group(**kwargs)

    def fetch_values(self, uid: uuid.UUID) -> Optional[float]:
        """
        Fetch the data values from the source h5file

        :param uid: Unique identifier of target data object

        :return value: Array of values
        """
        return H5Reader.fetch_values(self.h5file, uid)

    def fetch_vertices(self, uid: uuid.UUID) -> np.ndarray:
        """
        Fetch the vertices of an object from the source h5file

        :param uid: Unique identifier of target entity

        :return coordinates: Array of coordinate [x, y, z] locations
        """
        return H5Reader.fetch_vertices(self.h5file, uid)

    def fetch_cells(self, uid: uuid.UUID) -> Cell:
        """
        Fetch the cells of an object from the source h5file

        :param uid: Unique identifier of target entity

        :return cells: Cell object with vertices index
        """
        return H5Reader.fetch_cells(self.h5file, uid)

    def fetch_octree_cells(self, uid: uuid.UUID) -> np.ndarray:
        """
        Fetch the octree cells ordering from the source h5file

        :param uid: Unique identifier of target entity

        :return values: Array of [i, j, k, dimension] defining the octree mesh
        """
        return H5Reader.fetch_octree_cells(self.h5file, uid)

    def fetch_delimiters(
        self, uid: uuid.UUID
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch the delimiter attributes from the source h5file

        :param uid: Unique identifier of target data object

        :return (u_delimiters, v_delimiters, z_delimiters):
            Arrays of delimiters along the u, v, and w axis
        """
        return H5Reader.fetch_delimiters(self.h5file, uid)

    def fetch_property_groups(self, uid: uuid.UUID) -> List[PropertyGroup]:
        """
        Fetch all property_groups on an object from the source h5file

        :param uid: Unique identifier of target object
        """
        group_dict = H5Reader.fetch_property_groups(self.h5file, uid)

        property_groups = []
        for pg_id, attrs in group_dict.items():

            group = PropertyGroup(uid=uuid.UUID(pg_id))

            for attr, val in attrs.items():

                try:
                    setattr(group, group.attribute_map[attr], val)
                except AttributeError:
                    continue

            property_groups.append(group)

        return property_groups

    def activate(self):
        """ Makes this workspace the active one.

            In case the workspace gets deleted, Workspace.active() safely returns None.
        """
        if Workspace._active_ref() is not self:
            Workspace._active_ref = weakref.ref(self)

    def deactivate(self):
        """ Deactivate this workspace if it was the active one, else does nothing.
        """
        if Workspace._active_ref() is self:
            Workspace._active_ref = type(None)

    def _register_type(self, entity_type: "EntityType"):
        weakref_utils.insert_once(self._types, entity_type.uid, entity_type)

    def _register_group(self, group: "group.Group"):
        weakref_utils.insert_once(self._groups, group.uid, group)

    def _register_data(self, data_obj: "data.Data"):
        weakref_utils.insert_once(self._data, data_obj.uid, data_obj)

    def _register_object(self, obj: "object_base.ObjectBase"):
        weakref_utils.insert_once(self._objects, obj.uid, obj)

    def all_types(self) -> List["EntityType"]:
        """Get all active entity types registered in the workspace.
        """
        weakref_utils.remove_none_referents(self._types)
        return [cast("EntityType", v()) for v in self._types.values()]

    def find_type(
        self, type_uid: uuid.UUID, type_class: Type["EntityType"]
    ) -> Optional["EntityType"]:
        """
        Find an existing and active EntityType

        :param type_uid: Unique identifier of target type
        """
        found_type = weakref_utils.get_clean_ref(self._types, type_uid)
        return found_type if isinstance(found_type, type_class) else None

    def all_groups(self) -> List["group.Group"]:
        """Get all active Group entities registered in the workspace.
        """
        weakref_utils.remove_none_referents(self._groups)
        return [cast("group.Group", v()) for v in self._groups.values()]

    def find_group(self, group_uid: uuid.UUID) -> Optional["group.Group"]:
        """
        Find an existing and active Group object
        """
        return weakref_utils.get_clean_ref(self._groups, group_uid)

    def all_objects(self) -> List["object_base.ObjectBase"]:
        """Get all active Object entities registered in the workspace.
        """
        weakref_utils.remove_none_referents(self._objects)
        return [cast("object_base.ObjectBase", v()) for v in self._objects.values()]

    def find_object(self, object_uid: uuid.UUID) -> Optional["object_base.ObjectBase"]:
        """
        Find an existing and active Object
        """
        return weakref_utils.get_clean_ref(self._objects, object_uid)

    def all_data(self) -> List["data.Data"]:
        """Get all active Data entities registered in the workspace.
        """
        weakref_utils.remove_none_referents(self._data)
        return [cast("data.Data", v()) for v in self._data.values()]

    def find_data(self, data_uid: uuid.UUID) -> Optional["data.Data"]:
        """
        Find an existing and active Data entity
        """
        return weakref_utils.get_clean_ref(self._data, data_uid)

    def find_entity(self, entity_uid: uuid.UUID) -> Optional["Entity"]:
        """Get all active entities registered in the workspace.
        """
        return (
            self.find_group(entity_uid)
            or self.find_data(entity_uid)
            or self.find_object(entity_uid)
        )


@contextmanager
def active_workspace(workspace: Workspace):
    previous_active_ref = Workspace._active_ref  # pylint: disable=protected-access
    workspace.activate()
    yield workspace

    workspace.deactivate()
    # restore previous active workspace when leaving the context
    previous_active = previous_active_ref()
    if previous_active is not None:
        previous_active.activate()  # pylint: disable=no-member
