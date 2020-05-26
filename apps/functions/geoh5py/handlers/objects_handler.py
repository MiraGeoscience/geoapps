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

# pylint: skip-file

import uuid
from typing import TYPE_CHECKING, Dict, List

from .. import interfaces
from ..workspace import Workspace

if TYPE_CHECKING:
    from ..interfaces.objects import Object as i_Object
    from ..interfaces.objects import ObjectQuery as i_ObjectQuery
    from ..interfaces.objects import Points as i_Points
    from ..interfaces.objects import Curve as i_Curve
    from ..interfaces.objects import Surface as i_Surface
    from ..interfaces.objects import Grid2D as i_Grid2D
    from ..interfaces.objects import BlockModel as i_BlockModel
    from ..interfaces.objects import Drillhole as i_Drillhole
    from ..interfaces.objects import GeoImage as i_GeoImage
    from ..interfaces.objects import Octree as i_Octree
    from ..interfaces.objects import Label as i_Label
    from ..interfaces.objects import GeometryTransformation as i_GeometryTransformation
    from ..interfaces.shared import Uuid as i_Uuid


# pylint: disable=too-many-public-methods
class ObjectsHandler:
    def get_type(self, object_class: int) -> i_Uuid:
        # TODO
        pass

    def get_class(self, type_uid: i_Uuid) -> int:
        # TODO
        pass

    @staticmethod
    def get_all() -> List[i_Object]:
        Workspace.active().all_data()
        # TODO
        return []

    def find(self, query: i_ObjectQuery) -> List[i_Object]:
        # TODO
        pass

    def set_allow_move(self, objects: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def move_to_group(self, objects: List[i_Uuid], destination_group: i_Uuid) -> None:
        # TODO
        pass

    def get(self, uid: i_Uuid) -> i_Object:
        Workspace.active().find_object(uuid.UUID(uid.id))
        # TODO
        return interfaces.objects.Object()

    def narrow_points(self, uid: i_Uuid) -> i_Points:
        # TODO
        pass

    def narrow_curve(self, uid: i_Uuid) -> i_Curve:
        # TODO
        pass

    def narrow_surface(self, uid: i_Uuid) -> i_Surface:
        # TODO
        pass

    def narrow_grid2d(self, uid: i_Uuid) -> i_Grid2D:
        # TODO
        pass

    def narrow_drillhole(self, uid: i_Uuid) -> i_Drillhole:
        # TODO
        pass

    def narrow_blockmodel(self, uid: i_Uuid) -> i_BlockModel:
        # TODO
        pass

    def narrow_octree(self, uid: i_Uuid) -> i_Octree:
        # TODO
        pass

    def narrow_geoimage(self, uid: i_Uuid) -> i_GeoImage:
        # TODO
        pass

    def narrow_label(self, uid: i_Uuid) -> i_Label:
        # TODO
        pass

    def create_any_object(
        self,
        type_uid: i_Uuid,
        name: str,
        parent_group: i_Uuid,
        attributes: Dict[str, str],
    ) -> i_Object:
        # TODO
        pass

    def transform(
        self, objects: List[i_Uuid], transformation: i_GeometryTransformation
    ) -> None:
        # TODO
        pass

    def set_public(self, entities: List[i_Uuid], is_public: bool) -> None:
        # TODO
        pass

    def set_visible(self, entities: List[i_Uuid], visible: bool) -> None:
        # TODO
        pass

    def set_allow_delete(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def set_allow_rename(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def rename(self, entities: i_Uuid, new_name: str) -> None:
        # TODO
        pass
