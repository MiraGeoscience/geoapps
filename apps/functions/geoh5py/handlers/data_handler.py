# pylint: skip-file

import uuid
from typing import TYPE_CHECKING, List

from .. import interfaces
from ..workspace import Workspace

if TYPE_CHECKING:
    from ..interfaces.data import Data as i_Data
    from ..interfaces.data import DataQuery as i_DataQuery
    from ..interfaces.data import DataType as i_DataType
    from ..interfaces.data import DataTypeQuery as i_DataTypeQuery
    from ..interfaces.data import ReferencedValues as i_ReferencedValues
    from ..interfaces.data import DataSlab as i_DataSlab
    from ..interfaces.shared import Uuid as i_Uuid


class DataHandler:
    @staticmethod
    def get_all() -> List[i_Data]:
        Workspace.active().all_data()
        # TODO
        return []

    def find(self, query: i_DataQuery) -> List[i_Data]:
        # TODO
        pass

    def get(self, uid: i_Uuid) -> i_Data:
        Workspace.active().find_data(uuid.UUID(uid.id))
        # TODO
        return interfaces.data.Data()

    def get_float_values(self, data: i_Uuid, slab: i_DataSlab) -> List[float]:
        # TODO
        pass

    def get_integer_values(self, data: i_Uuid, slab: i_DataSlab) -> List[int]:
        # TODO
        pass

    def get_text_values(self, data: i_Uuid, slab: i_DataSlab) -> List[str]:
        # TODO
        pass

    def get_referenced_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> i_ReferencedValues:
        # TODO
        pass

    def get_datetime_values(self, data: i_Uuid, slab: i_DataSlab) -> List[str]:
        # TODO
        pass

    def get_filename_values(self, data: i_Uuid, slab: i_DataSlab) -> List[str]:
        # TODO
        pass

    def get_file_content(self, data: i_Uuid, file_name: str) -> str:
        # TODO
        pass

    def get_blob_values(self, data: i_Uuid, slab: i_DataSlab) -> List[int]:
        # TODO
        pass

    def get_blob_element(self, data: i_Uuid, index: int) -> str:
        # TODO
        pass

    def get_all_types(self,) -> List[i_DataType]:
        # TODO
        pass

    def find_types(self, query: i_DataTypeQuery) -> List[i_DataType]:
        # TODO
        pass

    def get_type(self, uid: i_Uuid) -> i_DataType:
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
