import uuid

from .object_base import ObjectBase, ObjectType


class Label(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0xE79F449D, 0x74E3, 0x4598, 0x9C, 0x9C, 0x351A28B8B69E)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        # TODO
        self.target_position = None
        self.label_position = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Label"

        # if object_type.description is None:
        #     self.entity_type.description = "Label"

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
