from .data import Data, PrimitiveTypeEnum


class DatetimeData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.DATETIME

    # TODO: implement specialization to access values.
    # Stored as a 1D array of variable-length strings formatted according to the ISO 8601
    # extended specification for representations of UTC dates and times (Qt implementation),
    # taking the form YYYY-MM-DDTHH:mm:ss[Z|[+|-]HH:mm]
    # No data value : empty string
