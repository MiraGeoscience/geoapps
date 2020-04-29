from .data import Data, PrimitiveTypeEnum


class BlobData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.BLOB

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 8-bit char type (native) (value '0' or '1').
    # For each index set to 1, an opaque data set named after the index (e.g. "1", "2", etc)
    # must be added under the Data instance, containing the binary data tied to that index.
    # No data value : 0
