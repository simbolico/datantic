# src/datantic/fields.py
from typing import Any, Dict, List, Optional

import pandera as pa
from pandera.api.checks import Check
from pydantic import Field as PydanticField  # Alias pydantic.Field
from pydantic.fields import FieldInfo


def _sqlmodel_type_to_pandera_dtype(sa_type: Any,
                                    custom_mapping: Optional[Dict[Any,
                                                                  Any]] = None
                                    ) -> Any:
    """Maps SQLAlchemy types to Pandera dtypes (internal utility)."""
    from sqlalchemy import (Boolean, Date, DateTime, Float, Integer,
                            LargeBinary, Numeric, String, Time, Uuid, JSON,
                            Enum, SmallInteger, BigInteger, Unicode,
                            UnicodeText, DECIMAL, Text)
    from sqlalchemy.types import DOUBLE_PRECISION

    if custom_mapping is not None and not isinstance(custom_mapping, dict):
        raise TypeError(
            f"custom_mapping must be a dict, got {type(custom_mapping)}")

    if custom_mapping and sa_type in custom_mapping:
        return custom_mapping[sa_type]

    if isinstance(sa_type, String): return pa.String
    if isinstance(sa_type, Unicode): return pa.String
    if isinstance(sa_type, Text): return pa.String
    if isinstance(sa_type, UnicodeText): return pa.String
    if isinstance(sa_type, Integer): return pa.Int
    if isinstance(sa_type, SmallInteger): return pa.Int
    if isinstance(sa_type, BigInteger): return pa.Int64
    if isinstance(sa_type, Float): return pa.Float
    if isinstance(sa_type, DOUBLE_PRECISION): return pa.Float64
    if isinstance(sa_type, Numeric): return pa.Float
    if isinstance(sa_type, DECIMAL): return pa.Decimal
    if isinstance(sa_type, Boolean): return pa.Bool
    if isinstance(sa_type, DateTime): return pa.DateTime
    if isinstance(sa_type, Date): return pa.Date
    if isinstance(sa_type, Time): return pa.Time
    if isinstance(sa_type, LargeBinary): return pa.String
    if isinstance(sa_type, Uuid): return pa.String
    if isinstance(sa_type, JSON): return pa.Object
    if isinstance(sa_type, Enum): return pa.Category

    raise TypeError(f"Could not map SQLAlchemy type to Pandera: {sa_type}")


def Field(  # This is datantic.Field
    *,
    dtype: Optional[Any] = None,
    checks: Optional[List[Check]] = None,
    nullable: bool = False,
    unique: bool = False,
    coerce: bool = False,
    regex: bool = False,
    gt: Optional[Any] = None,
    ge: Optional[Any] = None,
    lt: Optional[Any] = None,
    le: Optional[Any] = None,
    isin: Optional[List[Any]] = None,
    notin: Optional[List[Any]] = None,
    multiple_of: Optional[Any] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Adds Pandera validation to standard Pydantic `BaseModel` fields."""

    if dtype is not None:  # Runtime check for dtype
        try:
            # First, try mapping SQLAlchemy types
            pandera_dtype = _sqlmodel_type_to_pandera_dtype(dtype)
        except TypeError:
            # If that fails, try creating a Pandera DataType directly
            try:
                if isinstance(dtype, str):
                    pandera_dtype = pa.DataType(dtype) # Correct: use pa.DataType
                else:
                    pandera_dtype = dtype # Assume it is already a pa.DataType Object
            except Exception as e:  # Catch a broader range of exceptions here
                raise TypeError(
                    f"Invalid dtype '{dtype}'. Must be a valid Pandera dtype or SQLAlchemy type."
                ) from e
        else:
            # if _sqlmodel_type_to_pandera_dtype worked
            dtype = pandera_dtype

    metadata = {
        "pandera": {
            "dtype": dtype,
            "checks": checks or [],
            "nullable": nullable,
            "unique": unique,
            "coerce": coerce,
            "regex": regex,
            "gt": gt,
            "ge": ge,
            "lt": lt,
            "le": le,
            "isin": isin,
            "notin": notin,
            "multiple_of": multiple_of,
            "min_length": min_length,
            "max_length": max_length,
            "pattern": pattern,
        }
    }
    return PydanticField(**kwargs, json_schema_extra=metadata)


class DataFrameField:
    """Specifies Pandera validation rules for `DataFrameModel` fields."""

    def __init__(
        self,
        *,
        dtype: Optional[Any] = None,
        checks: Optional[List[Check]] = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        regex: bool = False,
        gt: Optional[Any] = None,
        ge: Optional[Any] = None,
        lt: Optional[Any] = None,
        le: Optional[Any] = None,
        isin: Optional[List[Any]] = None,
        notin: Optional[List[Any]] = None,
        multiple_of: Optional[Any] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        **kwargs: Any,
    ):

        if dtype is not None:  # Runtime check for dtype
            try:
                # First, try mapping SQLAlchemy types
                pandera_dtype = _sqlmodel_type_to_pandera_dtype(dtype)
            except TypeError:
                # If that fails, try creating a Pandera DataType directly
                try:
                    if isinstance(dtype, str):
                        pandera_dtype = pa.DataType(dtype)  # Correct.
                    else:
                        pandera_dtype = dtype # Assume already pa.DataType
                except Exception as e:  # More general exception catch
                    raise TypeError(
                        f"Invalid dtype '{dtype}'. Must be a valid Pandera dtype or SQLAlchemy type."
                    ) from e
            else:
                # if _sqlmodel_type_to_pandera_dtype worked:
                dtype = pandera_dtype

        self.dtype = dtype
        self.checks = checks or []
        self.nullable = nullable
        self.unique = unique
        self.coerce = coerce
        self.regex = regex
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le
        self.isin = isin
        self.notin = notin
        self.multiple_of = multiple_of
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.extra = kwargs  # Store extra metadata

    def to_pandera_field(self) -> Dict[str, Any]:
        """Returns a dictionary representation (for internal use)."""
        return self.__dict__