# src/datantic/model.py (Corrected Again and Again)
import inspect
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

import pandera as pa
from pandera.api.checks import Check
from pandera.typing import DataFrame

try:
    import pandera.polars as papl  # noqa F401

    _PANDERA_POLARS_INSTALLED = True
except ImportError:
    _PANDERA_POLARS_INSTALLED = False


from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from sqlmodel import SQLModel
from pydantic.fields import FieldInfo

from .fields import DataFrameField, Field
from .types import Optional, is_polars_dataframe
from .utils import create_checks, convert_sqlmodel_field

try:
    import pandas as pd

    _PANDAS_INSTALLED = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS_INSTALLED = False


logger = logging.getLogger(__name__)


def pandera_check(check_fn: Callable) -> Callable:
    """Decorator for registering class-level Pandera checks in `DataFrameModel`."""

    # The wrapper function is NOT a classmethod.  It just marks the
    # function.  The class is passed during schema creation.
    def wrapper(*args, **kwargs):
        return check_fn(*args, **kwargs)

    wrapper.__pandera_check__ = True
    return wrapper


class DataFrameModel(BaseModel):
    """Defines DataFrame schemas using Pydantic models."""

    __pandera_schema__: ClassVar[Optional[pa.DataFrameSchema]] = None
    __pandera_checks_cache__: ClassVar[
        Optional[List[Callable[..., pa.Check]]]
    ] = None  # CORRECTED type
    _polars_schema: ClassVar[bool] = False

    class Config:
        """Configuration for the generated Pandera schema."""

        index: Optional[DataFrameField] = None
        coerce: bool = False

    @classmethod
    def to_pandera_schema(cls) -> pa.DataFrameSchema:
        """Converts the Pydantic model to a Pandera `DataFrameSchema`."""
        if cls.__pandera_schema__ is not None:
            return cls.__pandera_schema__

        schema_dict: Dict[str, Any] = {"columns": {}}

        # --- Caching Logic --- CORRECTED: Cache check functions, not methods
        if cls.__pandera_checks_cache__ is None:
            cls.__pandera_checks_cache__ = []
            for name, obj in inspect.getmembers(cls):
                if callable(obj) and getattr(
                    obj, "__pandera_check__", False
                ):  # Check if callable
                    cls.__pandera_checks_cache__.append(obj)

        # CORRECTED: Apply checks correctly, handling the DataFrame argument.
        checks: List[pa.Check] = []
        if cls.__pandera_checks_cache__:
            for check_fn in cls.__pandera_checks_cache__:
                checks.append(pa.Check(lambda df, check_fn=check_fn: check_fn(cls, df)))


        pandera_fields = {}
        for field_name, field_info in cls.model_fields.items():
            if isinstance(field_info.default, DataFrameField):
                pandera_field_info = field_info.default
                field_checks = create_checks(pandera_field_info)
                pandera_fields[field_name] = pa.Column(
                    dtype=pandera_field_info.dtype,
                    checks=field_checks,
                    nullable=pandera_field_info.nullable,
                    unique=pandera_field_info.unique,
                    coerce=pandera_field_info.coerce,
                    regex=pandera_field_info.regex,
                )

            elif "pandera" in (field_info.json_schema_extra or {}):
                pandera_settings = field_info.json_schema_extra["pandera"]
                field_checks = create_checks(pandera_settings)
                pandera_fields[field_name] = pa.Column(
                    dtype=pandera_settings["dtype"],
                    checks=field_checks,
                    nullable=pandera_settings["nullable"],
                    coerce=pandera_settings["coerce"],
                    unique=pandera_settings.get("unique", False),
                    regex=pandera_settings.get("regex", False),
                )

            else:
                pandera_fields[field_name] = _map_field_to_pandera_column(
                    field_name, field_info
                )

        schema_dict["checks"] = checks
        schema_dict["columns"] = pandera_fields
        cls.__pandera_schema__ = pa.DataFrameSchema(**schema_dict)
        return cls.__pandera_schema__

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Defines how Pydantic should treat this class (Pydantic V2 compatibility)."""

        def validate_pydanticdataframe(value: Any, handler: Any) -> Any:
            if inspect.isclass(value) and issubclass(value, cls):
                return value  # If it's the class itself, return it
            return cls.to_pandera_schema().validate(value)  # Validate data

        return core_schema.no_info_after_validator_function(
            validate_pydanticdataframe, core_schema.any_schema()
        )

    @classmethod
    @property
    def schema(cls) -> pa.DataFrameSchema:
        """Returns the Pandera schema (alias for `to_pandera_schema`)."""
        return cls.to_pandera_schema()

    @classmethod
    def from_sqlmodel(
        cls, sqlmodel_class: Type[SQLModel], *, name: Optional[str] = None, type_mapping: Optional[Dict[Any, Any]] = None
    ) -> Type["DataFrameModel"]:
        """Creates a DataFrameModel from an SQLModel class."""
        if type_mapping is not None and not isinstance(type_mapping, dict):
            raise TypeError(f"type_mapping must be a dict, got {type(type_mapping)}")

        annotations = {}
        namespace = {
            "__annotations__": annotations,
            "model_config": {"arbitrary_types_allowed": True},
        }

        for field_name, field_info in sqlmodel_class.model_fields.items():
            final_annotation, df_field = convert_sqlmodel_field(
                field_name, field_info, custom_mapping=type_mapping
            )
            annotations[field_name] = final_annotation
            namespace[field_name] = df_field

        # Create the new DataFrameModel class dynamically
        new_model = type(
            name or f"{sqlmodel_class.__name__}DataFrame",  # Name for new class
            (DataFrameModel,),  # Inherit from DataFrameModel
            namespace,
        )
        return new_model


def _map_field_to_pandera_column(field_name: str, field_info: FieldInfo):
    """Maps a Pydantic field to a Pandera column."""
    # Extract field metadata
    metadata = field_info.json_schema_extra or {}
    
    # Handle nullable fields
    nullable = False
    field_type = field_info.annotation
    
    # Check for Optional/Union types
    origin = get_origin(field_info.annotation)
    if origin in (Optional, Union):
        args = get_args(field_info.annotation)
        if args:
            field_type = args[0]  # Get the first type argument
            # Check if Union contains None or type(None)
            if type(None) in args or None in args:
                nullable = True
    
    # Map Python types to Pandera types
    if field_type == int:
        dtype = pa.Int64
    elif field_type == str:
        dtype = pa.String
    elif field_type == float:
        dtype = pa.Float64
    elif field_type == bool:
        dtype = pa.Boolean
    elif field_type == DataFrameField:  # Handle DataFrameField type directly
        dtype = field_info.default.dtype
        nullable = field_info.default.nullable
    else:
        dtype = pa.Object
        logger.warning(f"Field '{field_name}' has unhandled type {field_type}, using pa.Object")

    # Create Pandera column
    column = pa.Column(
        dtype=dtype,
        nullable=nullable,
        **{k: v for k, v in metadata.items() if k not in ['dtype', 'nullable']}
    )
    
    return column