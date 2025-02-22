# src/datantic/utils.py
import logging
from typing import Any, Dict, List, Tuple, Union, Optional

import pandera as pa
from pandera.api.checks import Check
from pydantic.fields import FieldInfo
from .fields import Field, DataFrameField
import re # Import for regex

try:
    import pandas as pd
    _PANDAS_INSTALLED = True
except ImportError:
    pd = None
    _PANDAS_INSTALLED = False

logger = logging.getLogger(__name__)


def create_checks(field_settings: Union["DataFrameField", Dict[str, Any]]) -> List[pa.Check]:
    """Creates a list of Pandera checks from field settings."""
    # ... (existing imports)
    checks: List[Check] = []

    if isinstance(field_settings, DataFrameField):
        settings = field_settings.__dict__
    elif isinstance(field_settings, dict):
        settings = field_settings
    else:
        raise TypeError(
            f"field_settings must be DataFrameField or dict, got {type(field_settings)}"
        )

    # Handle constraints using the correct keys directly from settings
    if "gt" in settings and settings["gt"] is not None:
        if not isinstance(settings["gt"], (int, float)):
            raise TypeError("gt constraint must be numeric")
        checks.append(pa.Check.greater_than(settings["gt"]))
    if "lt" in settings and settings["lt"] is not None:  # Add checks for other constraints
        if not isinstance(settings["lt"], (int, float)):
            raise TypeError("lt constraint must be numeric")
        checks.append(pa.Check.less_than(settings["lt"]))
    if "ge" in settings and settings["ge"] is not None:
        if not isinstance(settings["ge"], (int, float)):
            raise TypeError("ge constraint must be numeric")
        checks.append(pa.Check.greater_than_or_equal_to(settings["ge"]))
    if "le" in settings and settings["le"] is not None:
        if not isinstance(settings["le"], (int, float)):
            raise TypeError("le constraint must be numeric")
        checks.append(pa.Check.less_than_or_equal_to(settings["le"]))
    if "isin" in settings and settings["isin"] is not None:
        if not isinstance(settings["isin"], list) and not isinstance(settings["isin"], tuple):
            raise TypeError("isin constraint must be a list or tuple")
        checks.append(pa.Check.isin(settings["isin"]))
    if "notin" in settings and settings["notin"] is not None:
        if not isinstance(settings["notin"], list) and not isinstance(settings["notin"], tuple):
            raise TypeError("notin constraint must be a list or tuple")
        checks.append(pa.Check.notin(settings["notin"]))
    if "pattern" in settings and settings["pattern"] is not None:
        if not isinstance(settings["pattern"], str):
             raise TypeError("pattern constraint must be a string")
        try:
            re.compile(settings["pattern"]) # Check if it compiles
            checks.append(pa.Check.str_matches(settings["pattern"]))
        except re.error as e:
           raise ValueError(f"Invalid regex pattern: {e}") from e
    if "min_length" in settings and settings["min_length"] is not None:
        checks.append(pa.Check.str_length(min_value=settings["min_length"]))

    if "max_length" in settings and settings["max_length"] is not None:
        checks.append(pa.Check.str_length(max_value=settings["max_length"]))

    return checks


from .fields import _sqlmodel_type_to_pandera_dtype


def convert_sqlmodel_field(
    field_name: str, field_info: FieldInfo, custom_mapping: Optional[Dict[Any, Any]] = None
) -> Tuple[Any, Any]:  # Use Any for return type to avoid circular import
    from datantic.types import Optional  # our custom Optional type
    from datantic.fields import DataFrameField  # Import inside function

    pandera_field_args: Dict[str, Any] = {}

    # Map SQLAlchemy type to Pandera dtype
    sa_type = getattr(field_info, "sa_type", None)
    if sa_type is None:
        extras = getattr(field_info, "json_schema_extra", {}) or {}
        sa_column = extras.get("sa_column")
        if sa_column is not None:
            sa_type = sa_column.type

    try:
        dtype = _sqlmodel_type_to_pandera_dtype(sa_type, custom_mapping) if sa_type and str(sa_type) != "PydanticUndefined" else None
    except TypeError:
        # If type mapping fails, infer from annotation
        if field_info.annotation == str:
            dtype = pa.String
        elif field_info.annotation == int:
            dtype = pa.Int
        elif field_info.annotation == float:
            dtype = pa.Float
        elif field_info.annotation == bool:
            dtype = pa.Bool
        else:
            dtype = None

    df_field = DataFrameField(
        dtype=dtype,
        nullable=not field_info.is_required(),
        unique=getattr(field_info, "unique", False),
        coerce=True
    )

    annotation = field_info.annotation
    return annotation, df_field