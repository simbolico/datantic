# src/datantic/nesting.py
import logging
import inspect
from typing import Type, Optional, List, Dict, Any, Union, get_origin, get_args

from pydantic import BaseModel, ValidationError
import pandera as pa
from pandera.errors import SchemaErrors

from .types import is_polars_dataframe
from .converters import to_pandas
import pandas as pd

logger = logging.getLogger(__name__)


class ModelColumns(BaseModel):
    name: str
    id_column: Optional[str]
    base_columns: List[str]
    list_columns: List["ModelColumns"]
    child_columns: List["ModelColumns"]

ModelColumns.model_config["arbitrary_types_allowed"] = True  # remove on pydantic 2.6+
ModelColumns.update_forward_refs()


def get_model_columns(model: Type[BaseModel], id_column_map: Optional[Dict[str, str]] = None, name: Optional[str] = None) -> ModelColumns:
    if not inspect.isclass(model) or not issubclass(model, BaseModel):
        raise TypeError(f"{model} is not a BaseModel")
    id_column_map = id_column_map or {}
    name = name or model.__name__
    id_column = id_column_map.get(name) or id_column_map.get(model.__name__)  # Use model name and provided name
    base_cols, list_cols, child_cols = [], [], []
    for fname, field in model.model_fields.items():
        ftype = field.annotation
        if get_origin(ftype) is list:
            arg = get_args(ftype)[0]
            if isinstance(arg, type) and issubclass(arg, BaseModel):  # Ensure it's a BaseModel
              list_cols.append(get_model_columns(arg, id_column_map, fname))
            else:
              base_cols.append(fname)
        elif isinstance(ftype, type) and issubclass(ftype, BaseModel):
            child_cols.append(get_model_columns(ftype, id_column_map, fname))
        else:
            base_cols.append(fname)
    return ModelColumns(name=name, id_column=id_column, base_columns=base_cols, list_columns=list_cols, child_columns=child_cols)


def serialize_dataframe(data: Any, cols: ModelColumns, model: Type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Serializes a Pandas or Polars DataFrame into a list of dictionaries
    with proper nesting structure based on the model columns.
    """
    # Convert to pandas if needed
    df = to_pandas(data)
    
    # Group by ID column if present
    if cols.id_column:
        groups = df.groupby(cols.id_column)
    else:
        groups = [(None, df)]
    
    result = []
    for group_id, group_df in groups:
        item = {}
        
        # Add base columns (take first row's values)
        if group_id is not None:
            item[cols.id_column] = group_id
        first_row = group_df.iloc[0]
        for col in cols.base_columns:
            if col in first_row.index:
                # Check if this is a list field in the model
                field_type = model.model_fields[col].annotation
                if get_origin(field_type) is list:
                    # For list fields, collect all values in the group
                    item[col] = group_df[col].tolist()
                else:
                    item[col] = first_row[col]
        
        # Handle list columns (nested BaseModel lists)
        for list_col in cols.list_columns:
            nested_items = []
            for _, row in group_df.iterrows():
                nested_item = {}
                for col in list_col.base_columns:
                    if col in row.index:
                        nested_item[col] = row[col]
                if nested_item:
                    nested_items.append(nested_item)
            if nested_items:
                item[list_col.name] = nested_items
        
        # Handle child columns (nested single BaseModel)
        for child_col in cols.child_columns:
            if all(col in group_df.columns for col in child_col.base_columns):
                child_item = {}
                # Take first row's values for child model
                for col in child_col.base_columns:
                    if col in first_row.index:
                        child_item[col] = first_row[col]
                if child_item:
                    item[child_col.name] = child_item
        
        if item:
            result.append(item)
    
    return result


def to_nested_pydantic(data: Any, model: Type[BaseModel], id_map: Dict[str, str]) -> List[BaseModel]:
    """
    Convert a DataFrame to a list of nested Pydantic models.

    Args:
        data: The DataFrame to convert (Pandas or Polars)
        model: The Pydantic model class to convert to
        id_map: A dictionary mapping model names to their ID columns

    Returns:
        A list of nested Pydantic models

    Raises:
        pa.errors.SchemaError: If a single validation error occurs
        pa.errors.SchemaErrors: If multiple validation errors occur
    """
    # Get model columns structure
    cols = get_model_columns(model, id_map)
    
    # Serialize DataFrame to nested structure
    serialized_data = serialize_dataframe(data, cols, model)
    
    try:
        # Convert to Pydantic models
        result = []
        validation_errors = []
        for idx, item in enumerate(serialized_data):
            try:
                result.append(model.model_validate(item))
            except ValidationError as e:
                # Collect validation errors with index information
                for err in e.errors():
                    # Create a schema for the error
                    error_schema = pa.DataFrameSchema(name=err["loc"][0] if err["loc"] else "validation")
                    validation_errors.append(
                        pa.errors.SchemaError(
                            schema=error_schema,
                            data=data,
                            message=err["msg"],
                            check=None,
                            check_output=None,
                            reason_code=pa.errors.SchemaErrorReason.WRONG_DATATYPE,
                            column_name=err["loc"][0] if err["loc"] else None,
                            failure_cases=pd.DataFrame({
                                "column": [err["loc"][0]],
                                "failure_case": [err["input"]],
                                "index": [idx]
                            })
                        )
                    )
        
        if validation_errors:
            if len(validation_errors) == 1:
                raise validation_errors[0]
            raise pa.errors.SchemaErrors(
                schema=pa.DataFrameSchema(),
                schema_errors=validation_errors,
                data=data
            )
            
        return result
    except ValidationError as e:
        # Handle any other validation errors at the top level
        errors = []
        for err in e.errors():
            error_schema = pa.DataFrameSchema(name=err["loc"][0] if err["loc"] else "validation")
            errors.append(
                pa.errors.SchemaError(
                    schema=error_schema,
                    data=data,
                    message=err["msg"],
                    check=None,
                    check_output=None,
                    reason_code=pa.errors.SchemaErrorReason.WRONG_DATATYPE,
                    column_name=err["loc"][0] if err["loc"] else None,
                    failure_cases=pd.DataFrame({
                        "column": [err["loc"][0]],
                        "failure_case": [err["input"]],
                        "index": [0]  # Default to 0 for top-level errors
                    })
                )
            )
        if len(errors) == 1:
            raise errors[0]
        raise pa.errors.SchemaErrors(schema=pa.DataFrameSchema(), schema_errors=errors, data=data)