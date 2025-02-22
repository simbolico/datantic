# src/datantic/plugins/polars.py
import logging
from typing import Any, Iterable, Hashable, Optional, Tuple, Type, Union, List, Dict, Literal

import pandera as pa
from pydantic import BaseModel, ValidationError
import pandas as pd
import polars as pl

from src.datantic.types import is_polars_dataframe, ErrorHandler
from src.datantic.model import DataFrameModel
from src.datantic.validator import Validator
from src.datantic.nesting import get_model_columns, serialize_dataframe, to_nested_pydantic

logger = logging.getLogger(__name__)


class PolarsDataFrameAccessor:
    """Polars DataFrame accessor for datantic validation."""

    def __init__(self, polars_obj: Any):
        if not is_polars_dataframe(polars_obj):
            raise TypeError("`polars_obj` must be a Polars DataFrame")
        self._obj = polars_obj

    def validate(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        errors: Literal["raise", "log"] = "raise",
        error_handler: Optional[ErrorHandler] = None,
    ) -> pl.DataFrame:
        """Validates the DataFrame against a schema."""
        validator = Validator(schema)
        result = validator.validate(
            self._obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
            errors=errors,
            error_handler=error_handler,
        )
        # Convert back to Polars if needed
        if isinstance(result, pd.DataFrame):
            return pl.from_pandas(result)
        return result

    def is_valid(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Check if the Polars DataFrame is valid without raising."""
        try:
            self.validate(
                schema,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                **kwargs,
            )
            return True
        except (ValidationError, pa.errors.SchemaErrors):
            return False

    def itertuples(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        verbose: bool = True,
        **kwargs: Any,
    ) -> Iterable[Tuple[Hashable, ...]]:
        """Iterates over *valid* rows of a Polars DataFrame, yielding tuples."""
        validator = Validator(schema)
        for i, row_data in validator.iterate(self._obj, verbose=verbose, **kwargs):
            if isinstance(row_data, dict):
                yield (i, *row_data.values())
            else:
                yield (i, *row_data.model_dump().values())

    def iterrows(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        verbose: bool = True,
        **kwargs: Any,
    ) -> Iterable[Tuple[Hashable, Any]]:
        """Iterates over valid rows of a Polars DataFrame, yielding (index, Series)."""
        import polars as pl

        validator = Validator(schema)
        for i, row_data in validator.iterate(self._obj, verbose=verbose, **kwargs):
            yield i, pl.Series(name=str(i), values=list(row_data.values()))

    def iterschemas(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        verbose: bool = True,
        **kwargs: Any,
    ) -> Iterable[Tuple[Hashable, Union[dict, BaseModel]]]:
        """Iterates, yielding (index, schema_instance) tuples."""
        validator = Validator(schema)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            for i, row in enumerate(self._obj.to_dicts()):
                try:
                    validated_model = schema.model_validate(row)
                    yield i, validated_model
                except ValidationError as e:
                    if verbose:
                        logger.info(f"Validation error at index {i}, skipping: {e}.")
                    continue
        else:
            yield from validator.iterate(self._obj, verbose=verbose, **kwargs)

    def to_pydantic(self, model_class: Type[BaseModel]) -> List[BaseModel]:
        """Converts the Polars DataFrame to a list of Pydantic model instances."""
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                "The `to_pydantic` method expects a Pydantic BaseModel class."
            )
        self.validate(model_class)
        return [model_class(**row) for row in self._obj.to_dicts()]

    def to_nested_pydantic(
        self,
        model: Type[BaseModel],
        id_map: Optional[Dict[str, str]] = None,
        validate: bool = False,
        **kwargs,
    ) -> List[BaseModel]:
        """Convert a DataFrame to a list of nested Pydantic models.

        Args:
            model: The Pydantic model class to convert to.
            id_map: A dictionary mapping model names to their ID columns.
            validate: Whether to validate the data against the model.
            **kwargs: Additional arguments to pass to the validator.

        Returns:
            A list of nested Pydantic models.

        Raises:
            TypeError: If the model is not a Pydantic BaseModel class.
            ValueError: If id_map is required but not provided.
        """
        if not issubclass(model, BaseModel):
            raise TypeError("The `model` argument must be a Pydantic BaseModel class.")

        if id_map is None:
            raise ValueError("The `id_column_map` argument is required for nested conversion.")

        # Check if any field types are not BaseModel
        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            if hasattr(field_type, "__origin__") and field_type.__origin__ == list:
                inner_type = field_type.__args__[0]
                if isinstance(inner_type, type) and not issubclass(inner_type, (str, int, float, bool)):
                    if not issubclass(inner_type, BaseModel):
                        raise TypeError("The `model` argument must be a Pydantic BaseModel class.")
            elif isinstance(field_type, type) and not issubclass(field_type, (BaseModel, str, int, float, bool)):
                raise TypeError("The `model` argument must be a Pydantic BaseModel class.")

        # First convert to nested structure
        result = to_nested_pydantic(self._obj, model, id_map)

        # Then validate if requested
        if validate:
            try:
                validator = Validator(model)
                # Convert back to DataFrame for validation
                df = pd.DataFrame([item.model_dump() for item in result])
                validator.validate(df)
            except ValidationError as e:
                # Convert to Pandera error
                errors = []
                for err in e.errors():
                    # Create a schema for the error
                    error_schema = pa.DataFrameSchema(name=err["loc"][0] if err["loc"] else "validation")
                    errors.append(
                        pa.errors.SchemaError(
                            schema=error_schema,
                            data=self._obj,
                            message=err["msg"],
                            check=None,
                            check_output=None,
                            reason_code=pa.errors.SchemaErrorReason.WRONG_DATATYPE,
                            column_name=err["loc"][0] if err["loc"] else None,
                            failure_cases=pd.DataFrame({
                                "column": [err["loc"][0]],
                                "failure_case": [err["input"]],
                                "index": [0]
                            })
                        )
                    )
                if len(errors) == 1:
                    raise errors[0]
                raise pa.errors.SchemaErrors(schema=pa.DataFrameSchema(), schema_errors=errors, data=self._obj)

        return result