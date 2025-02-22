# src/datantic/plugins/polars_plugin.py
import logging
from typing import Any, Iterable, Hashable, Optional, Tuple, Type, Union, List

import pandera as pa  # We import pandera
from pydantic import BaseModel, ValidationError

# We DO NOT import polars at the top level.
from datantic.types import is_polars_dataframe
from datantic.model import DataFrameModel
from datantic.validator import Validator

logger = logging.getLogger(__name__)


class PolarsDataFrameAccessor:
    """Polars DataFrame accessor for datantic validation."""

    def __init__(self, polars_obj: Any):  # Accept Any
        # Delay the Polars import and type check until __init__
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
        **kwargs: Any,
    ) -> Any:  # Return Any
        """Validate the Polars DataFrame and return the validated DataFrame."""
        validator = Validator(schema)
        # Always raise exceptions in the accessor
        validated_df = validator.validate(
            self._obj,  # Pass Polars DataFrame directly
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
            errors="raise",
            **kwargs,
        )
        return validated_df

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
            # Pass all arguments to validate
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
    ) -> Iterable[Tuple[Hashable, Any]]:  # Return Any Series
        """Iterates over valid rows of a Polars DataFrame, yielding (index, Series)."""
        import polars as pl  # Local import

        validator = Validator(schema)
        for i, row_data in validator.iterate(self._obj, verbose=verbose, **kwargs):
            yield i, pl.Series(name=str(i), values=list(row_data.values()))  # Convert to Polars Series

    def iterschemas(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        verbose: bool = True,
        **kwargs: Any,
    ) -> Iterable[Tuple[Hashable, Union[dict, BaseModel]]]:
        """Iterates, yielding (index, schema_instance) tuples."""
        validator = Validator(schema)
        # Cannot use validator.iterate with polars directly if it is a BaseModel
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            for i, row in enumerate(self._obj.to_dicts()):
                try:
                    validated_model = schema.model_validate(row)
                    yield i, validated_model
                except ValidationError as e:
                    if verbose:
                        logger.info(f"Validation error at index {i}, skipping: {e}.")
                    continue
        else:  # Use validator.iterate if it's a DataFrameModel or DataFrameSchema
            yield from validator.iterate(self._obj, verbose=verbose, **kwargs)

    def to_pydantic(self, model_class: Type[BaseModel]) -> List[BaseModel]:
        """Converts the Polars DataFrame to a list of Pydantic model instances."""
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                "The `to_pydantic` method expects a Pydantic BaseModel class."
            )
        self.validate(model_class)  # Validate, raising exceptions
        return [model_class(**row) for row in self._obj.to_dicts()]  # Use to_dicts()