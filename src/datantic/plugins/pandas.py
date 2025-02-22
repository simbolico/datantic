# src/datantic/plugins/pandas_plugin.py
import logging
from typing import Any, Iterable, Hashable, Optional, Tuple, Type, Union, List

import pandas as pd  # No conditional import *here*
import pandera as pa
from pydantic import BaseModel, ValidationError

from datantic.model import DataFrameModel
from datantic.validator import Validator

logger = logging.getLogger(__name__)


class DataFrameAccessor:
    """Pandas DataFrame accessor for datantic validation."""

    def __init__(self, pandas_obj: pd.DataFrame):
        if not isinstance(pandas_obj, pd.DataFrame):
            raise TypeError("`pandas_obj` must be a Pandas DataFrame")
        self._obj = pandas_obj

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
    ) -> pd.DataFrame:
        """Validate the DataFrame and return the validated DataFrame."""
        validator = Validator(schema)
        # Always raise exceptions on the accessor
        return validator.validate(
            self._obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
            errors="raise",
            **kwargs,
        )

    def is_valid(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,  # Consistent signature
        **kwargs: Any,
    ) -> bool:
        """Check if the DataFrame is valid without raising an exception."""
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
        """Iterates over *valid* rows, yielding namedtuples."""
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
    ) -> Iterable[Tuple[Hashable, pd.Series]]:  # Corrected type hint
        """Iterates over valid rows, yielding (index, Series) pairs."""
        validator = Validator(schema)
        for i, row_data in validator.iterate(self._obj, verbose=verbose, **kwargs):
            yield i, pd.Series(row_data)

    def iterschemas(
        self,
        schema: Union[Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]],
        verbose: bool = True,
        **kwargs: Any,
    ) -> Iterable[Tuple[Hashable, Union[dict, BaseModel]]]:
        """Iterates, yielding (index, schema_instance) tuples."""
        validator = Validator(schema)
        yield from validator.iterate(self._obj, verbose=verbose, **kwargs)

    def to_pydantic(self, model_class: Type[BaseModel]) -> List[BaseModel]:
        """Converts the DataFrame to a list of Pydantic model instances."""
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                "The `to_pydantic` method expects a Pydantic BaseModel class as input."
            )
        # Validate, raising exceptions on failure
        self.validate(model_class)
        return [model_class(**row) for _, row in self._obj.iterrows()]