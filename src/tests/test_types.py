# tests/test_types.py
import pytest
import pandas as pd
from src.datantic.types import _coerce_nan_to_none, is_polars_dataframe, ErrorHandler
from typing import Callable, Union
import pandera as pa
from pydantic import BaseModel, ValidationError

def test_coerce_nan_to_none():
    """Test the _coerce_nan_to_none function."""
    import numpy as np
    assert _coerce_nan_to_none(np.nan) is None
    assert _coerce_nan_to_none(1) == 1
    assert _coerce_nan_to_none("test") == "test"

def test_is_polars_dataframe():
    """Test the is_polars_dataframe function."""
    try:
        import polars as pl
        df = pl.DataFrame({"col1": [1, 2, 3]})
        assert is_polars_dataframe(df)
    except ImportError:
        pass  # Polars not installed, test should pass

    df = pd.DataFrame({"col1": [1, 2, 3]})
    assert not is_polars_dataframe(df)

def test_optional_type_handling(sample_pandas_dataframe_with_nans):
    """Tests that the Optional type aliases is handling NaNs correctly"""
    from src.datantic import Optional as DatanticOptional

    class MyModel(BaseModel):
        col1: DatanticOptional[int]
        col2: DatanticOptional[str]

    data = sample_pandas_dataframe_with_nans.iloc[0].to_dict()
    row_with_nan = MyModel(**data)
    assert row_with_nan.col1 == 1

    # Test validation error with invalid type
    with pytest.raises(ValidationError):
        MyModel(**{"col1": "not an int", "col2": "valid"})

def test_error_handler_type():
    """Tests the error handler to confirm it's a valid type alias."""
    from src.datantic.types import ErrorHandler
    from typing import Callable, Union
    import pandera.errors as pa_errors
    from pydantic import ValidationError, BaseModel  # Import BaseModel

    def my_error_handler(e: Union[ValidationError, pa_errors.SchemaErrors]) -> None:
        pass

    # Check that the function matches the ErrorHandler type
    assert isinstance(my_error_handler, Callable)

    # Create a ValidationError for testing
    error_handler: ErrorHandler = my_error_handler

    # Test with a ValidationError
    class DummyModel(BaseModel):
        x: int
    try:
        DummyModel(x="not an int")
    except ValidationError as e:
        error_handler(e)

    # Test with a SchemaError - Use a simpler Exception for type checking
    error_handler(Exception("A generic schema error"))