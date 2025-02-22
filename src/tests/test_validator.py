# src/tests/test_validator.py (Corrected)
import pytest
import pandas as pd
import pandera as pa
from pydantic import BaseModel, ValidationError, Field
from src.datantic.validator import Validator
from src.datantic.model import DataFrameModel
from src.datantic.fields import DataFrameField
from typing import Optional
import logging  # Import logging


def test_validator_with_dataframe_model(sample_pandas_dataframe):
    """Test Validator with a DataFrameModel."""

    class MySchema(DataFrameModel):
        col1: int = DataFrameField(dtype=pa.Int64)
        col2: str = DataFrameField(dtype=pa.String)

    validator = Validator(MySchema)
    validated_df = validator.validate(sample_pandas_dataframe)
    assert isinstance(validated_df, pd.DataFrame)


def test_validator_with_pandera_schema(
    sample_pandas_dataframe, sample_pandera_schema
):
    """Test Validator with a Pandera DataFrameSchema."""
    validator = Validator(sample_pandera_schema)
    validated_df = validator.validate(sample_pandas_dataframe)
    assert isinstance(validated_df, pd.DataFrame)


def test_validator_with_pydantic_model(sample_pandas_dataframe):
    """Test Validator with a Pydantic BaseModel for row-wise validation."""
    from src.datantic import Field

    class MyRow(BaseModel):
        col1: int = Field(ge=0)
        col2: str = Field(min_length=1)

    validator = Validator(MyRow)
    validated_df = validator.validate(sample_pandas_dataframe)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == len(sample_pandas_dataframe)


def test_validator_error_handling(sample_pandas_dataframe):
    """Test Validator with error handling (logging)."""
    # Use pydantic.Field here, and we're testing "raise" behavior
    class MyRow(BaseModel):
        col1: int = Field(gt=5)  # This will cause errors

    validator = Validator(MyRow)  # No error handler needed
    with pytest.raises(ValidationError):  # Expect a ValidationError
        validator.validate(sample_pandas_dataframe, errors="raise")


def test_validator_is_valid(sample_pandas_dataframe):
    """Tests the is_valid method."""

    class MySchema(DataFrameModel):
        col1: int = DataFrameField(dtype=pa.Int64)
        col2: str = DataFrameField(dtype=pa.String)

    validator = Validator(MySchema)
    assert validator.is_valid(sample_pandas_dataframe)


def test_validator_iterate(sample_pandas_dataframe):
    """Tests the iterate method."""

    class MySchema(DataFrameModel):
        col1: int = DataFrameField(dtype=pa.Int64)
        col2: str = DataFrameField(dtype=pa.String)

    validator = Validator(MySchema)
    for i, row in validator.iterate(sample_pandas_dataframe):
        assert isinstance(i, (int, str))  # Index can be int or str
        assert isinstance(row, dict)  # Expecting a dict for valid rows


def test_pydantic_field_warning():
    """Tests the logging warning if the Pydantic model has no datantic.Fields"""
    from pydantic import Field

    class UserRow(BaseModel):
        user_id: int = Field(ge=0)
        username: str = Field(min_length=3, max_length=20)
        email: str = Field(
            pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        )
        age: Optional[int] = Field(gt=0)

    # CORRECTED: Instantiate Validator *inside* the with block
    with pytest.warns(
        UserWarning,
        match=r"Field '.*' in Pydantic model '.*' does not have a datantic\.Field",
    ):
        Validator(UserRow)