# src/tests/test_model.py (Corrected)
import pytest
import pandas as pd
import pandera as pa
from src.datantic.model import DataFrameModel, pandera_check
from src.datantic.fields import DataFrameField
from typing import ClassVar, Optional  # Import Optional
from pydantic import BaseModel, Field # Import pydantic BaseModel and Field
from src.datantic import Validator  # Add import at top
import warnings

def test_dataframe_model_to_pandera_schema_basic():
    """Test basic DataFrameModel to Pandera schema conversion."""

    class MySchema(DataFrameModel):
        col1: int = DataFrameField(dtype=pa.Int64)
        col2: str = DataFrameField(dtype=pa.String)

    schema = MySchema.to_pandera_schema()
    assert isinstance(schema, pa.DataFrameSchema)
    assert "col1" in schema.columns
    assert "col2" in schema.columns
    assert isinstance(schema.columns["col1"].dtype, pa.dtypes.Int64)
    assert isinstance(schema.columns["col2"].dtype, pa.dtypes.String)


def test_dataframe_model_with_checks():
    """Test DataFrameModel with Pandera checks."""

    class MySchema(DataFrameModel):
        col1: int = DataFrameField(gt=0)
        col2: str = DataFrameField(min_length=2)

    schema = MySchema.to_pandera_schema()
    assert len(schema.columns["col1"].checks) == 1
    assert len(schema.columns["col2"].checks) == 1

def test_dataframe_model_with_optional():
    """Test DataFrameModel with Optional fields."""
    from typing import Optional
    from src.datantic import Optional as DatanticOptional

    class MySchema(DataFrameModel):
        col1: DatanticOptional[int] = DataFrameField(dtype=pa.Float64, nullable=True)
        col2: Optional[str] = DataFrameField(dtype=pa.String, nullable=True)

    schema = MySchema.to_pandera_schema()
    assert schema.columns["col1"].nullable is True
    assert schema.columns["col2"].nullable is True

    # Test with actual data
    data = {
        'col1': [1, None, 3],
        'col2': ['a', None, 'c']
    }
    df = pd.DataFrame(data)
    validated_df = schema.validate(df)
    assert validated_df is not None
    assert validated_df['col1'].isna().sum() == 1
    assert validated_df['col2'].isna().sum() == 1


def test_pandera_check_decorator():
    """Tests the @pandera_check decorator."""

    class MySchema(DataFrameModel):
        col1: int = DataFrameField(dtype=pa.Int)

        @pandera_check
        def check_col1_sum(cls, df):
            return pa.Check(lambda df: df["col1"].sum() > 0)

    schema = MySchema.to_pandera_schema()
    assert len(schema.checks) == 1
    assert isinstance(schema.checks[0], pa.Check)


def test_dataframe_model_from_sqlmodel():
    """Tests the from_sqlmodel method."""
    try:
        from sqlmodel import SQLModel, Field

        class User(SQLModel, table=True):
            id: int = Field(primary_key=True)
            name: str = Field(index=True)
            age: Optional[int] = Field(default=None, nullable=True)  # Use Optional

        DFUser = DataFrameModel.from_sqlmodel(User)
        assert issubclass(DFUser, DataFrameModel)
        schema = DFUser.to_pandera_schema()
        assert "id" in schema.columns
        assert "name" in schema.columns
        assert "age" in schema.columns
    except ImportError:
        pytest.skip("SQLModel is not installed.")

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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Validator(UserRow)
        assert len(w) > 0
        assert any(issubclass(warning.category, UserWarning) for warning in w)