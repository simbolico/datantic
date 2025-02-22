# tests/test_fields.py
import pytest
from src.datantic.fields import DataFrameField, Field
import pandera as pa

def test_dataframefield_creation():
    """Test the creation of a DataFrameField."""
    field = DataFrameField(dtype=pa.Int64, nullable=True, unique=True)
    assert field.dtype == pa.Int64
    assert field.nullable is True
    assert field.unique is True

def test_field_creation():
    """Test the creation of a Field for Pydantic models."""
    field = Field(dtype=pa.String, min_length=5)
    assert field.json_schema_extra is not None
    assert "pandera" in field.json_schema_extra
    assert field.json_schema_extra["pandera"]["dtype"] == pa.String
    assert field.json_schema_extra["pandera"]["min_length"] == 5

def test_field_invalid_dtype():
    """Tests that an invalid dtype for both Field and DataFrameField throws an error"""
    with pytest.raises(TypeError, match="Invalid dtype 'NotAValidDType'. Must be a valid Pandera dtype or SQLAlchemy type."):
        DataFrameField(dtype = "NotAValidDType")

    with pytest.raises(TypeError, match="Invalid dtype 'NotAValidDType'. Must be a valid Pandera dtype or SQLAlchemy type."):
        Field(dtype = "NotAValidDType")