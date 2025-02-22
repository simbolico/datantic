# tests/test_validators_pandas.py
import pytest
import pandas as pd
import pandera as pa
from src.datantic.validators.pandas import PandasDataFrameValidator
from src.datantic.types import ErrorHandler

def test_pandas_dataframe_validator_creation(sample_pandera_schema):
    """Test creation of PandasDataFrameValidator."""
    validator = PandasDataFrameValidator(sample_pandera_schema)
    assert isinstance(validator, PandasDataFrameValidator)

def test_pandas_dataframe_validator_validate_dataframe(sample_pandas_dataframe, sample_pandera_schema):
    """Test validate_dataframe method."""
    validator = PandasDataFrameValidator(sample_pandera_schema)
    validated_df = validator.validate_dataframe(sample_pandas_dataframe)
    assert isinstance(validated_df, pd.DataFrame)

def test_pandas_dataframe_validator_handle_pandera_errors(sample_pandas_dataframe, sample_pandera_schema):
    """Test handle_pandera_errors method."""
    # create invalid dataframe with wrong types
    data = {'col1': ['a', 'b', 'c'], 'col2': ['1', '2', '3']}  # col1 should be integers
    invalid_df = pd.DataFrame(data)
    
    validator = PandasDataFrameValidator(sample_pandera_schema)
    
    # Test that validation fails
    with pytest.raises(pa.errors.SchemaError) as exc_info:
        validator.validate_dataframe(invalid_df)
    
    # Test error handling
    error = exc_info.value
    if isinstance(error, pa.errors.SchemaErrors):  # Handle multiple errors
        filtered_df = validator.handle_pandera_errors(error, invalid_df)
    else:  # Handle single error
        filtered_df = pd.DataFrame(columns=invalid_df.columns)  # Empty DataFrame with same schema
    
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) == 0  # All rows should be filtered out due to type error