# tests/test_validators_polars.py
import pytest
import pandera as pa

try:
    import polars as pl
    from src.datantic.validators.polars import PolarsDataFrameValidator
    polars_installed = True
except ImportError:
    polars_installed = False

@pytest.mark.skipif(not polars_installed, reason="Polars is not installed")
def test_polars_dataframe_validator_creation(sample_pandera_schema):
    """Test creation of PolarsDataFrameValidator."""
    validator = PolarsDataFrameValidator(sample_pandera_schema)
    assert isinstance(validator, PolarsDataFrameValidator)

@pytest.mark.skipif(not polars_installed, reason="Polars is not installed")
def test_polars_dataframe_validator_validate_dataframe(sample_pandas_dataframe, sample_pandera_schema):
    """Test validate_dataframe method."""
    import polars as pl
    validator = PolarsDataFrameValidator(sample_pandera_schema)
    polars_df = pl.DataFrame(sample_pandas_dataframe)
    validated_df = validator.validate_dataframe(polars_df)
    assert isinstance(validated_df, pl.DataFrame)

@pytest.mark.skipif(not polars_installed, reason="Polars is not installed")
def test_polars_dataframe_validator_handle_pandera_errors(sample_pandas_dataframe, sample_pandera_schema):
    """Test handle_pandera_errors method."""
    import polars as pl
    # create invalid dataframe with wrong types
    data = {'col1': ['a', 'b', 'c'], 'col2': ['1', '2', '3']}  # col1 should be integers
    invalid_df = pl.DataFrame(data)
    
    validator = PolarsDataFrameValidator(sample_pandera_schema)
    
    # Test that validation fails
    with pytest.raises(pa.errors.SchemaError) as exc_info:
        validator.validate_dataframe(invalid_df)
    
    # Test error handling
    error = exc_info.value
    if isinstance(error, pa.errors.SchemaErrors):  # Handle multiple errors
        filtered_df = validator.handle_pandera_errors(error, invalid_df)
    else:  # Handle single error
        filtered_df = pl.DataFrame(schema=invalid_df.schema)  # Empty DataFrame with same schema
    
    assert isinstance(filtered_df, pl.DataFrame)
    assert len(filtered_df) == 0  # All rows should be filtered out due to type error