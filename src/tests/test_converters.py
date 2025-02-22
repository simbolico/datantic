# tests/test_converters.py
import pytest
import pandas as pd
# from src.datantic.converters import to_polars, to_pandas # Removed

def test_to_polars_conversion(sample_pandas_dataframe):
    """Tests that the to_polars function correctly converts a Pandas DataFrame to a Polars DataFrame."""
    try:
        import polars as pl
        from src.datantic.converters import to_polars # Moved here

        polars_df = to_polars(sample_pandas_dataframe)
        assert isinstance(polars_df, pl.DataFrame)
        assert polars_df.shape == sample_pandas_dataframe.shape
    except ImportError:
        pytest.skip("Polars is not installed.")

def test_to_pandas_conversion():
    """Tests that the to_pandas function correctly converts a Polars DataFrame to a Pandas DataFrame."""
    pd = pytest.importorskip("pandas") # Use importorskip
    pl = pytest.importorskip("polars") # Both are required
    from src.datantic.converters import to_pandas # Moved here

    polars_df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    pandas_df = to_pandas(polars_df)
    assert isinstance(pandas_df, pd.DataFrame)
    assert pandas_df.shape == polars_df.shape


def test_to_polars_raises_import_error_when_polars_not_installed():
    """Tests that to_polars raises a RuntimeError when polars is not installed"""
    try:
        import polars
        pytest.skip("Polars is installed, cannot test ImportError")
    except ImportError:
        from src.datantic.converters import to_polars  # Moved here
        with pytest.raises(RuntimeError, match="Polars is not installed"):
            to_polars("DummyPandasDF")

def test_to_pandas_raises_import_error_when_pandas_not_installed():
    """Tests that to_pandas raises a RuntimeError when pandas is not installed"""
    try:
        import pandas
        pytest.skip("Pandas is installed, cannot test ImportError")
    except ImportError:
        from src.datantic.converters import to_pandas # Moved here
        try:
            import polars as pl
            dummy_polars_df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            with pytest.raises(RuntimeError, match="Pandas is not installed"):
                to_pandas(dummy_polars_df)
        except ImportError:
            with pytest.raises(RuntimeError, match="Pandas is not installed"):
                to_pandas("DummyPolarsDF") # Dummy value as Polars might be missing as well