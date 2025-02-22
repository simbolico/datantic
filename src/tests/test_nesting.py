# src/tests/test_nesting.py
import pytest
from typing import List
from pydantic import BaseModel
import pandas as pd
import polars as pl
import inspect
import pandera as pa

from src.datantic.plugins.pandas import DataFrameAccessor as PandasAccessor
from src.datantic.plugins.polars import PolarsDataFrameAccessor
from src.datantic.nesting import get_model_columns, serialize_dataframe

# Register accessors (do this only once, ideally in a conftest.py)
pd.api.extensions.register_dataframe_accessor("datantic")(PandasAccessor)
pl.api.register_dataframe_namespace("datantic")(PolarsDataFrameAccessor)

class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    id: int
    name: str
    addresses: List[Address]

class ItemList(BaseModel):
    id: int
    values: List[str]

class NonBaseModel:
    def __init__(self, value: str):
        self.value = value

class Container(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    id: int
    nested: List[NonBaseModel]

@pytest.fixture
def pandas_df():
    data = {
        "id": [1, 1, 2],
        "name": ["Alice", "Alice", "Bob"],
        "street": ["1st St", "2nd St", "3rd St"],
        "city": ["Wonderland", "Wonderland", "Builderland"],
    }
    return pd.DataFrame(data)

@pytest.fixture
def polars_df(pandas_df):
    return pl.from_pandas(pandas_df)

@pytest.fixture
def pandas_item_list_df():
    data = {
        "id": [1, 1, 2, 2],
        "values": ["A", "B", "C", "D"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def pandas_non_basemodel_df():
    data = {
        "id": [1, 1],
        "value": ["A", "B"]
    }
    return pd.DataFrame(data)

def test_to_nested_pydantic_pandas(pandas_df):
    id_map = {"User": "id"}
    result = pandas_df.datantic.to_nested_pydantic(User, id_map)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], User)
    assert result[0].name == "Alice"
    assert len(result[0].addresses) == 2
    assert result[0].addresses[0].street == "1st St"
    assert result[1].name == "Bob"

def test_to_nested_pydantic_polars(polars_df):
    id_map = {"User": "id"}
    result = polars_df.datantic.to_nested_pydantic(User, id_map)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], User)
    assert result[0].name == "Alice"
    assert len(result[0].addresses) == 2
    assert result[0].addresses[0].street == "1st St"
    assert result[1].name == "Bob"

def test_to_nested_pydantic_missing_id_map(pandas_df):
    with pytest.raises(ValueError, match="The `id_column_map` argument is required"):
        pandas_df.datantic.to_nested_pydantic(User)

def test_to_nested_pydantic_invalid_model(pandas_df):
    id_map = {"User": "id"}
    with pytest.raises(TypeError, match="The `model` argument must be a Pydantic BaseModel class."):
        pandas_df.datantic.to_nested_pydantic(str, id_map)  # type: ignore

def test_to_nested_pydantic_with_validation_pandas(pandas_df):
    from src.datantic import DataFrameModel, Field
    class UserSchema(DataFrameModel):
        id: int = Field()
        name: str = Field()
        street: str = Field()
        city: str = Field()

    id_map = {"User": "id"}
    # Valid data should pass
    result = pandas_df.datantic.to_nested_pydantic(User, id_map, validate=True) # No error

    # Introduce an error
    invalid_df = pandas_df.copy()
    invalid_df.loc[0, "id"] = "not an int"

    with pytest.raises(pa.errors.SchemaError):
        invalid_df.datantic.to_nested_pydantic(User, id_map, validate=True)

def test_to_nested_pydantic_list_of_strings(pandas_item_list_df):
    id_map = {"ItemList": "id"}
    result = pandas_item_list_df.datantic.to_nested_pydantic(ItemList, id_map)
    assert isinstance(result, list)
    assert len(result) == 2  # Two unique IDs
    assert isinstance(result[0], ItemList)
    assert result[0].id == 1
    assert result[0].values == ["A", "B"]  # Correctly groups the strings

def test_to_nested_pydantic_non_basemodel_nested(pandas_non_basemodel_df):
    id_map = {"Container": "id"}
    with pytest.raises(TypeError, match="The `model` argument must be a Pydantic BaseModel class."):
        pandas_non_basemodel_df.datantic.to_nested_pydantic(Container, id_map)