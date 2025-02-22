# main.py
import logging
import pandas as pd
from pydantic import BaseModel, ValidationError
import pandera as pa
from pandera import Check
from typing import List

from datantic import (
    DataFrameModel,
    DataFrameField,
    Validator,
    Field,
    Optional,
    pandera_check,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example1_basic_dataframemodel():
    """Basic DataFrameModel validation with Pandas"""
    print("\n=== Example 1: Basic DataFrameModel Validation ===")

    class UserSchema(DataFrameModel):
        user_id: int = DataFrameField(ge=0)
        username: str = DataFrameField(min_length=3, max_length=20)
        email: str = DataFrameField(pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
        age: Optional[int] = DataFrameField(gt=0, nullable=True)

    # Valid data
    valid_data = {
        "user_id": [1, 2, 3],
        "username": ["alice123", "bob_456", "charlie789"],
        "email": ["alice@example.com", "bob@domain.org", "charlie@test.net"],
        "age": [25, 30, None]
    }

    df = pd.DataFrame(valid_data)
    validator = Validator(UserSchema)
    validated_df = validator.validate(df)
    print("✅ Valid data successfully validated!")
    print(validated_df.head())

def example2_rowwise_pydantic():
    """Row-wise validation with Pydantic BaseModel"""
    print("\n=== Example 2: Row-wise Pydantic Validation ===")

    class UserRow(BaseModel):
        user_id: int = Field(ge=0)
        username: str = Field(min_length=3, max_length=20)
        email: str = Field(pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
        age: Optional[int] = Field(gt=0)

    # Data with multiple errors
    invalid_data = {
        "user_id": [1, -2, 3],
        "username": ["al", "validusername", "ch"],  # Length errors
        "email": ["good@email.com", "bad-email", "another@good.com"],
        "age": [25, -5, None]
    }

    df = pd.DataFrame(invalid_data)
    validator = Validator(UserRow)

    try:
        validator.validate(df, errors="raise")
    except ValidationError as e:
        print("❌ Caught validation errors:")
        print(f"Total errors: {len(e.errors())}")
        for error in e.errors()[:2]:  # Print first 2 errors for brevity
            print(f"Field: {error['loc'][0]}, Error: {error['msg']}")

def example3_sqlmodel_integration():
    """SQLModel to DataFrameModel conversion"""
    print("\n=== Example 3: SQLModel Integration ===")

    try:
        from sqlmodel import SQLModel, Field as SQLField

        class SQLUser(SQLModel):
            id: int = SQLField(primary_key=True)
            name: str = SQLField(max_length=30)
            email: str = SQLField(max_length=50)
            age: Optional[int] = SQLField(default=None, nullable=True)

        # Convert to DataFrameModel
        DFUser = DataFrameModel.from_sqlmodel(SQLUser)

        # Valid data
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@example.com", "bob@domain.org", "charlie@test.net"],
            "age": [25, 30, None]
        }

        df = pd.DataFrame(data)
        validator = Validator(DFUser)
        validated_df = validator.validate(df)
        print("✅ SQLModel conversion and validation successful!")
        print(validated_df.dtypes)

    except ImportError:
        print("⚠️ SQLModel not installed, skipping example")

def example4_custom_checks():
    """Custom Pandera checks with DataFrameModel"""
    print("\n=== Example 4: Custom Pandera Checks ===")

    class ProductSchema(DataFrameModel):
        product_id: int = DataFrameField()
        price: float = DataFrameField(gt=0)
        quantity: int = DataFrameField(ge=0)

        @pandera_check
        def check_inventory_value(cls):
            return Check(
                lambda df: (df["price"] * df["quantity"]) < 1000,
                name="inventory_value_under_1000"
            )

    # Data that violates custom check
    data = {
        "product_id": [1, 2, 3],
        "price": [10.0, 20.0, 30.0],
        "quantity": [150, 60, 40]  # 150*10=1500 (invalid)
    }

    df = pd.DataFrame(data)
    validator = Validator(ProductSchema)

    try:
        validator.validate(df, errors="raise")
    except pa.errors.SchemaErrors as e:
        print("❌ Custom check failed:")
        print("Failure cases:")
        print(e.failure_cases)

def example5_error_handling():
    """Error handling and logging"""
    print("\n=== Example 5: Error Handling ===")

    class SimpleSchema(DataFrameModel):
        id: int = DataFrameField(ge=0)
        value: float = DataFrameField(gt=0)

    # Invalid data
    data = {
        "id": [1, -2, 3],
        "value": [10.0, -5.5, 0.0]
    }

    df = pd.DataFrame(data)
    validator = Validator(SimpleSchema)

    print("Testing error logging:")
    errors = []
    def error_handler(e):
        errors.append(e)
        print(f"Logged error: {e}")

    validator = Validator(SimpleSchema, error_handler=error_handler)
    validated_df = validator.validate(df, errors="log")
    print(f"Logged {len(errors)} validation failures")
    print("\nTesting error raising:")
    try:
        validator.validate(df, errors="raise")
    except pa.errors.SchemaErrors as e:
        print(f"Caught {len(e.failure_cases)} validation failures")

def example6_accessor_methods():
    """DataFrame accessor methods"""
    print("\n=== Example 6: Accessor Methods ===")

    class AccessorSchema(DataFrameModel):
        id: int = Field(ge=100)
        status: str = Field(isin=["active", "inactive"])

    # Pandas accessor example
    data = {
        "id": [100, 101, 99],
        "status": ["active", "inactive", "unknown"]
    }
    df = pd.DataFrame(data)

    print("Pandas accessor:")
    try:
        valid_df = df.datantic.validate(AccessorSchema)
    except AttributeError:
        print("⚠️ Pandas plugin not loaded")
    except pa.errors.SchemaErrors as e:
        print(f"⛔ Pandas accessor validation failed: {len(e.failure_cases)} errors")

    # Polars accessor example
    try:
        import polars as pl
        df_pl = pl.DataFrame(data)

        print("\nPolars accessor:")
        try:
            valid_df_pl = df_pl.datantic.validate(AccessorSchema)
        except AttributeError:
            print("⚠️ Polars plugin not loaded")
        except pa.errors.SchemaErrors as e:
            print(f"⛔ Polars accessor validation failed: {len(e.failure_cases)} errors")

    except ImportError:
        print("⚠️ Polars not installed, skipping Polars example")

def example7_nested_pydantic():
    """Nested Pydantic conversion example."""
    print("\n=== Example 7: Nested Pydantic Conversion ===")

    class Address(BaseModel):
        street: str
        city: str

    class User(BaseModel):
        id: int
        name: str
        addresses: List[Address]

    # Sample data (relational)
    data = {
        "id": [1, 1, 2],
        "name": ["Alice", "Alice", "Bob"],
        "street": ["1st St", "2nd St", "3rd St"],
        "city": ["Wonderland", "Wonderland", "Builderland"],
    }

    # Pandas example
    pandas_df = pd.DataFrame(data)
    print("Pandas DataFrame:")
    print(pandas_df)
    id_map = {"User": "id"}
    nested_users_pandas = pandas_df.datantic.to_nested_pydantic(User, id_map)
    print("\nNested Pydantic (Pandas):")
    print(nested_users_pandas)


    # Polars example (if Polars is installed)
    try:
        import polars as pl
        polars_df = pl.DataFrame(data)
        print("\nPolars DataFrame:")
        print(polars_df)

        nested_users_polars = polars_df.datantic.to_nested_pydantic(User, id_map)
        print("\nNested Pydantic (Polars):")
        print(nested_users_polars)
    except ImportError:
        print("\n⚠️ Polars not installed, skipping Polars example")


    # Example with validation
    class UserSchema(DataFrameModel):
        id: int = DataFrameField()
        name: str = DataFrameField()
        street: str = DataFrameField()
        city: str = DataFrameField()

    print("\nWith DataFrameModel Validation:")
    validator = Validator(UserSchema) # Create validator instance

    try:
        # Validate *before* conversion
        validated_df = validator.validate(pandas_df)  # or polars_df
        nested_users_validated = validated_df.datantic.to_nested_pydantic(User, id_map)
        print(nested_users_validated)
    except pa.errors.SchemaErrors as e:
        print(f"❌ Validation failed before conversion: {e}")

    # Or, equivalently, using validate=True in to_nested_pydantic
    print("\nWith to_nested_pydantic validation (Pandas):")
    try:
        nested_users = pandas_df.datantic.to_nested_pydantic(User, id_map, validate=True)
        print(nested_users)
    except pa.errors.SchemaErrors as e:
        print(f"❌ Validation failed during conversion: {e}")


if __name__ == "__main__":
    example1_basic_dataframemodel()
    example2_rowwise_pydantic()
    example3_sqlmodel_integration()
    example4_custom_checks()
    example5_error_handling()
    example6_accessor_methods()
    example7_nested_pydantic() # New example function call