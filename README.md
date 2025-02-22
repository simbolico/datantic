# datantic: Seamless Data Validation and Conversion for Pandas and Polars DataFrames

`datantic` unifies and simplifies DataFrame validation and conversion for both Pandas and Polars.  It combines the expressive power of Pydantic with the robust validation capabilities of Pandera, and adds powerful features for transforming DataFrames into nested Pydantic models.  Define your DataFrame schemas and conversion rules with type hints and constraints, and `datantic` handles the rest, intelligently dispatching to optimized logic based on your DataFrame type.

## Philosophy: Declarative Validation and Conversion, Effortless Integration

`datantic` is built on these core principles:

*   **Declarative Schemas:** Define *what* your data should look like, not *how* to validate or convert it. Use familiar Pydantic-style models.
*   **Unified API:** Validate and convert Pandas and Polars DataFrames using the *same* schema definition and `Validator` class.
*   **Optional Dependencies:** Pandas, Polars, and SQLModel are *optional*. Install only what you need.
*   **Performance Matters:** Leverages Polars' speed when possible and minimizes unnecessary conversions.
*   **Extensible Design:** Plugin architecture for DataFrame library-specific functionality.
*   **Seamless Integration:** Works harmoniously with Pydantic, Pandera, and SQLModel.

## Key Features

*   **Unified Validation:** One schema, two DataFrame libraries. Validate Pandas and Polars DataFrames with the same `DataFrameModel`.
*   **Automatic Dispatch:** Intelligently detects your DataFrame type and uses the most efficient validation logic.
*   **Pydantic-Powered Schemas:** Define schemas using Pydantic models, leveraging type hints and validators.
*   **Pandera Integration:** Utilizes Pandera for robust schema definition and validation.
*   **DataFrame Accessors:** Convenient `.datantic` accessor for Pandas and Polars DataFrames.
*   **Optional Dependencies:** Install only what you need: `datantic[pandas]`, `datantic[polars]`, `datantic[all]`, or just the core.
*   **Row-Wise Validation:** Validate individual rows against a Pydantic model.
*   **Iterate Valid Rows:** Efficiently iterate over only the valid rows.
*   **Custom Error Handling:** Raise exceptions, log errors, or provide a custom error handler.
*   **SQLModel Compatibility:** Generate `DataFrameModel` classes from `SQLModel` classes.
*   **Lazy Validation:** Supports Pandera's lazy validation.
*   **Type-Safe:** Extensively type-hinted.
*   **NEW: Nested Pydantic Conversion:** Transform DataFrames with hierarchical data into nested Pydantic models.

## Installation

```bash
# Install with Pandas support:
pip install "datantic[pandas]"

# Install with Polars support:
pip install "datantic[polars]"

# Install with both Pandas and Polars:
pip install "datantic[pandas,polars]"

# Install with all optional dependencies (Pandas, Polars, and SQLModel):
pip install "datantic[all]"

# Install base package (no DataFrame support, only Pydantic model validation):
pip install datantic
```

## Basic Usage (Validation)

```python
from datantic import DataFrameModel, Field, Validator
import pandas as pd  # Or import polars as pl

# Define your DataFrame schema
class ProductSchema(DataFrameModel):
    product_id: int = Field(gt=0)
    name: str = Field(nullable=False)
    price: float = Field(ge=0)
    in_stock: bool

    class Config:
        coerce = True  # Automatically coerce data types

# Create a Pandas DataFrame (or Polars)
data = {
    "product_id": [1, 2, "3", 4],
    "name": ["Laptop", "Mouse", "Keyboard", None],
    "price": [1200.0, 25.50, 75.0, 10.99],
    "in_stock": [True, True, False, True],
}
df = pd.DataFrame(data)
# df = pl.DataFrame(data)  # Works with Polars too

# Validate (using Validator directly)
validator = Validator(ProductSchema)
validated_df = validator.validate(df, errors="raise")
print(validated_df)

# Validate (using Pandas accessor)
# validated_df = df.datantic.validate(ProductSchema)

# Validate (using Polars accessor)
# validated_df = df.datantic.validate(ProductSchema)  # If df is a Polars DataFrame

# Check validity
is_valid = validator.is_valid(df)
print(f"DataFrame is valid: {is_valid}")
```

## Nested Pydantic Conversion

```python
from datantic import DataFrameModel, Field, Validator
from pydantic import BaseModel
from typing import List
import pandas as pd  # Or import polars as pl

# Define your Pydantic models (including nested models)
class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    id: int
    name: str
    addresses: List[Address]

# Sample DataFrame (relational data)
data = {
    "id": [1, 1, 2],
    "name": ["Alice", "Alice", "Bob"],
    "street": ["1st St", "2nd St", "3rd St"],
    "city": ["Wonderland", "Wonderland", "Builderland"],
}
df = pd.DataFrame(data)
# df = pl.DataFrame(data)  # Works with Polars too

# --- Conversion without prior validation ---
id_map = {"User": "id"}  # Map model names to ID columns
nested_users = df.datantic.to_nested_pydantic(User, id_map)
print(nested_users)


# --- Conversion WITH prior validation (recommended) ---

# 1. Define a DataFrameModel for validation:
class UserSchema(DataFrameModel):
    id: int = Field()
    name: str = Field()
    street: str = Field()
    city: str = Field()

# 2. Create a Validator instance:
validator = Validator(UserSchema)

# 3. Validate, then convert (two steps):
validated_df = validator.validate(df)  # Raises exception if invalid
nested_users = validated_df.datantic.to_nested_pydantic(User, id_map)

# 4. Or, validate during conversion (using the `validate` parameter):
try:
    nested_users = df.datantic.to_nested_pydantic(User, id_map, validate=True)
    # If validation fails, a pa.errors.SchemaErrors exception is raised.
except pa.errors.SchemaErrors as e:
  print(f"Validation failed, {e}")
```

**Explanation:**

*   **`to_nested_pydantic(model, id_column_map, validate=True, **kwargs)`:**  This is the key method.
    *   `model`:  The *root* Pydantic model class (e.g., `User`).
    *   `id_column_map`:  A dictionary *required* for nested conversion.  It maps Pydantic model names to the corresponding ID column names in your DataFrame.  This tells `datantic` how to group rows to form the nested structure.  For example: `{"User": "id", "Address": "address_id"}` (if you had a separate `address_id` column).
    *   `validate`:  If `True` (the default), `datantic` will validate the DataFrame against a `DataFrameModel` (if one is associated with the `Validator` used internally by the accessor) *before* performing the conversion. This is highly recommended.
    * `**kwargs`: Pass extra arguments from pandas or polars validation methods.
*   **Best Practice:** It's generally recommended to define a `DataFrameModel` and use the `validate=True` option. This ensures your data is valid *before* you attempt the potentially complex nested conversion.

## Advanced Usage

### Class-Level Checks (using `pandera_check`)

```python
from datantic import DataFrameModel, Field, pandera_check
import pandera as pa

class MySchema(DataFrameModel):
    col1: int = Field()
    col2: int = Field()

    @pandera_check
    def _check_sum(cls, df):  # Can operate on Pandas or Polars DataFrames
        return pa.Check(df["col1"] + df["col2"] > 10)
```

### Pure Pydantic Model Validation (Row-Wise)

```python
from datantic import Validator, Field
from pydantic import BaseModel

class User(BaseModel):  # Regular Pydantic model
    id: int = Field(gt=0)
    name: str
    age: int = Field(ge=18)

# ... (use Validator.validate(df) or validator.iterate(df) as before)
```

### SQLModel Integration

```python
from datantic import DataFrameModel
from sqlmodel import SQLModel, Field as SQLField

class UserSQL(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    name: str
    # ...

UserDataFrame = DataFrameModel.from_sqlmodel(UserSQL)
# ... (use Validator(UserDataFrame) as before)
```
### Other Features
* Using `datantic.Field` with Pydantic Models for Column-Level Checks
* Iterating with different outputs

```python

from datantic import DataFrameModel, Field, Validator
import pandas as pd

class MySchema(DataFrameModel):
    col1: int = Field()
    col2: str = Field()

df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
validator = Validator(MySchema)

# itertuples
for i, (v1, v2) in df.datantic.itertuples(MySchema):
    print(f"{i=}, {v1=}, {v2=}")

# iterrows
for i, row in df.datantic.iterrows(MySchema):
    print(f"{i=}, {row=}")

# iterschemas (DataFrameModel or DataFrameSchema)
for i, row_dict in df.datantic.iterschemas(MySchema):
    print(f"{i=}, {row_dict=}")

# iterschemas (BaseModel)
from pydantic import BaseModel
class MyBase(BaseModel):
    col1: int
    col2: str
for i, model in df.datantic.iterschemas(MyBase):
    print(f"{i=}, {model=}")
```

## Why `datantic`?

*   **Simplified DataFrame Operations:** Bridges Pydantic, Pandera, Pandas and Polars for validation and transformation.
*   **Reduced Boilerplate:** Avoid repetitive validation and conversion logic.
*   **Improved Code Readability:** Declarative schemas make your data expectations clear.
*   **Enhanced Data Quality:** Catch data errors early.
*   **Faster Development:** Spend less time on manual validation and conversion.
*   **Type Safety:**  DataFrame operations become type-safe.