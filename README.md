# datantic: Seamless Data Validation for Pandas and Polars DataFrames

`datantic` unifies and simplifies DataFrame validation for both Pandas and Polars, combining the expressive power of Pydantic with the robust validation capabilities of Pandera. Define your DataFrame schemas with type hints and constraints, and `datantic` handles the rest, intelligently dispatching to optimized validation logic based on your DataFrame type.

## Philosophy: Declarative Validation, Effortless Integration

`datantic` is built on these core principles:

*   **Declarative Schemas:** Define *what* your data should look like, not *how* to validate it. Use familiar Pydantic-style models with type hints and constraints.
*   **Unified API:**  Validate Pandas and Polars DataFrames using the *same* schema definition and `Validator` class. No more duplicated validation logic.
*   **Optional Dependencies:**  Pandas and Polars are *optional*. Install only what you need, keeping your environment lightweight.
*   **Performance Matters:** Leverages Polars' speed when possible, minimizing conversions, and utilizes Pandera's optimized validation engine.
*   **Extensible Design:**  Plugin architecture for DataFrame library-specific functionality (like accessors) makes extending `datantic` easy.
*   **Seamless Integration:** Works harmoniously with Pydantic and Pandera, providing a consistent and intuitive experience.
* **SQLModel Support**: Easily create `DataFrameModel`s from your existing `SQLModel` classes, bridging the gap between database and data validation.

## Key Features

*   **Unified Validation:**  One schema, two DataFrame libraries. Validate Pandas and Polars DataFrames with the same `DataFrameModel`.
*   **Automatic Dispatch:**  `datantic` intelligently detects your DataFrame type (Pandas or Polars) and uses the most efficient validation logic.
*   **Pydantic-Powered Schemas:**  Define schemas using Pydantic models, leveraging type hints and built-in validators.
*   **Pandera Integration:**  Utilizes Pandera for robust schema definition and validation, inheriting its powerful features.
*   **DataFrame Accessors:**  Convenient `.datantic` accessor for both Pandas (`df.datantic.validate()`) and Polars (`pl_df.datantic.validate()`) DataFrames (requires respective library installation).
*   **Optional Dependencies:**  Install only what you need: `datantic[pandas]`, `datantic[polars]`, `datantic[all]`, or just the core `datantic` library.
*   **Row-Wise Validation:**  Validate individual rows against a Pydantic model.
*   **Iterate Valid Rows:**  Efficiently iterate over only the rows that pass validation, perfect for data cleaning pipelines.
*   **Custom Error Handling:**  Choose to raise exceptions, log errors, or provide your own error handling function.
*   **SQLModel Compatibility:** Generate `DataFrameModel` classes directly from `SQLModel` classes, streamlining your workflow.
*   **Lazy Validation:** Supports Pandera's lazy validation for improved performance on large DataFrames.
*   **Type-Safe:**  Extensively type-hinted for better code clarity and maintainability.

## Installation

```bash
# Install with Pandas support:
pip install "datantic[pandas]"

# Install with Polars support:
pip install "datantic[polars]"

# Install with both:
pip install "datantic[pandas,polars]"

# Install with all optional dependencies (includes Pandas, Polars, and SQLModel):
pip install "datantic[all]"

# Install base package (no DataFrame support, only Pydantic model validation):
pip install datantic
```

## Basic Usage

```python
from datantic import DataFrameModel, Field, Validator
import pandas as pd  # Or import polars as pl

# Define your DataFrame schema
class ProductSchema(DataFrameModel):
    product_id: int = Field(gt=0, description="Unique product identifier")
    name: str = Field(nullable=False, description="Product name")
    price: float = Field(ge=0, description="Product price")
    in_stock: bool = Field(description="Whether the product is in stock")

    class Config:
        coerce = True  # Automatically coerce data types if possible

# Create a Pandas DataFrame (or a Polars DataFrame)
data = {
    "product_id": [1, 2, "3", 4],  # Note: "3" will be coerced to an integer
    "name": ["Laptop", "Mouse", "Keyboard", None],
    "price": [1200.0, 25.50, 75.0, 10.99],
    "in_stock": [True, True, False, True],
}
df = pd.DataFrame(data)
# df = pl.DataFrame(data) # Works the same with Polars


# Validate the DataFrame
validator = Validator(ProductSchema)
validated_df = validator.validate(df, errors="raise")  # Raise exceptions on errors
print(validated_df)

# Using the Pandas accessor (if Pandas is installed)
# validated_df = df.datantic.validate(ProductSchema)

# Using the Polars accessor (if polars is installed)
# validated_df = pl_df.datantic.validate(ProductSchema)

# Check if valid without raising an exception
is_valid = validator.is_valid(df)
print(f"DataFrame is valid: {is_valid}")

# Iterate over only the valid rows.
for index, row in validator.iterate(df):
	print(f"Index: {index}, Row: {row}")

# Example with a custom error handler:
def my_error_handler(error):
    print(f"Validation Error: {error}")

validated_df = validator.validate(df, error_handler=my_error_handler)
print(validated_df)
```

## Advanced Usage

### Class-Level Checks (using `pandera_check`)

```python
from datantic import DataFrameModel, Field, pandera_check
import pandera as pa
import pandas as pd

class MySchema(DataFrameModel):
    col1: int = Field()
    col2: int = Field()

    @pandera_check
    def _check_sum(cls, df: pd.DataFrame) -> pa.Check:  # or -> bool
        return pa.Check(df["col1"] + df["col2"] > 10, name="sum_check")

# Example usage same as before with Validator or accessor
```

### Pure Pydantic Model Validation (Row-Wise)

```python
from datantic import Validator, Field
from pydantic import BaseModel
import pandas as pd

class User(BaseModel):  # Regular Pydantic model, *not* DataFrameModel
    id: int = Field(gt=0)
    name: str
    age: int = Field(ge=18)

data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 15]}
df = pd.DataFrame(data)

validator = Validator(User)  # Pass the Pydantic model
validated_df = validator.validate(df, errors="raise")
print(validated_df)

# Iterate over validated models
for index, user in validator.iterate(df):
     print(user)
     print(type(user))
```

### SQLModel Integration

```python
from datantic import DataFrameModel
from sqlmodel import SQLModel, Field as SQLField, Column, Integer, String

# Define your SQLModel
class UserSQL(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    name: str = SQLField()
    age: int = SQLField(sa_column=Column(Integer, nullable=False))

# Create a DataFrameModel from the SQLModel
UserDataFrame = DataFrameModel.from_sqlmodel(UserSQL)

# Now you can use UserDataFrame for validation:
# validator = Validator(UserDataFrame)
# ...
```

### Using `datantic.Field` with Pydantic Models for Column-Level Checks

```python
from pydantic import BaseModel
from datantic import Field, Validator
import pandas as pd

class MyModel(BaseModel):
    value: int = Field(gt=10, lt=100) # Use datantic.Field


df = pd.DataFrame({"value": [5, 20, 50, 110]})

validator = Validator(MyModel)
validated_df = validator.validate(df)
print(validated_df)
```
### Iterating with different outputs

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

- **Simplified DataFrame Validation:**  `datantic` bridges the gap between Pydantic's data modeling and Pandera's DataFrame validation, providing a clean and intuitive API.
- **Reduced Boilerplate:**  Avoid writing repetitive validation logic for different DataFrame libraries.
- **Improved Code Readability:**  Declarative schemas make your data expectations explicit and easy to understand.
- **Enhanced Data Quality:**  Catch data errors early and ensure data integrity.
- **Faster Development:**  Spend less time on manual validation and more time building your applications.
- **Type safety**: It makes your dataframe operations type safe.

`datantic` empowers you to work with DataFrames confidently, knowing that your data is validated and consistent, regardless of whether you're using Pandas or Polars.

## Contributing

We welcome contributions!  Please see [CONTRIBUTING.md](CONTRIBUTING.md) (to be created) for guidelines.

## License

`datantic` is licensed under the MIT License. See [LICENSE](LICENSE) for details.