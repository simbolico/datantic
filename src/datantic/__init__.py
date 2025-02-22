"""
`datantic`: Seamless Data Validation for Pandas and Polars DataFrames

`datantic` provides a unified and flexible way to validate Pandas and Polars
DataFrames using Pydantic-like schemas. It builds upon the foundation of
`pandera` and Pydantic, offering a clean, type-hinted API for defining
DataFrame structures and constraints.

Key Features:

*   **Unified API:** Validate both Pandas and Polars DataFrames using the same
    `Validator` class and `DataFrameModel` schema definitions.
*   **Automatic Dispatch:** `datantic` automatically detects the DataFrame
    type (Pandas or Polars) and dispatches to the appropriate validation logic.
*   **Optional Dependencies:** Pandas and Polars are *optional* dependencies.
    Install only what you need.
*   **Performance Optimization:** Leverages Polars' performance advantages when
    possible, minimizing unnecessary DataFrame conversions.
*   **Extensible Plugin Architecture:** DataFrame library-specific functionality
    (like DataFrame accessors) is implemented as plugins, loaded conditionally.
*   **Tight Integration with Pandera:**  Uses `pandera` for schema definition
    and validation, ensuring compatibility and leveraging its powerful features.
*   **Pydantic Compatibility:** Works seamlessly with Pydantic models for
    row-wise validation and data conversion.
*   **SQLModel Support:** DataFrameModels can be easily created from existing
    SQLModel classes.

Basic Usage:

```python
from datantic import DataFrameModel, DataFrameField, Validator
import pandas as pd  # Or import polars as pl

class MySchema(DataFrameModel):
    id: int = DataFrameField(ge=0)
    name: str = DataFrameField(nullable=False)
    value: float = DataFrameField(gt=0)

df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [1.0, 2.0]})

validator = Validator(MySchema)
validated_df = validator.validate(df)  # Returns a validated DataFrame

# Accessor (if Pandas is installed)
validated_df = df.datantic.validate(MySchema)
```

Installation:

```bash
# Install with Pandas support:
pip install "datantic[pandas]"

# Install with Polars support:
pip install "datantic[polars]"

# Install with both:
pip install "datantic[pandas,polars]"

# Install with all optional dependencies
pip install "datantic[all]"

# Install base package (no DataFrame support):
pip install datantic
```

Exports:
    The following members are explicitly exported by the package and are
    available for use.

"""
from .model import DataFrameModel, pandera_check
from .validator import Validator
from .fields import DataFrameField, Field
from .types import Optional

# The plugins are loaded conditionally in plugins/__init__.py

__all__ = [
    "DataFrameModel",
    "Validator",
    "DataFrameField",
    "Field",
    "Optional",
    "pandera_check",
]

# Development version - hardcoded until package is installed
__version__ = "0.1.0"