# src/datantic/validators/polars.py (Corrected)
import logging
from typing import Any, Optional, Union

import pandera as pa
from pandera.api.pandas.container import DataFrameSchema
from pandera.errors import SchemaError

from ..converters import to_pandas
from ..types import ErrorHandler

logger = logging.getLogger(__name__)

class PolarsDataFrameValidator:
    """Validator for Polars DataFrames."""

    def __init__(self, pandera_schema: DataFrameSchema):
        """Initialize the validator with a Pandera schema."""
        if not isinstance(pandera_schema, DataFrameSchema):
            raise TypeError("pandera_schema must be a pandera.DataFrameSchema")
        self.pandera_schema = pandera_schema

    def validate_dataframe(
        self,
        data: Any,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        errors: str = "raise",
        error_handler: Optional[ErrorHandler] = None,
    ) -> Any:
        """Validates a Polars DataFrame against a Pandera schema."""
        try:
            # Convert to pandas for validation
            pandas_df = to_pandas(data)

            # Validate using pandera
            validated_df = self.pandera_schema.validate(
                pandas_df,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=False
            )

            # Convert back to polars
            import polars as pl
            return pl.from_pandas(validated_df)

        except SchemaError as e:
            if error_handler:
                error_handler(e)
                return data
            elif errors == "raise":
                raise
            elif errors == "log":  # CORRECTED: Added error handling for "log"
                logger.error(str(e))
                return data
            else:
                raise ValueError(f"Unknown error handling mode: {errors}")
        except Exception as e:
            if errors == "raise":
                raise RuntimeError(f"Error during validation: {e}") from e
            logger.error(f"Error during validation: {e}")
            return data

    def handle_pandera_errors(self, e: SchemaError, original_df: Any) -> Any:
        """Handles Pandera SchemaErrors, returning a filtered Polars DataFrame."""
        try:
            import polars as pl
            if not isinstance(original_df, pl.DataFrame):
                raise TypeError("original_df must be a polars.DataFrame")

            failure_cases = e.failure_cases
            if failure_cases is None or failure_cases.empty:
                return original_df

            # Convert failure cases to Polars
            invalid_indices = pl.from_pandas(failure_cases)["index"].to_list()
            return original_df.filter(~pl.col("index").is_in(invalid_indices))

        except Exception as e:
            logger.error(f"Error handling schema errors: {e}")
            return original_df