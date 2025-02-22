# src/datantic/validators/pandas.py (Corrected)

import logging
from typing import Any, Literal, Optional, Type, Union

import pandas as pd
import pandera as pa
from pydantic import BaseModel

from ..types import ErrorHandler  # Corrected import: Relative import from '..'
from ..utils import _PANDAS_INSTALLED # Corrected import: Relative import from '..'

logger = logging.getLogger(__name__)

class PandasDataFrameValidator:
    """Validator for Pandas DataFrames."""

    def __init__(self, pandera_schema: pa.DataFrameSchema):
        if not _PANDAS_INSTALLED:
            raise RuntimeError("Pandas is required for validating Pandas DataFrames.")
        self.pandera_schema = pandera_schema

    def validate_dataframe(
        self,
        data: pd.DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        errors: Literal["raise", "log"] = "raise",
        error_handler: Optional[ErrorHandler] = None,
    ) -> pd.DataFrame:
        """Validates a Pandas DataFrame against a Pandera schema."""

        try:
            validated_df = self.pandera_schema.validate(
                data, head=head, tail=tail, sample=sample, random_state=random_state, lazy=lazy, inplace=inplace
            )
            return validated_df
        except pa.errors.SchemaErrors as e:
            if error_handler:
                error_handler(e)
            elif errors == "raise":
                raise e
            elif errors == "log":  # CORRECTED: Added error handling for "log"
                logger.error(e)
                return data  # Return original df
            else:
                raise ValueError(f"Unknown error handling mode: {errors}")

    def handle_pandera_errors(self, e: pa.errors.SchemaErrors, original_df: pd.DataFrame) -> pd.DataFrame:
        """Handles Pandera SchemaErrors, returning a filtered Pandas DataFrame."""
        if not _PANDAS_INSTALLED:
            raise RuntimeError("Pandas is required for handling errors in Pandas DataFrames.")

        failure_cases = e.failure_cases
        if failure_cases is None or failure_cases.empty:
            return original_df

        invalid_indices = failure_cases.index.to_list()
        return original_df[~original_df.index.isin(invalid_indices)]