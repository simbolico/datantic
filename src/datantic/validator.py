# ./src/datantic/validator.py
import logging
import warnings
from typing import Any, Dict, Hashable, Iterable, List, Literal, Optional, Type, Union, Tuple

import pandera as pa
import pandas as pd
import polars as pl
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from .fields import Field
from .model import DataFrameModel
from .types import ErrorHandler, is_polars_dataframe, _PANDAS_INSTALLED, _POLARS_INSTALLED  # Import flags
from .utils import create_checks
from .validators.pandas import PandasDataFrameValidator # Import from validators submodule
from .validators.polars import PolarsDataFrameValidator # Import from validators submodule
from .converters import to_pandas, to_polars

logger = logging.getLogger(__name__)


class Validator:
    def __init__(
        self,
        schema: Union[
            Type[BaseModel], Type[pa.DataFrameSchema], Type[DataFrameModel]
        ],
        *,
        error_handler: Optional[ErrorHandler] = None,
    ):
        if not isinstance(schema, (type, pa.DataFrameSchema)): # Runtime type check for schema
            raise TypeError(
                "Schema must be a Pydantic BaseModel class, Pandera DataFrameSchema, or DataFrameModel class."
            )

        if isinstance(schema, type) and issubclass(schema, DataFrameModel):
            self.pandera_schema = schema.to_pandera_schema()
            self._polars_schema = schema._polars_schema  # Flag from model
            self.pydantic_model = None
        elif isinstance(schema, pa.DataFrameSchema):
            self.pandera_schema = schema
            self._polars_schema = any(
                str(col.dtype).startswith("pl.")
                for col in self.pandera_schema.columns.values()
            )
            self.pydantic_model = None
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self.pandera_schema = None
            self._polars_schema = False  # Pydantic Models are not Polars Schemas
            self.pydantic_model = schema
            # Check for Field usage and warn if missing:
            for field_name, field_info in schema.model_fields.items():
                if not any(
                    isinstance(metadata, FieldInfo) and "pandera" in metadata.extra
                    for metadata in field_info.metadata
                ):
                    warnings.warn(
                        f"Field '{field_name}' in Pydantic model '{schema.__name__}' "
                        f"does not have a datantic.Field. Validation may be incomplete.",
                        UserWarning,
                    )
            # Generate Pandera schema from Pydantic model for Field checks:
            self._pandera_schema_from_pydantic: Optional[pa.DataFrameSchema] = None
            if issubclass(schema, BaseModel) and not issubclass(schema, DataFrameModel):
                self._generate_pandera_schema_from_pydantic(schema)
        else:
            raise TypeError(
                "Schema must be a Pydantic BaseModel, Pandera DataFrameSchema, "
                "or DataFrameModel."
            )
        self.schema = schema
        self.error_handler = error_handler
        self._dataframe: Any = None  # Raw input, no immediate conversion
        self._polars_df: Any = None  # Cache for converted Polars DF
        self._pandas_validator: Optional[PandasDataFrameValidator] = None # Initialize validators
        self._polars_validator: Optional[PolarsDataFrameValidator] = None
        if self.pandera_schema:
            self._pandas_validator = PandasDataFrameValidator(self.pandera_schema) # Initialize validators based on schema
            self._polars_validator = PolarsDataFrameValidator(self.pandera_schema)

    def _generate_pandera_schema_from_pydantic(self, model_class: Type[BaseModel]) -> None:
        """Generates a Pandera schema from a Pydantic model's datantic.Field metadata (internal)."""
        columns: Dict[str, pa.Column] = {}
        for field_name, field_info in model_class.model_fields.items():
            pandera_field = next(
                (
                    m
                    for m in field_info.metadata
                    if isinstance(m, FieldInfo) and "pandera" in m.extra
                ),
                None,
            )
            if pandera_field:
                pandera_settings = pandera_field.extra["pandera"]
                dtype = pandera_settings["dtype"]
                checks = create_checks(pandera_settings)  # Use utility function
                columns[field_name] = pa.Column(
                    dtype=dtype,
                    checks=checks,
                    nullable=pandera_settings["nullable"],
                    coerce=pandera_settings["coerce"],
                    unique=pandera_settings["unique"],
                    regex=pandera_settings["regex"],
                )
        self._pandera_schema_from_pydantic = pa.DataFrameSchema(columns=columns)

    def validate(
        self,
        data: Any,  # Accept any DataFrame type
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        errors: Literal["raise", "log"] = "raise",
        error_handler: Optional[ErrorHandler] = None,
    ) -> Any:  # type: ignore #Return Any
        """Validates input data against the defined schema."""

        self._dataframe = data  # Store input DataFrame

        if _PANDAS_INSTALLED and isinstance(data, pd.DataFrame):
            # Pandas DataFrame input
            if self.pandera_schema:
                if self._polars_schema: # Pandas DF, but Polars schema -> Convert and validate with Polars
                    if not _POLARS_INSTALLED:
                        raise RuntimeError("Polars is required for validating with Polars Schemas.")
                    polars_df = to_polars(data) # Use converter
                    return self._validate_polars_dataframe_internal( # Call internal polars validation
                        polars_df, head, tail, sample, random_state, lazy, inplace, errors, error_handler
                    )
                else: # Pandas DF and Pandas schema -> Use Pandas validator
                    return self._validate_pandas_dataframe_internal( # Call internal pandas validation
                        data, head, tail, sample, random_state, lazy, inplace, errors, error_handler
                    )
            elif self.pydantic_model: # Pydantic Model validation on Pandas DF
                return self._validate_dataframe_with_pydantic(
                    data, errors, error_handler
                )
        elif is_polars_dataframe(data):
            # Polars DataFrame input
            if self.pandera_schema:
                if not self._polars_schema: # Polars DF, but Pandas schema -> Convert and validate with Pandas
                    if not _PANDAS_INSTALLED:
                        raise RuntimeError("Pandas is required for validating Polars DataFrames with Pandas Schemas.")
                    pandas_df = to_pandas(data) # Use converter
                    return self._validate_pandas_dataframe_internal( # Call internal pandas validation
                        pandas_df, head, tail, sample, random_state, lazy, inplace, errors, error_handler
                    )
                else: # Polars DF and Polars schema -> Use Polars validator
                    return self._validate_polars_dataframe_internal( # Call internal polars validation
                        data, head, tail, sample, random_state, lazy, inplace, errors, error_handler
                    )
            elif self.pydantic_model: # Pydantic Model validation on Polars DF (convert to Pandas first)
                if not _PANDAS_INSTALLED:
                    raise RuntimeError("Pandas is required for Pydantic validation on Polars DataFrames.")
                pandas_df = to_pandas(data) # Use converter
                return self._validate_dataframe_with_pydantic(
                    pandas_df, errors, error_handler
                )
        else:
            raise TypeError(
                "Input data must be a pandas DataFrame or a polars DataFrame."
            )

    def _validate_pandas_dataframe_internal(self, data: pd.DataFrame, *args, **kwargs) -> Any:
        """Internal method to validate using Pandas validator."""
        if self._pandas_validator is None:
            raise ValueError("Pandas validator not initialized.")
        return self._pandas_validator.validate_dataframe(data, *args, **kwargs)

    def _validate_polars_dataframe_internal(self, data: pl.DataFrame, *args, **kwargs) -> Any:
        """Internal method to validate using Polars validator."""
        if self._polars_validator is None:
            raise ValueError("Polars validator not initialized.")
        return self._polars_validator.validate_dataframe(data, *args, **kwargs)


    def _validate_dataframe_with_pydantic(
        self, data: pd.DataFrame, errors: Literal["raise", "log"], error_handler: Optional[ErrorHandler] = None
    ) -> Any:
        """Validates a DataFrame using a Pydantic model (row-wise) (internal)."""
        if not _PANDAS_INSTALLED:
            raise RuntimeError("Pandas is required for row-wise Pydantic validation.")

        # ... (Pydantic validation logic - mostly unchanged, but ensure Pandas iteration) ...
        if (
            self.pydantic_model is not None and self.pandera_schema is None
        ):  # It's a pure Pydantic BaseModel
            # Validate with Pandera schema from datantic.Fields FIRST:
            if self._pandera_schema_from_pydantic:
                try:
                    data = self._pandera_schema_from_pydantic.validate(data)
                except pa.errors.SchemaErrors as e:
                    if error_handler:
                        error_handler(e)  # Call error handler
                    elif errors == "raise":
                        raise e
                    elif errors == "log":
                        logger.error(e)

            # Log warning for large DataFrames:
            if len(data) > 10000:
                warnings.warn(
                    "Row-wise Pydantic validation on a large DataFrame may be slow. "
                    "Consider using a DataFrameModel or DataFrameSchema for better performance.",
                    UserWarning,
                )

            # Row-wise Pydantic validation:
            validated_data: List[Dict[str, Any]] = []
            for _, row in data.iterrows():
                try:
                    validated_model = self.pydantic_model.model_validate(row.to_dict())
                    validated_data.append(validated_model.model_dump())
                except ValidationError as e:
                    if error_handler:
                        error_handler(e)  # Call error handler
                    elif errors == "raise":
                        raise e
                    elif errors == "log":
                        logger.error(e)
                        validated_data.append(
                            row.to_dict()
                        )  # keep original values if logging errors
                    else:
                        raise ValueError(f"Unknown error handling mode: {errors}")
            return pd.DataFrame(validated_data)
        else:
            raise RuntimeError(
                "Cannot validate with pydantic if pandera_schema is available."
            )


    def is_valid(
        self,
        data: Any,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> bool:
        """Verify if data is valid without raising an exception."""
        try:
            self.validate(
                data=data,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
            )
            return True
        except (pa.errors.SchemaErrors, ValidationError):
            return False

    def iterate(
        self, dataframe: Any, verbose: bool = True
    ) -> Iterable[Tuple[Hashable, Union[Dict[str, Any], BaseModel]]]:
        """Iterates over *valid* rows, yielding (index, row_data) tuples."""
        if _PANDAS_INSTALLED and isinstance(dataframe, pd.DataFrame) and self._pandas_validator:
            validator = self._pandas_validator
            error_handler = validator.handle_pandera_errors
        elif _POLARS_INSTALLED and is_polars_dataframe(dataframe) and self._polars_validator:
            validator = self._polars_validator
            error_handler = validator.handle_pandera_errors
        else: # Fallback to handle errors in base validator (e.g. type errors)
            validator = self
            error_handler = self._handle_pandera_errors_fallback # dummy fallback handler, returns original df

        if not (isinstance(dataframe, pd.DataFrame) or is_polars_dataframe(dataframe)):
            raise TypeError("iterate() requires a DataFrame input.")


        # --- Prepare DataFrame (ensure index is available for Polars) ---
        df_to_iterate = dataframe
        if is_polars_dataframe(dataframe):
            import polars as pl # Local import for polars

            if self.pandera_schema and "index" not in dataframe.columns:
                raise ValueError(
                    "The DataFrame must have a column named 'index' when using "
                    "iterate() with Polars and DataFrameModel/DataFrameSchema. "
                    "This column is required for filtering invalid rows."
                )
            df_to_iterate = dataframe


        if self.pydantic_model is not None and self.pandera_schema is None:  # Pure Pydantic
             # Pydantic row-wise validation (Pandas and Polars handled the same)
            for i, row in (df_to_iterate.iterrows() if isinstance(df_to_iterate, pd.DataFrame)
                          else enumerate(df_to_iterate.to_dicts())):
                try:
                    # Adapt for Pandas/Polars
                    row_data = row.to_dict() if isinstance(row, pd.Series) else row
                    validated_model = self.pydantic_model.model_validate(row_data)
                    yield i, validated_model
                except ValidationError as e:
                    if verbose:
                        logger.info(f"Validation error at index {i}, skipping: {e}.")
                    continue
        elif self.pandera_schema:  # DataFrameModel or DataFrameSchema
            try:
                # Validate the entire DataFrame upfront (lazy=True is important for efficiency)
                validated_df = self.pandera_schema.validate(df_to_iterate, lazy=True) # Use standard pandera validate here

                # Iterate over the *validated* DataFrame
                for i, row in (validated_df.iterrows() if isinstance(validated_df, pd.DataFrame)
                              else enumerate(validated_df.to_dicts())):
                    # Adapt for Pandas/Polars
                    row_data = row.to_dict() if isinstance(row, pd.Series) else row
                    yield i, row_data

            except pa.errors.SchemaErrors as e:
                # Handle schema errors and extract valid rows using specific validator
                if validator is not self: # Use specific validator if available, otherwise fallback does not filter
                    valid_df = error_handler(e, df_to_iterate)
                    for i, row in (valid_df.iterrows() if isinstance(valid_df, pd.DataFrame)
                                  else enumerate(valid_df.to_dicts())):

                        row_data = row.to_dict() if isinstance(row, pd.Series) else row
                        yield i, row_data
                else: # Fallback for type errors etc. - no filtering possible, yield all rows of original dataframe
                    warnings.warn("Could not use specific validator for error handling, falling back to returning unfiltered data.", UserWarning)
                    for i, row in (df_to_iterate.iterrows() if isinstance(df_to_iterate, pd.DataFrame)
                                  else enumerate(df_to_iterate.to_dicts())):
                        row_data = row.to_dict() if isinstance(row, pd.Series) else row
                        yield i, row_data
        else:
            if verbose:
                warnings.warn("Dataframe type not supported, skipping iteration")
            return None

    def _handle_pandera_errors_fallback(self, e: pa.errors.SchemaErrors, original_df: Any) -> Any:
        """Dummy fallback error handler when specific validators are not available."""
        warnings.warn("Using fallback error handler, no filtering of invalid rows.")
        return original_df # Returns original dataframe without filtering