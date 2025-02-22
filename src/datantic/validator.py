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
from .nesting import get_model_columns, serialize_dataframe

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

        if isinstance(schema, type) and not issubclass(schema, (BaseModel, DataFrameModel)):
            raise TypeError(
                "The `schema` argument must be a Pydantic BaseModel class, DataFrameModel class, or Pandera DataFrameSchema."
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
        """Validates data against the schema."""
        # Handle different DataFrame types
        if is_polars_dataframe(data):
            return self._validate_polars_dataframe_internal(
                data,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                errors=errors,
                error_handler=error_handler,
            )
        elif isinstance(data, pd.DataFrame):
            return self._validate_pandas_dataframe_internal(
                data,
                head=head,
                tail=tail,
                sample=sample,
                random_state=random_state,
                lazy=lazy,
                inplace=inplace,
                errors=errors,
                error_handler=error_handler,
            )
        else:
            raise TypeError("Data must be a Pandas or Polars DataFrame")

    def _validate_pandas_dataframe_internal(self, data: pd.DataFrame, *args, **kwargs) -> Any:
        """Internal method to validate Pandas DataFrame."""
        if self.pydantic_model is not None and self.pandera_schema is None:
            return self._validate_dataframe_with_pydantic(data, kwargs.get("errors", "raise"), kwargs.get("error_handler"))
        else:
            # Remove datantic-specific kwargs that pandera doesn't understand
            pandera_kwargs = {k: v for k, v in kwargs.items() if k not in ["errors", "error_handler"]}
            return self.pandera_schema.validate(data, *args, **pandera_kwargs)

    def _validate_polars_dataframe_internal(self, data: Any, *args, **kwargs) -> Any:
        """Internal method to validate Polars DataFrame."""
        import polars as pl
        # Convert to pandas for validation
        pandas_df = to_pandas(data)
        result = self._validate_pandas_dataframe_internal(pandas_df, *args, **kwargs)
        # Convert back to polars
        if isinstance(result, pd.DataFrame):
            return pl.from_pandas(result)
        return result

    def _validate_dataframe_with_pydantic(
        self, data: pd.DataFrame, errors: Literal["raise", "log"], error_handler: Optional[ErrorHandler] = None
    ) -> Any:
        """Validates a DataFrame using a Pydantic model (row-wise) (internal)."""
        if not _PANDAS_INSTALLED:
            raise RuntimeError("Pandas is required for row-wise Pydantic validation.")

        try:
            validated_data = []
            validation_errors = []
            for _, row in data.iterrows():
                try:
                    # Validate the row
                    validated_model = self.pydantic_model.model_validate(row.to_dict())
                    validated_data.append(validated_model.model_dump())
                except ValidationError as e:
                    validation_errors.extend(e.errors())
                    if errors == "log":
                        if error_handler:
                            error_handler(e)
                        else:
                            logger.error(str(e))
                    continue

            if validation_errors and errors == "raise":
                raise ValidationError.from_exception_data(
                    title="",
                    line_errors=validation_errors
                )

            if not validated_data:
                return pd.DataFrame()

            return pd.DataFrame(validated_data)

        except Exception as e:
            if errors == "raise":
                raise e
            elif errors == "log":
                if error_handler:
                    error_handler(e)
                else:
                    logger.error(str(e))
            return pd.DataFrame()

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