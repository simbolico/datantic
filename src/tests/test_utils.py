# tests/test_utils.py
import pytest
from src.datantic.utils import create_checks
from src.datantic.fields import DataFrameField
import pandera as pa
from pandera import Check
import re

def test_create_checks_gt():
    """Test create_checks with gt."""
    field_settings = DataFrameField(gt=0)
    checks = create_checks(field_settings)
    assert len(checks) == 1
    assert isinstance(checks[0], Check)

def test_create_checks_isin():
    """Test create_checks with isin."""
    field_settings = DataFrameField(isin=[1, 2, 3])
    checks = create_checks(field_settings)
    assert len(checks) == 1
    assert isinstance(checks[0], Check)

def test_create_checks_pattern():
    """Test create_checks with a pattern."""
    field_settings = DataFrameField(pattern=r"^\d+$")
    checks = create_checks(field_settings)
    assert len(checks) == 1
    assert isinstance(checks[0], Check)

def test_create_checks_multiple_constraints():
    """Tests create_checks with more than one constraint."""
    field_settings = DataFrameField(gt=0, lt=10, isin=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    checks = create_checks(field_settings)
    assert len(checks) == 3

def test_create_checks_multiple_constraints_dict():
    """Tests that create_checks also works with dicts"""
    field_settings = {"gt": 0, "lt": 10, "isin": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    checks = create_checks(field_settings)
    assert len(checks) == 3

def test_create_checks_invalid_numeric_constraint():
    """Tests that invalid numeric constraints such as a string throw an error"""
    field_settings = {"gt": "notANumber"}
    with pytest.raises(TypeError, match="gt constraint must be numeric"):
        create_checks(field_settings)

def test_create_checks_invalid_list_constraint():
    """Tests that invalid list constraints such as an int throw an error"""
    field_settings = {"isin": 1}
    with pytest.raises(TypeError, match="isin constraint must be a list or tuple"):
        create_checks(field_settings)

def test_create_checks_invalid_pattern_constraint():
    """Tests that invalid pattern constraints such as an int throw an error"""
    field_settings = {"pattern": 1}
    with pytest.raises(TypeError, match="pattern constraint must be a string"):
        create_checks(field_settings)

def test_create_checks_invalid_regex_pattern():
    """Tests that invalid regex patterns such as missing closing brackets throw an error"""
    field_settings = {"pattern": "["}
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        create_checks(field_settings)