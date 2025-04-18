import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.overrides import apply_overrides_with_loop, build_condition

class MockConfig:
    """Mock configuration class for testing."""
    def __init__(self, mode):
        self.mode = mode

@pytest.fixture
def pi_config():
    """Create a PI mode configuration for testing."""
    return MockConfig("pi")

@pytest.fixture
def comment_config():
    """Create a comment mode configuration for testing."""
    return MockConfig("comment")

@pytest.fixture
def pi_csv_dict():
    """Create a sample CSV dictionary for PI mode testing."""
    return [
        {
            "catalog": "catalog1",
            "schema": "schema1",
            "table": "table1",
            "column": "column1",
            "classification": "new_classification",
            "type": "new_type"
        }
    ]

@pytest.fixture
def comment_csv_dict():
    """Create a sample CSV dictionary for comment mode testing."""
    return [
        {
            "catalog": "catalog1",
            "schema": "schema1",
            "table": "table1",
            "column": "column1",
            "comment": "new_comment"
        }
    ]

def test_apply_overrides_with_loop_pi_mode(sample_df, pi_csv_dict, pi_config):
    """Test apply_overrides_with_loop in PI mode."""
    result_df = apply_overrides_with_loop(sample_df, pi_csv_dict, pi_config)
    
    matching_row = result_df.filter(
        (col("catalog") == "catalog1") & 
        (col("schema") == "schema1") & 
        (col("table_name") == "table1") & 
        (col("column_name") == "column1")
    ).collect()[0]
    
    assert matching_row.classification == "new_classification"
    assert matching_row.type == "new_type"
    
    other_row = result_df.filter(
        (col("catalog") == "catalog2")
    ).collect()[0]
    
    assert other_row.classification == "classification2"
    assert other_row.type == "type2"

def test_apply_overrides_with_loop_comment_mode(sample_df, comment_csv_dict, comment_config):
    """Test apply_overrides_with_loop in comment mode."""
    result_df = apply_overrides_with_loop(sample_df, comment_csv_dict, comment_config)
    
    matching_row = result_df.filter(
        (col("catalog") == "catalog1") & 
        (col("schema") == "schema1") & 
        (col("table_name") == "table1") & 
        (col("column_name") == "column1")
    ).collect()[0]
    
    assert matching_row.column_content == "new_comment"
    
    other_row = result_df.filter(
        (col("catalog") == "catalog2")
    ).collect()[0]
    
    assert other_row.column_content == "content2"

def test_apply_overrides_with_loop_invalid_mode(sample_df, pi_csv_dict):
    """Test apply_overrides_with_loop with invalid mode."""
    invalid_config = MockConfig("invalid_mode")
    
    with pytest.raises(ValueError, match="Invalid mode provided."):
        apply_overrides_with_loop(sample_df, pi_csv_dict, invalid_config)

def test_apply_overrides_with_loop_empty_csv_dict(sample_df, pi_config):
    """Test apply_overrides_with_loop with empty CSV dictionary."""
    empty_csv_dict = []
    
    result_df = apply_overrides_with_loop(sample_df, empty_csv_dict, pi_config)
    
    # Verify the DataFrame was not changed
    original_rows = sample_df.collect()
    result_rows = result_df.collect()
    
    for i in range(len(original_rows)):
        assert original_rows[i].classification == result_rows[i].classification
        assert original_rows[i].type == result_rows[i].type


@pytest.fixture
def column_only_pi_csv_dict():
    """Create a sample CSV dictionary for PI mode testing with column-only specification."""
    return [
        {
            "catalog": "",
            "schema": "",
            "table": "",
            "column": "ssn",
            "classification": "pi",
            "type": "pii"
        },
    ]

@pytest.fixture
def mixed_pi_csv_dict():
    """Create a sample CSV dictionary for PI mode testing with mixed specifications."""
    return [
        {
            "catalog": "",
            "schema": "",
            "table": "",
            "column": "email",
            "classification": "pi",
            "type": "pii"
        },
        {
            "catalog": "catalog1",
            "schema": "schema1",
            "table": "table1",
            "column": "email",
            "classification": "pi",
            "type": "pii"
        }
    ]

def test_apply_overrides_with_loop_column_only(sample_df, column_only_pi_csv_dict, pi_config):
    """Test apply_overrides_with_loop with column-only specification."""
    result_df = apply_overrides_with_loop(sample_df, column_only_pi_csv_dict, pi_config)
    
    ssn_rows = result_df.filter(col("column_name") == "ssn").collect()
    
    for row in ssn_rows:
        assert row.type == "PII"
    
    non_ssn_row = result_df.filter(col("column_name") != "ssn").first()
    assert non_ssn_row.type.upper() != "PII"

def test_apply_overrides_with_loop_mixed_specifications(sample_df, mixed_pi_csv_dict, pi_config):
    """Test apply_overrides_with_loop with mixed specifications."""
    result_df = apply_overrides_with_loop(sample_df, mixed_pi_csv_dict, pi_config)
    
    ssn_rows = result_df.filter(col("column_name") == "ssn").collect()
    for row in ssn_rows:
        assert row.type.upper() == "PII"
    
    email_row = result_df.filter(
        (col("catalog") == "catalog1") & 
        (col("schema") == "schema1") & 
        (col("table_name") == "table1") & 
        (col("column_name") == "email")
    ).first()
    
    assert email_row.type.upper() == "PII"
    
    other_row = result_df.filter(
        (col("column_name") != "ssn") & 
        (col("column_name") != "email")
    ).first()
    
    assert other_row.type.upper() != "PII"


