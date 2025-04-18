import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.overrides import build_condition

def test_build_condition_all_parameters(sample_df):
    """Test build_condition with all parameters provided."""
    condition = build_condition(sample_df, "table1", "column1", "schema1", "catalog1")
    filtered_df = sample_df.filter(condition)
    
    assert filtered_df.count() == 1
    row = filtered_df.collect()[0]
    assert row.catalog == "catalog1"
    assert row.schema == "schema1"
    assert row.table_name == "table1"
    assert row.column_name == "column1"

def test_build_condition_table_only(sample_df):
    """Test build_condition with only table parameter."""
    with pytest.raises(ValueError, match="Unsupported parameter combination"):
        build_condition(sample_df, "table1", None, None, None)

def test_build_condition_partial_parameters(sample_df):
    """Test build_condition with some parameters."""
    with pytest.raises(ValueError, match="Unsupported parameter combination"):
        build_condition(sample_df, "table1", None, "schema1", None)

def test_build_condition_no_match(sample_df):
    """Test build_condition with parameters that don't match any rows."""
    with pytest.raises(ValueError, match="Unsupported parameter combination"):
        build_condition(sample_df, "non_existent", None, None, None)

def test_build_condition_no_specification():
    """Test build_condition with no specification."""
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame([], "column_name: string")
    with pytest.raises(ValueError, match="At least one parameter .* must be provided"):
        build_condition(df, None, None, None, None)

def test_build_condition_column_only():
    """Test build_condition with column-only specification."""
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame([], "column_name: string")
    condition = build_condition(df, None, "test_column", None, None)
    # Don't assert exact string format, which is implementation-dependent
    assert "column_name" in str(condition)
    assert "test_column" in str(condition)
    
    # Empty strings should also work like None
    condition = build_condition(df, "", "test_column", "", "")
    assert "column_name" in str(condition)
    assert "test_column" in str(condition)

def test_build_condition_full_specification():
    """Test build_condition with full specification."""
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame([], "column_name: string, table_name: string, schema: string, catalog: string")
    condition = build_condition(df, "test_table", "test_column", "test_schema", "test_catalog")
    # Check for presence of key terms rather than exact format
    condition_str = str(condition)
    assert "column_name" in condition_str and "test_column" in condition_str
    assert "table_name" in condition_str and "test_table" in condition_str
    assert "schema" in condition_str and "test_schema" in condition_str
    assert "catalog" in condition_str and "test_catalog" in condition_str

def test_build_condition_partial_specification():
    """Test build_condition with partial specification (should fail)."""
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame([], "column_name: string, table_name: string")
    
    # Test various partial specifications
    with pytest.raises(ValueError):
        build_condition(df, "test_table", "test_column", None, None)
        
    with pytest.raises(ValueError):
        build_condition(df, None, "test_column", "test_schema", None)
        
    with pytest.raises(ValueError):
        build_condition(df, "test_table", "test_column", "test_schema", None)
