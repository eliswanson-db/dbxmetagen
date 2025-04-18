import pytest
from pyspark.sql import SparkSession

@pytest.fixture
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .master("local[*]") \
        .appName("unit_test") \
        .getOrCreate()

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    spark = SparkSession.builder.getOrCreate()
    data = [
        ("catalog1", "schema1", "table1", "column1", "type1", "classification1", "content1"),
        ("catalog1", "schema1", "table1", "email", "type1", "classification1", "content1"),
        ("catalog2", "schema2", "table2", "column2", "type2", "classification2", "content2"),
        ("catalog3", "schema3", "table3", "ssn", "type3", "classification3", "content3"),
    ]
    columns = ["catalog", "schema", "table_name", "column_name", "type", "classification", "column_content"]
    return spark.createDataFrame(data, columns)

@pytest.fixture
def mixed_pi_csv_dict():
    """Create a sample CSV dictionary for PI mode testing with mixed specifications."""
    return [
        {
            "column": "ssn",
            "classification": "PII",
            "type": "string"
        },
        {
            "catalog": "catalog1",
            "schema": "schema1",
            "table": "table1",
            "column": "email",
            "classification": "PII",
            "type": "string"
        }
    ]