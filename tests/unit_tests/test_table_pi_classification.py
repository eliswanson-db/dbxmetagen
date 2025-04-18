import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import collect_set
from src.dbxmetagen.processing import determine_table_classification
from src.dbxmetagen.overrides import build_condition

@pytest.fixture
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .appName("UnitTest") \
        .master("local[1]") \
        .getOrCreate()

def create_test_df(spark, data):
    """Helper function to create a test DataFrame with 'type' column."""
    schema = StructType([StructField("type", StringType(), True)])
    return spark.createDataFrame(data, schema)

def test_determine_table_classification_none(spark):
    """Test when classification is None."""
    data = [("None",)]
    df = create_test_df(spark, data)
    result = determine_table_classification(df)
    assert result == "None"

def test_determine_table_classification_pii(spark):
    """Test when classification is pii."""
    # Test case 1: Only "pii"
    data1 = [("pii",)]
    df1 = create_test_df(spark, data1)
    result1 = determine_table_classification(df1)
    assert result1 == "pii"
    
    # Test case 2: "pii" and "None"
    data2 = [("pii",), ("None",)]
    df2 = create_test_df(spark, data2)
    result2 = determine_table_classification(df2)
    assert result2 == "pii"

def test_determine_table_classification_pci(spark):
    """Test when classification is pci."""
    # Test case: "pci" without "phi" or "medical_information"
    data = [("pci",), ("other",)]
    df = create_test_df(spark, data)
    result = determine_table_classification(df)
    assert result == "pci"

def test_determine_table_classification_medical_information(spark):
    """Test when classification is medical_information."""
    # Test case 1: Only "medical_information"
    data1 = [("medical_information",)]
    df1 = create_test_df(spark, data1)
    result1 = determine_table_classification(df1)
    assert result1 == "medical_information"
    
    # Test case 2: "medical_information" and "None"
    data2 = [("medical_information",), ("None",)]
    df2 = create_test_df(spark, data2)
    result2 = determine_table_classification(df2)
    assert result2 == "medical_information"

def test_determine_table_classification_phi(spark):
    """Test when classification is phi."""
    # Test case 1: Only "phi"
    data1 = [("phi",)]
    df1 = create_test_df(spark, data1)
    result1 = determine_table_classification(df1)
    assert result1 == "phi"
    
    # Test case 2: "phi" and "None"
    data2 = [("phi",), ("None",)]
    df2 = create_test_df(spark, data2)
    result2 = determine_table_classification(df2)
    assert result2 == "phi"
    
    # Test case 3: "phi" and "pii"
    data3 = [("phi",), ("pii",)]
    df3 = create_test_df(spark, data3)
    result3 = determine_table_classification(df3)
    assert result3 == "phi"
    
    # Test case 4: "phi", "pii", and "None"
    data4 = [("phi",), ("pii",), ("None",)]
    df4 = create_test_df(spark, data4)
    result4 = determine_table_classification(df4)
    assert result4 == "phi"
    
    # Test case 5: "pii" and "medical_information"
    data5 = [("pii",), ("medical_information",)]
    df5 = create_test_df(spark, data5)
    result5 = determine_table_classification(df5)
    assert result5 == "phi"

def test_determine_table_classification_all(spark):
    """Test when classification is all."""
    # Test case 1: "pci" and "phi"
    data1 = [("pci",), ("phi",)]
    df1 = create_test_df(spark, data1)
    result1 = determine_table_classification(df1)
    assert result1 == "all"
    
    # Test case 2: "pci" and "medical_information"
    data2 = [("pci",), ("medical_information",)]
    df2 = create_test_df(spark, data2)
    result2 = determine_table_classification(df2)
    assert result2 == "all"
    
    # Test case 3: "pci", "phi", and "medical_information"
    data3 = [("pci",), ("phi",), ("medical_information",)]
    df3 = create_test_df(spark, data3)
    result3 = determine_table_classification(df3)
    assert result3 == "all"

def test_determine_table_classification_unknown(spark):
    """Test when classification is unknown."""
    # Create test data with values that don't match any condition
    data = [("unknown_type",)]
    df = create_test_df(spark, data)
    
    # Test the function
    result = determine_table_classification(df)
    assert result == "Unknown"

def test_determine_table_classification_edge_cases(spark):
    """Test edge cases."""
    # Test with empty DataFrame
    schema = StructType([StructField("type", StringType(), True)])
    empty_df = spark.createDataFrame([], schema)
    
    # This should return "None" since collect_set on an empty DataFrame returns an empty array
    result = determine_table_classification(empty_df)
    assert result == "None"

def test_determine_table_classification_complex_combinations(spark):
    """Test more complex combinations of classifications."""
    # Test case: "pii", "pci", "other"
    data = [("pii",), ("pci",),]
    df = create_test_df(spark, data)
    result = determine_table_classification(df)
    assert result == "pci"  # Should be "pci" since no "phi" or "medical_information"
    
    # Test case: "phi", "pii", "other"
    data2 = [("phi",), ("pii",),]
    df2 = create_test_df(spark, data2)
    result2 = determine_table_classification(df2)
    assert result2 == "phi"
