import pytest
from pyspark.sql import SparkSession

def test_simple(sample_df):
    assert sample_df.count() == 4