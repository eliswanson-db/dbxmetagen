import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.sql.column import Column
from typing import List

def override_metadata_from_csv(df: DataFrame, csv_path: str) -> DataFrame:
    """
    Overrides the type and classification in the DataFrame based on the CSV file.
    This would need to be optimized if a customer has a large number of overrides.

    Args:
        df (DataFrame): The input DataFrame.
        csv_path (str): The path to the CSV file.

    Returns:
        DataFrame: The updated DataFrame with overridden type and classification.
    """
    csv_df = pd.read_csv(csv_path)
    spark = SparkSession.builder.getOrCreate()
    csv_spark_df = spark.createDataFrame(csv_df)
    nrows = csv_spark_df.count()

    if nrows == 0:
        return df
    elif nrows < 1000:
        df = apply_overrides_with_loop(df, csv_spark_df)
    else:
        df = apply_overrides_with_joins(df, csv_spark_df)

    return df

def apply_overrides_with_loop(df: DataFrame, csv_spark_df: DataFrame) -> DataFrame:
    """
    Applies overrides using a loop for small CSV files.

    Args:
        df (DataFrame): The input DataFrame.
        csv_spark_df (DataFrame): The CSV DataFrame.

    Returns:
        DataFrame: The updated DataFrame with overridden type and classification.
    """
    for row in csv_spark_df.collect():
        catalog = row['catalog']
        schema = row['schema']
        table = row['table']
        column = row['column'] if 'column' in row else None
        type_override = row['pi_classification']

        condition = build_condition(df, table, column, schema, catalog)
        print("Condition", condition)
        df = df.withColumn('type', when(condition, lit(type_override)).otherwise(col('type')))
        df = df.withColumn('classification', when(condition, lit(type_override)).otherwise(col('classification')))
    return df

def apply_overrides_with_joins(df: DataFrame, csv_spark_df: DataFrame) -> DataFrame:
    """
    Applies overrides using joins for large CSV files.

    Args:
        df (DataFrame): The input DataFrame.
        csv_spark_df (DataFrame): The CSV DataFrame.

    Returns:
        DataFrame: The updated DataFrame with overridden type and classification.
    """
    join_conditions = get_join_conditions(df, csv_spark_df)

    for condition in join_conditions:
        df = df.join(csv_spark_df, condition, 'left') \
            .withColumn('type', when(csv_spark_df['column'].isNotNull(), csv_spark_df['pi_classification']).otherwise(df['type'])) \
            .withColumn('classification', when(csv_spark_df['column'].isNotNull(), csv_spark_df['pi_classification']).otherwise(df['classification'])) \
            .drop(csv_spark_df['pi_classification'])
    return df

def build_condition(df: DataFrame, table: str, column: str, schema: str, catalog: str) -> Column:
    """
    Builds the condition for the DataFrame filtering.

    Args:
        df (DataFrame): The input DataFrame.
        table (str): The table name.
        column (str): The column name.
        schema (str): The schema name.
        catalog (str): The catalog name.

    Returns:
        Column: The condition column.
    """
    condition = (col('table') == table)
    if column:
        condition = condition & (col('column') == column)
    if schema:
        condition = condition & (col('schema') == schema)
    if catalog:
        condition = condition & (col('catalog') == catalog)
    return condition

def get_join_conditions(df: DataFrame, csv_spark_df: DataFrame) -> List[Column]:
    """
    Generates the join conditions for the DataFrame joins.

    Args:
        df (DataFrame): The input DataFrame.
        csv_spark_df (DataFrame): The CSV DataFrame.

    Returns:
        List[Column]: The list of join conditions.
    """
    join_condition_full = [
        (df['catalog'] == csv_spark_df['catalog']) & 
        (df['schema'] == csv_spark_df['schema']) & 
        (df['table'] == csv_spark_df['table']) & 
        (df['column'] == csv_spark_df['column'])
    ]

    join_condition_schema_table_column = [
        (df['schema'] == csv_spark_df['schema']) & 
        (df['table'] == csv_spark_df['table']) & 
        (df['column'] == csv_spark_df['column'])
    ]

    join_condition_table_column = [
        (df['table'] == csv_spark_df['table']) & 
        (df['column'] == csv_spark_df['column'])
    ]

    join_condition_table = [
        (df['table'] == csv_spark_df['table'])
    ]

    return [join_condition_full, join_condition_schema_table_column, join_condition_table_column, join_condition_table]