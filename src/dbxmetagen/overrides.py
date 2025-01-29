import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.sql.column import Column
from typing import List, Dict
from src.dbxmetagen.config import MetadataConfig

def override_metadata_from_csv(df: DataFrame, csv_path: str, config: MetadataConfig) -> DataFrame:
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
    csv_dict = csv_df.to_dict('records')
    spark = SparkSession.builder.getOrCreate()
    csv_spark_df = spark.createDataFrame(csv_df)
    nrows = csv_spark_df.count()

    if nrows == 0:
        return df
    elif nrows < 10000:
        df = apply_overrides_with_loop(df, csv_dict, config)
    else:
        raise ValueError("CSV file is too large. Please implement a more efficient method for large datasets.")
        #df = apply_overrides_with_joins(df, csv_spark_df, config)

    return df

def apply_overrides_with_loop(df: DataFrame, csv_dict: Dict, config: MetadataConfig) -> DataFrame:
    """
    Applies overrides using a loop for small CSV files.

    Args:
        df (DataFrame): The input DataFrame.
        csv_spark_df (DataFrame): The CSV DataFrame.

    Returns:
        DataFrame: The updated DataFrame with overridden type and classification.
    """
    if config.mode == "pi":
        for row in csv_dict:
            catalog = row['catalog']
            schema = row['schema']
            table = row['table']
            column = row['column']
            classification_override = row['classification']
            type_override = row['type']
        condition = build_condition(df, table, column, schema, catalog)
        df = df.withColumn('classification', when(condition, lit(classification_override)).otherwise(col('classification')))
        df = df.withColumn('type', when(condition, lit(type_override)).otherwise(col('type')))
    elif config.mode == "comment":
        for row in csv_dict:
            catalog = row['catalog']
            schema = row['schema']
            table = row['table']
            column = row['column']
            comment_override = row['comment']            
        condition = build_condition(df, table, column, schema, catalog)
        df = df.withColumn('column_content', when(condition, lit(comment_override)).otherwise(col('column_content')))
    else:
        raise ValueError("Invalid mode provided.")
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
    if config.mode == "pi":
        df = df.join(csv_spark_df, join_conditions, "left_outer") \
            .withColumn('classification', when(col('classification_override').isNotNull(), col('classification_override')).otherwise(col('classification'))) \
            .withColumn('type', when(col('type_override').isNotNull(), col('type_override')).otherwise(col('type'))) \
            .drop('classification_override', 'type_override')
    elif config.mode == "comment":
        df = df.join(csv_spark_df, join_conditions, "left_outer") \
            .withColumn('column_content', when(col('comment_override').isNotNull(), col('comment_override')).otherwise(col('column_content'))) \
            .drop('comment_override')
    else:
        raise ValueError("Invalid mode provided.")

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
    
    if column:
        column_condition = (col('column_name') == column)
    if table:
        table_condition = (col('table_name') == table)
    if schema:
        schema_condition = (col('schema') == schema)
    if catalog:
        catalog_condition = (col('catalog') == catalog)

    if column and not table and not schema and not catalog:
        return column_condition
    if column and table and schema and catalog:
        return column_condition & table_condition & schema_condition & catalog_condition
    

def get_join_conditions(df: DataFrame, csv_spark_df: DataFrame) -> List[Column]:
    """
    Generates the join conditions for the DataFrame joins.

    Args:
        df (DataFrame): The input DataFrame.
        csv_spark_df (DataFrame): The CSV DataFrame.

    Returns:
        List[Column]: The list of join conditions.
    """
    join_condition = None

    if 'column' in csv_spark_df.columns:
        join_condition = (df['column_name'] == csv_spark_df['column'])
    if 'table' in csv_spark_df.columns:
        table_condition = (df['table_name'] == csv_spark_df['table'])
        join_condition = join_condition & table_condition if join_condition else table_condition
    if 'schema' in csv_spark_df.columns:
        schema_condition = (df['schema'] == csv_spark_df['schema'])
        join_condition = join_condition & schema_condition if join_condition else schema_condition
    if 'catalog' in csv_spark_df.columns:
        catalog_condition = (df['catalog'] == csv_spark_df['catalog'])
        join_condition = join_condition & catalog_condition if join_condition else catalog_condition
    return join_condition
