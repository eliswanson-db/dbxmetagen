from functools import reduce
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
    csv_df = csv_df.where(pd.notna(csv_df), None)
    csv_dict = csv_df.to_dict('records')
    print("csv_dict in overrides", csv_dict)
    spark = SparkSession.builder.getOrCreate()
    csv_spark_df = spark.createDataFrame(csv_df)
    nrows = csv_spark_df.count()
    print("nrows", nrows)

    if nrows == 0:
        return df
    elif nrows < 10000:
        df = apply_overrides_with_loop(df, csv_dict, config)
    else:
        raise ValueError("CSV file is too large. Please implement a more efficient method for large datasets.")
        #df = apply_overrides_with_joins(df, csv_spark_df, config)
    return df

def apply_overrides_with_loop(df, csv_dict, config):
    if not csv_dict:
        return df
        
    if config.mode == "pi":
        for row in csv_dict:
            catalog = row.get('catalog')
            schema = row.get('schema')
            table = row.get('table')
            column = row.get('column')
            classification_override = row['classification']
            type_override = row['type']

            if not column:
                continue
                
            try:
                condition = build_condition(df, table, column, schema, catalog)
                print("condition:", condition)
                df = df.withColumn('classification', when(condition, lit(classification_override)).otherwise(col('classification')))
                df = df.withColumn('type', when(condition, lit(type_override)).otherwise(col('type')))
            except ValueError as e:
                print(f"Skipping row due to: {e}")
    
    elif config.mode == "comment":
        for row in csv_dict:
            catalog = row.get('catalog')
            schema = row.get('schema')
            table = row.get('table')
            column = row.get('column')
            comment_override = row['comment']
            
            if not column:
                continue
                
            try:
                condition = build_condition(df, table, column, schema, catalog)
                df = df.withColumn('column_content', when(condition, lit(comment_override)).otherwise(col('column_content')))
            except ValueError as e:
                print(f"Skipping row due to: {e}")
    
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

    NOT FULLY IMPLEMENTED
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

def build_condition(df, table, column, schema, catalog):
    """
    Builds the condition for the DataFrame filtering.
    
    Only two parameter combinations are supported:
    1. Only column name is provided (all other parameters are None or empty)
    2. All parameters (catalog, schema, table, column) are provided
    
    Args:
        df (DataFrame): The input DataFrame.
        table (str): The table name.
        column (str): The column name.
        schema (str): The schema name.
        catalog (str): The catalog name.
    
    Returns:
        Column: The condition column.
    
    Raises:
        ValueError: If the combination of inputs is not one of the supported patterns.
    """
    table = table if table else None
    schema = schema if schema else None
    catalog = catalog if catalog else None
    
    if not column:
        raise ValueError("At least one parameter (column) must be provided.")
    
    only_column = column and not any([table, schema, catalog])
    all_params = all([column, table, schema, catalog])
    
    if only_column:
        return col('column_name') == column
    elif all_params:
        return reduce(lambda x, y: x & y, [
            col('column_name') == column,
            col('table_name') == table,
            col('schema') == schema,
            col('catalog') == catalog
        ])
    else:
        raise ValueError("Unsupported parameter combination. Either provide all parameters (catalog, schema, table, column) or only column name.")


def get_join_conditions(df: DataFrame, csv_spark_df: DataFrame) -> List[Column]:
    """
    Generates the join conditions for the DataFrame joins.

    Args:
        df (DataFrame): The input DataFrame.
        csv_spark_df (DataFrame): The CSV DataFrame.

    Returns:
        List[Column]: The list of join conditions.
    
    NOT FULLY IMPLEMENTED
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
