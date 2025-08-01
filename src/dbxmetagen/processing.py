import logging
import os
import re
import json
import random
import time
from abc import ABC
from datetime import datetime
from typing import List, Dict, Any, Literal, Tuple
import traceback

import csv
from shutil import copyfile
import mlflow
import nest_asyncio
import pandas as pd
from pydantic import BaseModel, Field, Extra, ValidationError, ConfigDict
from pydantic.dataclasses import dataclass
from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql.functions import (
    col, struct, to_timestamp, current_timestamp, lit, when, 
    sum as spark_sum, max as spark_max, concat_ws, collect_list, 
    collect_set, udf, trim, split, expr, split_part
)
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, FloatType, DoubleType

from openai import OpenAI
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from mlflow.types.llm import TokenUsageStats, ChatResponse

from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.sampling import determine_sampling_ratio
from src.dbxmetagen.prompts import Prompt, PIPrompt, CommentPrompt, PromptFactory
from src.dbxmetagen.error_handling import exponential_backoff, validate_csv
from src.dbxmetagen.comment_summarizer import TableCommentSummarizer
from src.dbxmetagen.metadata_generator import (
    Response, PIResponse, CommentResponse, PIColumnContent, MetadataGeneratorFactory, 
    PIIdentifier, MetadataGenerator, CommentGenerator
)
from src.dbxmetagen.overrides import (override_metadata_from_csv, apply_overrides_with_loop, 
    apply_overrides_with_joins, build_condition, get_join_conditions
)
from src.dbxmetagen.parsing import cleanse_sql_comment

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DDLGenerator(ABC):
    def __init__(self):
        pass

class Input(BaseModel):
    ### Currently not implemented.
    model_config = ConfigDict(extra="forbid")

    table_name: str

    @classmethod
    def from_df(cls, df: DataFrame) -> Dict[str, Any]:
        return {
            "table_name": f"{catalog_name}.{schema_name}.{table_name}",
            "column_contents": cls.df.toPandas().to_dict(orient='list')
        }

def tag_table(table_name: str, tags: Dict[str, str]) -> None:
    """
    Tags a table with the provided tags.

    Args:
        table_name (str): The name of the table to tag.
        tags (Dict[str, str]): A dictionary of tags to apply to the table.
    """
    spark = SparkSession.builder.getOrCreate()
    for key, value in tags.items():
        spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('{key}' = '{value}');")


def write_to_log_table(log_data: Dict[str, Any], log_table_name: str) -> None:
    """
    Writes log data to a specified log table.

    Args:
        log_data (Dict[str, Any]): The log data to write.
        log_table_name (str): The name of the log table.
    """
    spark = SparkSession.builder.getOrCreate()
    log_df = spark.createDataFrame([log_data])
    log_df.write.format("delta").option("mergeSchema", "true").mode("append").saveAsTable(log_table_name)


def count_df_columns(df: DataFrame) -> int:
    """
    Count the number of columns in a spark dataframe and return.
    """
    return len(df.columns)


def chunk_df(df: DataFrame, columns_per_call: int = 5) -> List[DataFrame]:
    """
    Splits a DataFrame into multiple DataFrames, each containing a specified number of columns.

    Args:
        df (DataFrame): The input DataFrame to split.
        columns_per_chunk (int, optional): The number of columns per chunk. Defaults to 10.

    Returns:
        List[DataFrame]: A list of DataFrames, each containing a subset of the original columns.
    """
    col_names = df.columns
    n_cols = count_df_columns(df)
    num_chunks = (n_cols + columns_per_call - 1) // columns_per_call # Calculate the number of chunks

    dataframes = []
    for i in range(num_chunks):
        chunk_col_names = col_names[i * columns_per_call:(i + 1) * columns_per_call]
        chunk_df = df.select(chunk_col_names)
        dataframes.append(chunk_df)

    return dataframes


def get_extended_metadata_for_column(config, table_name, column_name):
    spark = SparkSession.builder.getOrCreate()
    query = f"""DESCRIBE EXTENDED {config.catalog_name}.{config.schema_name}.{table_name} `{column_name}`;"""
    return spark.sql(query)


def sample_df(df: DataFrame, nrows: int, sample_size: int = 5) -> DataFrame:
    """
    Sample dataframe to a given size and filter out rows with lots of nulls.

    Args:
        df (DataFrame): The DataFrame to be analyzed.
        nrows (int): number of rows in dataframe
        sample_size (int): The number of rows to sample.

    Returns:
        DataFrame: A DataFrame with columns to generate metadata for.
    """
    if nrows < sample_size:
        print(f"Not enough rows for a proper sample. Continuing with inference with {nrows} rows...")
        return df.limit(sample_size)

    larger_sample = sample_size * 100
    sampling_ratio = determine_sampling_ratio(nrows, larger_sample)
    sampled_df = df.sample(withReplacement=False, fraction=sampling_ratio)
    null_counts_per_row = sampled_df.withColumn(
        "null_count", sum(when(col(c).isNull(), 1).otherwise(0) for c in sampled_df.columns)
    )
    threshold = len(sampled_df.columns) // 2
    filtered_df = null_counts_per_row.filter(col("null_count") < threshold).drop("null_count")
    result_rows = filtered_df.count()
    if result_rows < sample_size:
        print("Not enough non-NULL rows, returning available rows, despite large proportion of NULLs. Result rows:", result_rows, "vs sample size:", sample_size)
        return df.limit(sample_size)

    print(f"Filtering {result_rows} result rows down to {sample_size} rows...")
    return filtered_df.limit(sample_size)


def append_table_row(config: MetadataConfig, rows: List[Row], full_table_name: str, response: Dict[str, Any], tokenized_full_table_name: str) -> List[Row]:
    """
    Appends a table row to the list of rows.

    Args:
        rows (List[Row]): The list of rows to append to.
        full_table_name (str): The full name of the table.
        response (Dict[str, Any]): The response dictionary containing table information.

    Returns:
        List[Row]: The updated list of rows with the new table row appended.
    """
    row = Row(
        table=full_table_name,
        tokenized_table=tokenized_full_table_name,
        ddl_type='table',
        column_name='None',
        column_content=response.table,
    )
    rows.append(row)
    return rows


def append_column_rows(config: MetadataConfig, rows: List[Row], full_table_name: str, response: Dict[str, Any], tokenized_full_table_name: str) -> List[Row]:
    """
    Appends column rows to the list of rows.

    Args:
        rows (List[Row]): The list of rows to append to.
        full_table_name (str): The full name of the table.
        response (Dict[str, Any]): The response dictionary containing column information.

    Returns:
        List[Row]: The updated list of rows with the new column rows appended.
    """
    for column_name, column_content in zip(response.columns, response.column_contents):
        if (isinstance(column_content, dict) or isinstance(column_content, PIColumnContent)) and config.mode == "pi":
            if isinstance(column_content, PIColumnContent):
                column_content = column_content.model_dump()
            row = Row(
                table=full_table_name,
                tokenized_table=tokenized_full_table_name,
                ddl_type='column',
                column_name=column_name,
                **column_content
            )
        elif isinstance(column_content, str) and config.mode == "comment":
            row = Row(
                table=full_table_name,
                tokenized_table=tokenized_full_table_name,
                ddl_type='column',
                column_name=column_name,
                column_content=column_content
            )
        else:
            raise ValueError("Invalid column contents type, should be dict or string.")
        rows.append(row)
    return rows


def define_row_schema(config):
    if config.mode == "pi":
        schema = StructType([
            StructField("table", StringType(), True),
            StructField("tokenized_table", StringType(), True),
            StructField("ddl_type", StringType(), True),
            StructField("column_name", StringType(), True),
            StructField("classification", StringType(), True),
            StructField("type", StringType(), True),
            StructField("confidence", DoubleType(), True),
        ])
    elif config.mode == "comment":
        schema = StructType([
            StructField("table", StringType(), True),
            StructField("tokenized_table", StringType(), True),
            StructField("ddl_type", StringType(), True),
            StructField("column_name", StringType(), True),
            StructField("column_content", StringType(), True),
        ])
    return schema


def rows_to_df(rows: List[Row], config: MetadataConfig) -> DataFrame:
    """
    Converts a list of rows to a Spark DataFrame.

    Args:
        rows (List[Row]): The list of rows to convert.

    Returns:
        DataFrame: The Spark DataFrame created from the list of rows.
    """
    spark = SparkSession.builder.getOrCreate()
    if len(rows) == 0:
        return None
    else:
        schema = define_row_schema(config)
        df = spark.createDataFrame(rows, schema).withColumn("_created_at", current_timestamp())
        return df


def add_ddl_to_column_comment_df(df: DataFrame, ddl_column: str) -> DataFrame:
    """
    Adds a DDL statement to a DataFrame for column comment.

    Args:
        df (DataFrame): The DataFrame to add the DDL statement to.
        ddl_column (str): The name of the DDL column.

    Returns:
        DataFrame: The updated DataFrame with the DDL statement added.
    """
    return df.withColumn(ddl_column, generate_column_comment_ddl('tokenized_table', 'column_name', 'column_content'))


def add_ddl_to_table_comment_df(df: DataFrame, ddl_column: str) -> DataFrame:
    """
    Adds a DDL statement to a DataFrame for table comment.

    Args:
        df (DataFrame): The DataFrame to add the DDL statement to.
        ddl_column (str): The name of the DDL column.

    Returns:
        DataFrame: The updated DataFrame with the DDL statement added.
    """
    return df.withColumn(ddl_column, generate_table_comment_ddl('tokenized_table', 'column_content'))


def add_table_ddl_to_pi_df(df: DataFrame, ddl_column: str) -> DataFrame:
    """
    Adds a DDL statement to a DataFrame for PI information.

    Args:
        df (DataFrame): The DataFrame to add the DDL statement to.
        ddl_column (str): The name of the DDL column.

    Returns:
        DataFrame: The updated DataFrame with the DDL statement added.
    """
    return df.withColumn(ddl_column, generate_table_pi_information_ddl('tokenized_table', 'classification', 'type'))


def add_column_ddl_to_pi_df(config, df: DataFrame, ddl_column: str) -> DataFrame:
    """
    Adds a DDL statement to a DataFrame for PI information.

    Args:
        df (DataFrame): The DataFrame to add the DDL statement to.
        ddl_column (str): The name of the DDL column.

    Returns:
        DataFrame: The updated DataFrame with the DDL statement added.
    """
    if not config.tag_none_fields:
        df = df.filter(col("type") != "None")
    df = df.withColumn(ddl_column, generate_pi_information_ddl('tokenized_table', 'column_name', 'classification', 'type'))
    return df



def get_current_user() -> str:
    """
    Retrieves the current user.

    Returns:
        str: The current user.
    """
    spark = SparkSession.builder.getOrCreate()
    return spark.sql("SELECT current_user()").collect()[0][0]


def df_to_sql_file(df: DataFrame, catalog_name: str, dest_schema_name: str, table_name: str, volume_name: str, sql_column: str, filename: str) -> str:
    """
    Writes a DataFrame to a SQL file.

    Args:
        df (DataFrame): The DataFrame to write.
        catalog_name (str): The catalog name.
        dest_schema_name (str): The destination schema name.
        table_name (str): The table name.
        volume_name (str): The volume name.
        sql_column (str): The name of the SQL column.
        filename (str): The name of the file.

    Returns:
        str: The path to the SQL file.
    """
    print("Converting dataframe to SQL file...")
    selected_column_df = df.select(sql_column)
    column_list = [row[sql_column] for row in selected_column_df.collect()]
    uc_volume_path = f"/Volumes/{catalog_name}/{dest_schema_name}/{filename}.sql"
    with open(uc_volume_path, 'w') as file:
        for item in column_list:
            file.write(f"{item}\n")
    return uc_volume_path


class DataFrameToExcelError(Exception):
    """Custom exception for DataFrame to Excel export errors."""


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensures that the specified directory exists, creating it if necessary.

    Args:
        directory_path (str): The directory to check or create.

    Raises:
        DataFrameToExcelError: If directory creation fails.
    """
    try:
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
            logger.info(f"Created directory: {directory_path}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise DataFrameToExcelError(f"Directory creation failed: {e}")


def df_column_to_excel_file(
    df: pd.DataFrame,
    filename: str,
    base_path: str,
    excel_column: str
) -> str:
    """
    Exports a specified column from a DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        filename (str): The name of the output Excel file (without extension).
        volume_name (str): Volume name (used in path).
        excel_column (str): The column to export.

    Returns:
        str: The path to the created Excel file.

    Raises:
        DataFrameToExcelError: If export fails.
    """
    logger.info("Starting export of DataFrame column to Excel.")
    try:
        if excel_column not in df.columns:
            logger.error(f"Column '{excel_column}' not found in DataFrame.")
            raise DataFrameToExcelError(f"Column '{excel_column}' does not exist in DataFrame.")

        output_dir = base_path
        ensure_directory_exists(output_dir)
        excel_file_path = os.path.join(output_dir, f"{filename}.xlsx")
        local_path = f"/local_disk0/tmp/{filename}.xlsx"
        df[[excel_column]].to_excel(local_path, index=False, engine="openpyxl")
        copyfile(local_path, excel_file_path)
        logger.info(f"Successfully wrote column '{excel_column}' to Excel file: {excel_file_path}")

        if not os.path.isfile(excel_file_path):
            logger.error(f"Excel file was not created: {excel_file_path}")
            raise DataFrameToExcelError(f"Excel file was not created: {excel_file_path}")

        print(f"Excel file created at: {excel_file_path}")
        return excel_file_path

    except Exception as e:
        logger.error(f"Error exporting DataFrame to Excel: {e}")
        raise DataFrameToExcelError(f"Failed to export DataFrame to Excel: {e}")


def populate_log_table(df, config, current_user, base_path):
    return (df.withColumn("current_user", lit(current_user))
                .withColumn("model", lit(config.model))
                .withColumn("sample_size", lit(config.sample_size))
                .withColumn("max_tokens", lit(config.max_tokens))
                .withColumn("temperature", lit(config.temperature))
                .withColumn("columns_per_call", lit(config.columns_per_call))
                .withColumn("status", lit("No Volume specified..."))
            )
    

def get_control_table(config: MetadataConfig) -> str:
    """
    Returns the control table name based on the provided configuration.

    Args:
        config (MetadataConfig): Configuration object containing setup and model parameters.

    Returns:
        str: The control table name.
    """
    spark = SparkSession.builder.getOrCreate()
    if config.job_id and config.cleanup_control_table == "true":
        formatted_control_table = config.control_table.format(sanitize_email(get_current_user()))+str(config.job_id)
    else:
        formatted_control_table = config.control_table.format(sanitize_email(get_current_user()))
    return formatted_control_table



def mark_as_deleted(table_name: str, config: MetadataConfig) -> None:
    """
    Updates the _deleted_at and _updated_at columns to the current timestamp for the specified table.

    Args:
        table_name (str): The name of the table to update.
        config (MetadataConfig): Configuration object containing setup and model parameters.
    """
    spark = SparkSession.builder.getOrCreate()
    formatted_control_table = get_control_table(config)
    control_table = f"{config.catalog_name}.{config.schema_name}.{formatted_control_table}"
    update_query = f"""
    UPDATE {control_table}
    SET _deleted_at = current_timestamp(),
        _updated_at = current_timestamp()
    WHERE table_name = '{table_name}'
    """
    spark.sql(update_query)
    print(f"Marked {table_name} as deleted in the control table...")


def run_log_table_ddl(config):
    spark = SparkSession.builder.getOrCreate()
    if config.mode == "comment":
        spark.sql(f"""CREATE TABLE IF NOT EXISTS {config.catalog_name}.{config.schema_name}.{config.mode}_metadata_generation_log (
            table STRING, 
            tokenized_table STRING, 
            ddl_type STRING, 
            column_name STRING, 
            _created_at TIMESTAMP,
            column_content STRING, 
            catalog STRING, 
            schema STRING, 
            table_name STRING, 
            ddl STRING, 
            current_user STRING, 
            model STRING, 
            sample_size INT, 
            max_tokens INT, 
            temperature DOUBLE, 
            columns_per_call INT, 
            status STRING
        )"""
    )
    elif config.mode == "pi":
        spark.sql(f"""CREATE TABLE IF NOT EXISTS {config.catalog_name}.{config.schema_name}.{config.mode}_metadata_generation_log (
            table STRING, 
            tokenized_table STRING, 
            ddl_type STRING, 
            column_name STRING,
            _created_at TIMESTAMP,
            classification STRING,
            type STRING,
            confidence DOUBLE,
            catalog STRING, 
            schema STRING, 
            table_name STRING, 
            ddl STRING, 
            current_user STRING, 
            model STRING, 
            sample_size INT, 
            max_tokens INT, 
            temperature DOUBLE, 
            columns_per_call INT, 
            status STRING
        )"""
    )
    else: 
        raise ValueError(f"Invalid mode: {config.mode}, please choose pi or comment.")


def output_df_pandas_to_tsv(df, output_file):
    pandas_df = df.toPandas()
    write_header = not os.path.exists(output_file)
    pandas_df.to_csv(
        output_file,
        sep='\t',
        header=write_header,
        index=False,
        mode='a'
    )


def _export_table_to_tsv(df, config):
    """
    Reads a table from Databricks, writes it as a TSV file to a volume, and drops the original table.

    Args:
        df: Spark DataFrame to export.
        config: Configuration object containing catalog_name, schema_name, mode, current_user, volume_name, etc.

    Returns:
        str: Table name if operation was successful, False otherwise.
    """
    try:
        required_attrs = ['catalog_name', 'schema_name', 'mode', 'volume_name']
        for attr in required_attrs:
            if not hasattr(config, attr) or not getattr(config, attr):
                raise ValueError(f"Missing or empty required config attribute: {attr}")

        if df is None:
            raise ValueError("Input DataFrame is None.")
        if not hasattr(df, 'count') or not callable(getattr(df, 'count')):
            raise TypeError("Input is not a valid Spark DataFrame.")

        date = datetime.now().strftime("%Y%m%d")
        if not hasattr(config, 'log_timestamp') or not config.log_timestamp:
            config.log_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        timestamp = config.log_timestamp

        filename = f"review_metadata_{config.mode}_{config.log_timestamp}.tsv"
        local_path = f"/local_disk0/tmp/{filename}"
        current_user = sanitize_email(get_current_user())
        table_name = f"{config.catalog_name}.{config.schema_name}.{config.mode}_temp_metadata_generation_log_{current_user}"
        volume_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}"
        folder_path = f"{volume_path}/{current_user}/{date}/exportable_run_logs/"
        output_file = f"{folder_path}review_metadata_{config.mode}_{config.log_timestamp}.tsv"

        try:
            create_folder_if_not_exists(folder_path)
        except Exception as e:
            print(f"Error creating output directory '{folder_path}': {str(e)}")
            return False

        try:
            row_count = df.count()
        except Exception as e:
            print(f"Error counting rows in DataFrame: {str(e)}")
            return False
        if row_count == 0:
            print("Warning: Table is empty")

        print(f"Writing to TSV file: {output_file}")

        try:
            df.write.mode("append").saveAsTable(table_name)
        except Exception as e:
            print(f"Error writing Spark DataFrame to table '{table_name}': {str(e)}")
            return False

        try:
            output_df_pandas_to_tsv(df, local_path)
            copyfile(local_path, output_file)
        except Exception as e:
            print(f"Error writing DataFrame to TSV file '{output_file}': {str(e)}")
            return False

        print("Export completed successfully...")
        return table_name

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        return False
    except TypeError as te:
        print(f"TypeError: {str(te)}")
        return False
    except Exception as e:
        print(f"Unexpected error during full log export process: {str(e)}")
        return False
    
    
class ExportError(Exception):
    """Custom exception for export errors."""


def create_folder_if_not_exists(path: str) -> None:
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise ExportError(f"Directory creation failed: {e}")


def export_df_to_excel(df: pd.DataFrame, output_file: str, export_folder: str) -> None:
    try:
        local_path = f"/local_disk0/tmp/{output_file}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            df.to_excel(local_path, index=False)
        else:
            with pd.ExcelWriter(local_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name='Sheet1', startrow=writer.sheets['Sheet1'].max_row, header=False, index=False)
        copyfile(local_path, os.path.join(export_folder, output_file))
        logger.info(f"Excel file created at: {output_file}")
    except Exception as e:
        logger.error(f"Failed to export DataFrame to Excel: {e}")
        raise ExportError(f"Failed to export DataFrame to Excel: {e}")


def _export_table_to_excel(df: Any, config: Any) -> str:
    """
    Reads a table from Databricks, writes it as an Excel file to a volume, and drops the original table.

    Args:
        df: DataFrame to export (Spark or pandas)
        config: Configuration object containing catalog_name, schema_name, mode, current_user, and volume_name

    Returns:
        str: The path to the Excel file if successful

    Raises:
        ExportError: If export fails
    """
    date = datetime.now().strftime("%Y%m%d")
    if not hasattr(config, 'log_timestamp') or not config.log_timestamp:
        config.log_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp = config.log_timestamp

    try:
        current_user = sanitize_email(get_current_user())
        table_name = f"{config.catalog_name}.{config.schema_name}.{config.mode}_temp_metadata_generation_log_{current_user}"
        volume_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}"
        export_folder = f"{volume_path}/{current_user}/{date}/exportable_run_logs/"
        create_folder_if_not_exists(export_folder)
        output_filename = f"review_metadata_{config.mode}_{config.log_timestamp}.xlsx"
        output_file = f"{export_folder}{output_filename}"

        if hasattr(df, 'count') and callable(df.count):
            if df.count() == 0:
                print("Warning: Table is empty")
                logger.warning("Table is empty")
        elif isinstance(df, pd.DataFrame) and df.empty:
            print("Warning: Table is empty")
            logger.warning("Table is empty")

        if hasattr(df, 'toPandas') and callable(df.toPandas):
            pdf = df.toPandas()
        elif isinstance(df, pd.DataFrame):
            pdf = df
        else:
            logger.error("Unsupported DataFrame type")
            raise ExportError("Unsupported DataFrame type")

        print(f"Writing to Excel file: {output_filename}")
        logger.info(f"Writing to Excel file: {output_filename}")
        export_df_to_excel(pdf, output_filename, export_folder)

        print("Export completed successfully")
        logger.info("Export completed successfully")
        return output_file

    except Exception as e:
        print(f"Error during full log export process: {str(e)}")
        logger.error(f"Error during full log export process: {str(e)}")
        raise ExportError(f"Error during full log export process: {str(e)}")


def log_metadata_generation(df: DataFrame, config: MetadataConfig, table_name: str, volume_name: str) -> None:
    run_log_table_ddl(config)
    df.write.mode('append').option("mergeSchema", "true").saveAsTable(f"{config.catalog_name}.{config.schema_name}.{config.mode}_metadata_generation_log")
    mark_as_deleted(table_name, config)


def set_classification_to_null(df: DataFrame, config: MetadataConfig) -> DataFrame:
    if config.mode == "pi":
        df = df.withColumn("classification", lit(None))
    return df


def set_protected_classification(df: DataFrame, config: MetadataConfig) -> DataFrame:
    if df is None:
        return None
        
    if config.mode == "pi":
        df = df.withColumn(
            "classification", 
            when(
                (df['type'] == "pii") | 
                (df['type'] == "pci") | 
                (df['type'] == "medical_information") | 
                (df['type'] == "phi"), 
                lit("protected")
            ).otherwise(lit(None))
        )
    return df


def replace_medical_information_with_phi(df: DataFrame, config: MetadataConfig) -> DataFrame:
    if df is None:
        return None
    if config.mode == "pi" and config.disable_medical_information_value:
        df = df.withColumn(
            "type", 
            when(
                (df['type'] == "medical_information"),
                lit("phi")
            ).otherwise(lit(None))
        )
    return df


def filter_and_write_ddl(df: DataFrame,                          
                         config: MetadataConfig,
                         base_path: str,
                         full_table_name: str,
                         current_user: str,
                         current_date: str
                         ) -> DataFrame:
    """Filter the DataFrame based on the table name and write the DDL statements to a SQL file.
    Args:
        df (DataFrame): The DataFrame containing the DDL statements.
        config: MetadataConfig
        base_path: str
        full_table_name: str
        current_user: str
        current_date: str
    """
    print("Filtering dataframe based on table name to write DDL to SQL file in volume...")
    df = df.filter(df['table'] == full_table_name)
    ddl_statements = df.select("ddl").collect()
    table_name = re.sub(r'[^\w\s/]', '_', full_table_name)
    file_root = f"{table_name}_{config.mode}"
    write_ddl_to_volume(file_root, base_path, ddl_statements, config.ddl_output_format)
    df = df.withColumn("status", lit("Success"))
    try:
        write_ddl_to_volume(file_root, base_path, ddl_statements, config.ddl_output_format)
        df = df.withColumn("status", lit("Success"))
    except ValueError as ve:
        print(f"Error: {ve}")
        df = df.withColumn("status", lit("Failed: Invalid output format"))
    except IOError as ioe:
        print(f"Error writing DDL to volume: {ioe}. Check if Volume exists and if your permissions are correct.")
        df = df.withColumn("status", lit("Failed: IO Error"))
    except Exception as e:
        print(f"Unexpected error: {e}")
        df = df.withColumn("status", lit("Failed: Unexpected error"))
    finally:
        log_metadata_generation(df, config, full_table_name, base_path)
        if config.reviewable_output_format == "excel":
            _export_table_to_excel(df, config)
        elif config.reviewable_output_format == "tsv":
            _export_table_to_tsv(df, config)
        else:
            raise ValueError("Invalid output format for reviewable_output_format. Please choose either 'excel' or 'tsv'.")


def create_folder_if_not_exists(folder_path):
    """Creates a folder if it doesn't exist.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def write_ddl_to_volume(file_name, base_path, ddl_statements, output_format):
    try:
        create_folder_if_not_exists(base_path)
    except Exception as e:
        print(f"Error creating folder: {e}. Check if Volume exists and if your permissions are correct.")
    if output_format in ['sql', 'tsv']:
        full_path = os.path.join(base_path, f"{file_name}.{output_format}")
        with open(full_path, "w") as file:
            for statement in ddl_statements:
                file.write(f"{statement[0]}\n")
    elif output_format == "excel":
        ddl_list = [row.ddl for row in ddl_statements]
        df = pd.DataFrame(ddl_list, columns=['ddl'])
        df_column_to_excel_file(df, file_name, base_path, 'ddl')
    else:
        raise ValueError("Invalid output format. Please choose either 'sql', 'tsv' or 'excel'.")


def create_and_persist_ddl(df: DataFrame,
                           config: MetadataConfig,
                           table_name: str
                           ) -> None:
    """
    Writes the DDL statements from the DataFrame to a volume as SQL files.

    Args:
        df (DataFrame): The DataFrame containing the DDL statements.
        catalog (str): The catalog name.
        dest_schema (str): The destination schema name.
        table_name (str): A list of table names.
        volume_name (str): The volume name.
    """
    print("Running create and persist ddl...")
    current_user = get_current_user()
    current_date = datetime.now().strftime('%Y%m%d')
    if config.volume_name:
        base_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}/{current_user}/{current_date}"
        table_df = df[f'{config.mode}_table_df']
        table_df = populate_log_table(table_df, config, current_user, base_path)
        modified_path = re.sub(r'[^\w\s/]', '_', base_path)
        column_df = df[f'{config.mode}_column_df']
        column_df = populate_log_table(column_df, config, current_user, base_path)
        modified_path = re.sub(r'[^\w\s/]', '_', base_path)
        unioned_df = table_df.union(column_df)
        filter_and_write_ddl(unioned_df, config, modified_path, table_name, current_user, current_date)
    else:
        print("Volume name provided as None in configuration. Not writing DDL to volume...")
        table_df = populate_log_table(df['comment_table_df'], config, current_user, base_path)
        log_metadata_generation(table_df, config)
        column_df = populate_log_table(df['comment_column_df'], config, current_user, base_path)
        log_metadata_generation(column_df, config)

def get_generated_metadata(
    config: MetadataConfig,
    full_table_name: str
    ) -> List[Tuple[PIResponse, CommentResponse]]:
    """
    Generates metadata for a given table. Wraps get_generated_metadata_data_aware() to allow different handling when data is allowed versus disallowed. Currently no difference is implemented between the two routes, but in the future can be if needed.

    Args:
        catalog (str): The catalog name.
        schema (str): The schema name.
        table_name (str): The table name.
        model (str): model name
        prompt_template (str): prompt template

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the generated metadata.
    """
    spark = SparkSession.builder.getOrCreate()

    if config.sample_size == 0:
        responses = get_generated_metadata_data_aware(spark, config, full_table_name)
    elif config.sample_size >= 1:
        responses = get_generated_metadata_data_aware(spark, config, full_table_name)
    return responses


def get_generated_metadata_data_aware(spark: SparkSession, config: MetadataConfig, full_table_name: str):
    df = spark.read.table(full_table_name)
    responses = []
    nrows = df.count()
    chunked_dfs = chunk_df(df, config.columns_per_call)
    for chunk in chunked_dfs:
        sampled_chunk = sample_df(chunk, nrows, config.sample_size)
        prompt = PromptFactory.create_prompt(config, sampled_chunk, full_table_name)
        prompt_messages = prompt.create_prompt_template()
        num_words = check_token_length_against_num_words(prompt_messages, config)
        if config.registered_model_name != "default":
            call_registered_model(config, prompt)
        else:
            chat_response = MetadataGeneratorFactory.create_generator(config)
        response, payload = chat_response.get_responses(config, prompt_messages, prompt.prompt_content)
        responses.append(response)
    return responses


def check_token_length_against_num_words(prompt: str, config: MetadataConfig):
    """
    This function is not intended to catch every instance of overflowing token length, but to avoid significant overflow. Specifically, we compare the number of words in the prompt to the maximum number of tokens allowed in the model. If the number of words exceeds the maximum, an error is raised. This is potentially quite a conservative metric.
    """
    num_words = len(str(prompt).split())
    if num_words > config.max_prompt_length:
        raise ValueError(f"Number of words in prompt exceeds max_tokens. Please reduce the number of columns or increase max_tokens.")
    else:
        return num_words


def call_registered_model(config: MetadataConfig):
    model_name = config.registered_model_name
    model_version = config.registered_model_version
    full_model_name = None
    model = mlflow.pyfunc.load_model(model_name)
    prediction = model.predict()


def choose_registered_model(config, df, full_table_name):
    """Will be implemented."""
    pass


def review_and_generate_metadata(
    config: MetadataConfig,
    full_table_name: str
    ) -> Tuple[DataFrame, DataFrame]:
    """
    Reviews and generates metadata for a list of tables based on the mode.

    Args:
        catalog (str): The catalog name.
        schema (str): The schema name.
        table_names (str): A list of table names.
        model (str): model name
        mode (str): Mode to determine whether to process 'pi' or 'comment'

    Returns:
        Tuple[DataFrame, DataFrame]: DataFrames containing the generated metadata.
    """
    print("Review and generate metadata...")
    table_rows = []
    column_rows = []
    responses = get_generated_metadata(config, full_table_name)
    for response in responses:
        tokenized_full_table_name = replace_catalog_name(config, full_table_name)
        if config.mode == 'comment':
            table_rows = append_table_row(config, table_rows, full_table_name, response, tokenized_full_table_name)
        column_rows = append_column_rows(config, column_rows, full_table_name, response, tokenized_full_table_name)
    return rows_to_df(column_rows, config), rows_to_df(table_rows, config)


def replace_catalog_name(config, full_table_name):
    """
    Replaces __CATALOG_NAME__ in a string with the actual catalog name from a fully scoped table name.

    Args:
        config (MetadataConfig): Configuration object containing setup parameters.
        full_table_name (str): The fully scoped table name.

    Returns:
        str: The string with the catalog name replaced.
    """
    catalog_tokenizable = config.catalog_tokenizable
    parts = full_table_name.split('.')
    if len(parts) != 3:
        raise ValueError("full_table_name must be in the format 'catalog.schema.table'")
    catalog_name, schema_name, table_name = parts
    if config.format_catalog:
        replaced_catalog_name = catalog_tokenizable.replace('__CATALOG_NAME__', catalog_name).format(env=config.env)
    else:
        replaced_catalog_name = catalog_tokenizable.replace('__CATALOG_NAME__', catalog_name)
    logger.debug("Replaced catalog name...", replaced_catalog_name)
    return f"{replaced_catalog_name}.{schema_name}.{table_name}"


def apply_comment_ddl(df: DataFrame, config: MetadataConfig) -> None:
    """
    Applies the comment DDL statements stored in the DataFrame to the table.

    Args:
        df (DataFrame): The DataFrame containing the DDL statements.
    """
    spark = SparkSession.builder.getOrCreate()
    ddl_statements = df.select("ddl").collect()
    for row in ddl_statements:
        ddl_statement = row["ddl"]
        if not config.dry_run:
            spark.sql(ddl_statement)


def process_and_add_ddl(config: MetadataConfig, table_name: str) -> DataFrame:
    """
    Processes the metadata, splits the DataFrame based on 'table' values, applies DDL functions, and returns a unioned DataFrame.

    Args:
        catalog (str): The catalog name data is being read from and written to.
        dest_schema (str): The destination schema name.
        table_names (str): A list of table names.
        model (str): The model name.

    Returns:
        DataFrame: The unioned DataFrame with DDL statements added.
    """
    column_df, table_df = review_and_generate_metadata(config, table_name)
    column_df = split_name_for_df(column_df)
    column_df = hardcode_classification(column_df, config)
    table_df = split_name_for_df(table_df)
    table_df = hardcode_classification(table_df, config)
    if config.allow_manual_override:
        logger.info("Overriding metadata from CSV...")
        column_df = override_metadata_from_csv(column_df, config.override_csv_path, config)
    dfs = add_ddl_to_dfs(config, table_df, column_df, table_name)
    return dfs


def hardcode_classification(df, config):
    df = replace_medical_information_with_phi(df, config)
    df = set_protected_classification(df, config)
    return df

def split_name_for_df(df):
    if df is not None:
        df = split_fully_scoped_table_name(df, 'table')
        logger.info("df columns after generating metadata in process_and_add_ddl", df.columns)
    return df

def add_ddl_to_dfs(config, table_df, column_df, table_name):
    dfs = {}
    if config.mode == "comment":
        summarized_table_df = summarize_table_content(table_df, config, table_name)
        summarized_table_df = split_name_for_df(summarized_table_df)
        dfs['comment_table_df'] = add_ddl_to_table_comment_df(summarized_table_df, "ddl")
        dfs['comment_column_df'] = add_ddl_to_column_comment_df(column_df, "ddl")
        if config.apply_ddl:
            apply_ddl_to_tables(dfs, config)
    elif config.mode == "pi":
        dfs['pi_column_df'] = add_column_ddl_to_pi_df(config, column_df, "ddl")
        table_df = create_pi_table_df(dfs['pi_column_df'], table_name)
        if table_df is not None:
            dfs['pi_table_df']= set_protected_classification(table_df, config)
        if config.apply_ddl:
            apply_ddl_to_tables(dfs, config)
    else:
        raise ValueError("Invalid mode. Use 'pi' or 'comment'.")
    return dfs


def apply_ddl_to_tables(dfs, config):
    apply_comment_ddl(dfs[f'{config.mode}_table_df'], config)
    apply_comment_ddl(dfs[f'{config.mode}_column_df'], config) 


def create_pi_table_df(column_df: DataFrame, table_name: str) -> DataFrame:
    """
    Creates a DataFrame for PI information at the table level. Can be expanded to indicate the type of PI, but for tables it's a little more complicated because they can contain multiple, or even have PI that results from multiple columns, such as PHI that are not present in individual columns.

    Args:
        column_df (DataFrame): The DataFrame containing PI information at the column level.
        table_name (str): The name of the table.

    Returns:
        DataFrame: A DataFrame with PI information at the table level.
    """
    pi_rows = column_df.filter(col("type").isNotNull())
    max_confidence = pi_rows.agg(spark_max("confidence")).collect()[0][0]
    table_classification = determine_table_classification(pi_rows)
    table_name = table_name.split(".")[-1]

    pi_table_row = pi_rows.limit(1).drop('ddl_type') \
                                    .drop('confidence') \
                                    .drop('ddl') \
                                    .withColumn("ddl_type", lit("table")) \
                                    .withColumn("confidence", lit(max_confidence)) \
                                    .withColumn("column_name", lit("None")) \
                                    .withColumn("type", lit(table_classification)) \
                                    .withColumn("classification", lit(table_classification)) \
                                    .withColumn("table_name", lit(table_name))
    pi_table_row = add_table_ddl_to_pi_df(pi_table_row, 'ddl')
    logger.info("PI table rows...", pi_table_row.count())
    return pi_table_row.select(column_df.columns)


def determine_table_classification(pi_rows: DataFrame) -> str:
    """
    Determines the classification based on the values in the 'classification' column of the pi_rows DataFrame.

    Args:
        pi_rows (DataFrame): The DataFrame containing PI information.

    Returns:
        str: The determined classification.
    """
    classification_set = set(pi_rows.select(collect_set("type")).first()[0])

    if classification_set == {} or not classification_set:
        return "None"
    elif classification_set == {"None"}: # No PI information. Only case, done.
        return "None"
    elif (classification_set == {"pii"} or 
          classification_set == {"pii", "None"}
        ):
        return "pii"
    elif "pci" in classification_set and not {"phi", "medical_information"} & classification_set:
        return "pci"
    elif classification_set == {"medical_information"} or classification_set == {"medical_information", "None"}:
        return "medical_information"
    elif (("phi" in classification_set) or 
        ({"pii", "medical_information"}.issubset(classification_set))) and "pci" not in classification_set:
        return "phi"
    elif "pci" in classification_set and ("phi" in classification_set or "medical_information" in classification_set):
        return "all"
    else:
        return "Unknown"


def summarize_table_content(table_df, config, table_name):
    """Create a new completion class for this."""
    if table_df.count() > 1:
        summarizer = TableCommentSummarizer(config, table_df)
        summary = summarizer.summarize_comments(table_name)
        summary_df = table_df.limit(1).withColumn("column_content", lit(summary))
        return summary_df
    elif table_df.count() == 1:
        return table_df
    else:
        raise ValueError("No table rows found during summarization...")


def setup_ddl(config: MetadataConfig) -> None:
    """
    Creates a schema volume if it does not already exist.

    Args:
        setup_params (Dict[str, Any]): A dictionary containing setup parameters including:
            - catalog (str): The catalog name.
            - dest_schema (str): The destination schema name.
            - volume_name (str): The volume name.
    """
    spark = SparkSession.builder.getOrCreate()
    ### Add error handling here
    if config.schema_name:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.catalog_name}.{config.schema_name};")
    print(f"CREATE VOLUME IF NOT EXISTS {config.catalog_name}.{config.schema_name}.{config.volume_name};")
    if config.volume_name:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {config.catalog_name}.{config.schema_name}.{config.volume_name};")
        review_output_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}/{sanitize_email(config.current_user)}/reviewed_outputs/"
        os.makedirs(review_output_path, exist_ok=True)



def create_tables(config: MetadataConfig) -> None:
    """
    Creates a schema volume if it does not already exist.

    Args:
        setup_params (Dict[str, Any]): A dictionary containing setup parameters including:
            - catalog (str): The catalog name.
            - dest_schema (str): The destination schema name.
            - control_table (str): The destination table used for tracking table queue.
    """
    spark = SparkSession.builder.getOrCreate()
    if config.control_table:        
        formatted_control_table = get_control_table(config)
        logger.info("Formatted control table...", formatted_control_table)
        spark.sql(f"""CREATE TABLE IF NOT EXISTS {config.catalog_name}.{config.schema_name}.{formatted_control_table} (table_name STRING, _updated_at TIMESTAMP, _deleted_at TIMESTAMP)""")


def sanitize_email(email: str) -> str:
    """
    Replaces '@' and '.' in an email address with '_'.

    Args:
        email (str): The email address to sanitize.

    Returns:
        str: The sanitized email address.
    """
    return email.replace('@', '_').replace('.', '_')


def instantiate_metadata_objects(env, mode, catalog_name=None, schema_name=None, table_names=None, base_url=None):
    """By default, variables from variables.yml will be used. If widget values are provided, they will override.
    """
    #config = MetadataConfig()
    METADATA_PARAMS = {
        "table_names": table_names
        }
    if catalog_name and catalog_name != "":
        METADATA_PARAMS["catalog_name"] = catalog_name
    if schema_name and schema_name != "":
        METADATA_PARAMS["dest_schema"] = schema_name
    if mode and mode != "":
        METADATA_PARAMS["mode"] = mode
    if mode and mode != "":
        METADATA_PARAMS["env"] = env
    if base_url and base_url != "":
        METADATA_PARAMS["base_url"] = base_url
    return METADATA_PARAMS


def trim_whitespace_from_df(df: DataFrame) -> DataFrame:
    """
    Trims whitespace from all string columns in the DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with trimmed string columns.
    """
    string_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
    for col_name in string_columns:
        df = df.withColumn(col_name, trim(col(col_name)))
    return df

class TableProcessingError(Exception):
    """Custom exception for table processing failures."""

def generate_and_persist_metadata(config: Any) -> None:
    """
    Generates and persists comments for tables based on the provided setup and model parameters.

    Args:
        config: Configuration object containing setup and model parameters.
    """
    spark = SparkSession.builder.getOrCreate()
    logger = logging.getLogger("metadata_processing")
    logger.setLevel(logging.INFO)

    for table in config.table_names:
        log_dict = {}
        try:
            logger.info(f"[generate_and_persist_metadata] Processing table {table}...")

            if not spark.catalog.tableExists(table):
                msg = f"Table {table} does not exist. Deleting from control table and skipping..."
                logger.warning(f"[generate_and_persist_metadata] {msg}")
                mark_as_deleted(table, config)
                log_dict = {
                    "full_table_name": table,
                    "status": "Table does not exist",
                    "user": sanitize_email(config.current_user),
                    "mode": config.mode,
                    "apply_ddl": config.apply_ddl,
                    "_updated_at": str(datetime.now()),
                }
            else:
                df = process_and_add_ddl(config, table)
                logger.info(f"[generate_and_persist_metadata] Generating and persisting ddl for {table}...")
                create_and_persist_ddl(df, config, table)
                log_dict = {
                    "full_table_name": table,
                    "status": "Table processed",
                    "user": sanitize_email(config.current_user),
                    "mode": config.mode,
                    "apply_ddl": config.apply_ddl,
                    "_updated_at": str(datetime.now()),
                }

        except TableProcessingError as tpe:
            logger.error(f"[generate_and_persist_metadata] TableProcessingError for {table}: {tpe}")
            log_dict = {
                "full_table_name": table,
                "status": f"Processing failed: {tpe}",
                "user": sanitize_email(config.current_user),
                "mode": config.mode,
                "apply_ddl": config.apply_ddl,
                "_updated_at": str(datetime.now()),
            }
            raise  # Optionally re-raise if you want to halt further processing

        except Exception as e:
            logger.error(
                f"[generate_and_persist_metadata] Unexpected error for {table}: {e}\n{traceback.format_exc()}"
            )
            log_dict = {
                "full_table_name": table,
                "status": f"Processing failed: {e}",
                "user": sanitize_email(config.current_user),
                "mode": config.mode,
                "apply_ddl": config.apply_ddl,
                "_updated_at": str(datetime.now()),
            }
            raise

        finally:
            try:
                write_to_log_table(log_dict, f"{config.catalog_name}.{config.schema_name}.table_processing_log")
                logger.info(f"[generate_and_persist_metadata] Log written for table {table}.")
            except Exception as log_err:
                logger.error(
                    f"[generate_and_persist_metadata] Failed to write log for {table}: {log_err}\n{traceback.format_exc()}"
                )
            print(f"Finished processing table {table} and writing to log table.")


def setup_queue(config: MetadataConfig) -> List[str]:
    """
    Checks a control table for any records and returns a list of table names.
    If the queue table is empty, reads a CSV with table names based on the flag set in the config file.

    Args:
        config (MetadataConfig): Configuration object containing setup and model parameters.

    Returns:
        List[str]: A list of table names.
    """
    spark = SparkSession.builder.getOrCreate()
    formatted_control_table = get_control_table(config)
    control_table = f"{config.catalog_name}.{config.schema_name}.{formatted_control_table}"
    queued_table_names = set()
    if spark.catalog.tableExists(control_table):
        control_df = spark.sql(f"""SELECT table_name FROM {control_table} WHERE _deleted_at IS NULL""")
        queued_table_names = {row["table_name"] for row in control_df.collect()}
    config_table_string = config.table_names
    config_table_names = [name.strip() for name in config_table_string.split(',') if len(name.strip()) > 0]
    file_table_names = load_table_names_from_csv(config.source_file_path)
    combined_table_names = list(set().union(queued_table_names, config_table_names, file_table_names))
    combined_table_names = ensure_fully_scoped_table_names(combined_table_names, config.catalog_name)
    return combined_table_names


def ensure_fully_scoped_table_names(table_names: List[str], default_catalog: str) -> List[str]:
    """
    Ensures that table names are fully scoped with catalog and schema.

    Args:
        table_names (List[str]): A list of table names.
        default_catalog (str): The default catalog name to use if not specified.

    Returns:
        List[str]: A list of fully scoped table names.
    """
    fully_scoped_table_names = []
    for table_name in table_names:
        parts = table_name.split('.')
        if len(parts) == 2:
            fully_scoped_table_names.append(f"{default_catalog}.{table_name}")
        elif len(parts) == 3:
            fully_scoped_table_names.append(table_name)
        else:
            raise ValueError(f"Invalid table name format: {table_name}")
    return fully_scoped_table_names


def upsert_table_names_to_control_table(table_names: List[str], config: MetadataConfig) -> None:
    """
    Upserts a list of table names into the control table, ensuring no duplicates are created.

    Args:
        table_names (List[str]): A list of table names to upsert.
        config (MetadataConfig): Configuration object containing setup and model parameters.
    """
    print(f"Upserting table names to control table {table_names}...")
    spark = SparkSession.builder.getOrCreate()
    formatted_control_table = get_control_table(config)
    control_table = f"{config.catalog_name}.{config.schema_name}.{formatted_control_table}"
    table_names = ensure_fully_scoped_table_names(table_names, config.catalog_name)
    table_names_df = spark.createDataFrame([(name,) for name in table_names], ["table_name"])
    table_names_df = trim_whitespace_from_df(table_names_df)
    existing_df = spark.read.table(control_table)
    new_table_names_df = table_names_df.join(existing_df, on="table_name", how="left_anti") \
                                    .withColumn("_updated_at", current_timestamp()) \
                                    .withColumn("_deleted_at", lit(None).cast(TimestampType()))
    if new_table_names_df.count() > 0:
        new_table_names_df.write.format("delta").mode("append").saveAsTable(control_table)
        print(f"Inserted {new_table_names_df.count()} new table names into the control table {control_table}...")
    else:
        print("No new table names to upsert.")


def load_table_names_from_csv(csv_file_path):
    spark = SparkSession.builder.getOrCreate()
    df_tables = spark.read.csv(f"file://{os.path.join(os.getcwd(), csv_file_path)}", header=True)
    table_names = [row["table_name"] for row in df_tables.select("table_name").collect()]
    return sanitize_string_list(table_names)

def sanitize_string_list(string_list: List[str]):
    sanitized_list = []
    for s in string_list:
        s = str(s)
        s = s.strip()
        s = ' '.join(s.split())
        s = s.lower()
        s = ''.join(c for c in s if c.isalnum() or c.isspace() or c == '.' or c == "_")
        sanitized_list.append(s)
    return sanitized_list

def split_fully_scoped_table_name(df: DataFrame, full_table_name_col: str) -> DataFrame:
    """
    Splits a fully scoped table name column into catalog, schema, and table columns.

    Args:
        df (DataFrame): The input DataFrame.
        full_table_name_col (str): The name of the column containing the fully scoped table name.

    Returns:
        DataFrame: The updated DataFrame with catalog, schema, and table columns added.
    """
    split_col = split(col(full_table_name_col), r'\.')
    df = df.withColumn('catalog', split_col.getItem(0)) \
           .withColumn('schema', split_col.getItem(1)) \
           .withColumn('table_name', split_col.getItem(2))
    return df


def split_table_names(table_names: str) -> List[str]:
    if not table_names:
        return []
    return table_names.split(',')

def replace_fully_scoped_table_column(df):
    return df.withColumn('table', split_part(col('table'), '.', -1))

@udf
def generate_table_comment_ddl(full_table_name: str, comment: str) -> str:
    """
    Generates a DDL statement for creating a table with the given schema.

    Args:
        table_name (str): The name of the table to create.
        schema (StructType): The schema of the table.

    Returns:
        str: The DDL statement for adding the
    """
    ddl_statement = f"""COMMENT ON TABLE {full_table_name} IS "{cleanse_sql_comment(comment)}";"""
    return ddl_statement


@udf
def generate_column_comment_ddl(full_table_name: str, column_name: str, comment: str) -> str:
    """
    Generates a DDL statement for creating a table with the given schema.

    Args:
        table_name (str): The name of the table to create.
        schema (StructType): The schema of the table.

    Returns:
        str: The DDL statement for adding the column comment to the table.
    """
    dbr_number = os.environ.get('DATABRICKS_RUNTIME_VERSION')
    if float(dbr_number) >= 16:
        ddl_statement = f"""COMMENT ON COLUMN {full_table_name}.`{column_name}` IS "{cleanse_sql_comment(comment)}";"""
    elif float(dbr_number) >=14 and float(dbr_number) < 16:
        ddl_statement = f"""ALTER TABLE {full_table_name} ALTER COLUMN `{column_name}` COMMENT "{cleanse_sql_comment(comment)}";"""
    else: 
        raise ValueError(f"Unsupported Databricks runtime version: {dbr_number}")
    return ddl_statement


@udf
def generate_table_pi_information_ddl(table_name: str, classification: str, pi_type: str) -> str:
    """
    Generates a DDL statement for ALTER TABLE that will tag a column with information about pi.

    Args:
        table_name (str): The name of the table to create.
        column_name (str): The schema of the table.
        pi_tag (str): The schema of the table.

    Returns:
        str: The DDL statement for adding the pi tag to the table.
    """
    ddl_statement = f"ALTER TABLE {table_name} SET TAGS ('data_classification' = '{classification}', 'data_subclassification' = '{pi_type}');"
    return ddl_statement


@udf
def generate_pi_information_ddl(table_name: str, column_name: str, classification: str, pi_type: str) -> str:
    """
    Generates a DDL statement for ALTER TABLE that will tag a column with information about pi.

    Args:
        table_name (str): The name of the table to create.
        column_name (str): The schema of the table.
        pi_tag (str): The schema of the table.

    Returns:
        str: The DDL statement for adding the pi tag to the table.
    """
    ddl_statement = f"ALTER TABLE {table_name} ALTER COLUMN `{column_name}` SET TAGS ('data_classification' = '{classification}', 'data_subclassification' = '{pi_type}');"
    return ddl_statement
