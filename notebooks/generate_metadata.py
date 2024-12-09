# Databricks notebook source
# MAGIC %md
# MAGIC # Generate comments and create DDL

# COMMAND ----------

# TODO: register a pyfunc so that we can iterate on prompts
# TODO: Flag for detailed column metrics - null count, min, max, number of values and examples of values.
# TODO: Separate out table comments into its own Response
# TODO: Summarizer step for table comments
# TODO: Improve outputs to improve prompting for Genie
# TODO: Add async
# TODO: any utility of agent framework?
# TODO: Fix input data classes
# TODO: Move modules out of notebook
# TODO: whl build

# COMMAND ----------

# MAGIC %pip install pydantic==2.9.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Can change these

# COMMAND ----------

METADATA_PARAMS = {
    "table_names": [],
    "dest_schema": "test",
    "mode": "comment"
    }
acro_content = "{'DBX': 'Databricks'}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Don't change these unless you know what you are doing and do it intentionally

# COMMAND ----------

api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

from pydantic import BaseModel, Field, Extra, ValidationError, ConfigDict
from openai import OpenAI
import os
from pydantic.dataclasses import dataclass
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from mlflow.types.llm import TokenUsageStats, ChatResponse
import json
import re
from typing import List, Dict, Any, Literal, Tuple
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, Row
from pyspark.sql.functions import col, struct, to_timestamp, current_timestamp
from typing import List, Dict, Any
from pyspark.sql import DataFrame
import nest_asyncio
from pyspark.sql import Row
from datetime import datetime
import mlflow
from pyspark.sql.functions import lit
import time
import random
from abc import ABC

from src.prompts import create_prompt_template
from src.config import MetadataConfig

# COMMAND ----------

os.environ["DATABRICKS_TOKEN"]=api_key
os.environ["DATABRICKS_HOST"]=MetadataConfig.SETUP_PARAMS['base_url']

# COMMAND ----------

class DDLGenerator(ABC):
    def __init__(self):
        pass

class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str

    @classmethod
    def from_df(cls, df: DataFrame) -> Dict[str, Any]:        
        return {
            "table_name": "{catalog_name}.{schema_name}.{table_name}",
            "column_contents": cls.df.toPandas().to_dict(orient='list')
        }

class PInput(Input):
    
    column_contents: List[Dict[str, Any]]


class CommentInput(Input):
    
    column_contents: List[List[Any]]


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    column_names: List[str]

    
class PIResponse(Response):
    model_config = ConfigDict(extra="forbid")

    column_contents: list[dict[str, Any]]


class CommentResponse(Response):
    model_config = ConfigDict(extra="forbid")

    column_contents: list[str]


class PIChatCompletionResponse(ABC):

    def get_pi_response(self, content: str, prompt_content: str, model: str):
        chat_completion = get_response(content, prompt_content, model)
        response_payload = chat_completion.choices[0].message
        response = response_payload.content
        retries = 0
        try:            
            response_dict = json.loads(response)
            if not isinstance(response_dict, dict):
                raise ValueError("columns field is not a valid dict")
            chat_response = PIResponse(**response_dict)
            return chat_response, response_payload
        except (ValidationError, json.JSONDecodeError, AttributeError) as e:
            retries+=1
            raise ValueError(f"Validation error: {e} {response}")
            return None

# TODO: change ChatCompletionResponse to a factory 
class CommentChatCompletionResponse(ABC):
    def __init__(self, config, df, full_table_name):
        self.config = config
        self.df = df
        self.full_table_name = full_table_name
        self.comment_input_data = self.convert_to_comment_input()
        if self.config.SETUP_PARAMS.get('add_metadata'):
            self.add_metadata_to_comment_input()
        print("Instantiating chat completion response...")

    def convert_to_comment_input(self) -> Dict[str, Any]:
        return {
            "table_name": self.full_table_name,
            "column_contents": self.df.toPandas().to_dict(orient='split'),
        }

    def add_metadata_to_comment_input(self) -> None:
        print("Adding metadata to comment input...")
        config = self.config
        column_metadata_dict = {}
        for column_name in self.comment_input_data['column_contents']['columns']:
            extended_metadata_df = spark.sql(
                f"DESCRIBE EXTENDED {self.full_table_name} {column_name}"
            )            
            filtered_metadata_df = extended_metadata_df.filter(extended_metadata_df["info_value"] != "NULL")            
            column_metadata = filtered_metadata_df.toPandas().to_dict(orient='list')
            combined_metadata = dict(zip(column_metadata['info_name'], column_metadata['info_value']))
            column_metadata_dict[column_name] = combined_metadata
            
        self.comment_input_data['column_contents']['column_metadata'] = column_metadata_dict

    def get_comment_response(self, 
                             config: MetadataConfig,
                             content: str, 
                             prompt_content: str, 
                             model: str, 
                             max_tokens: int, 
                             temperature: float,
                             retries: int = 0, 
                             max_retries: int = 5) -> Tuple[CommentResponse, Dict[str, Any]]:
        try:
            chat_completion = self._get_chat_completion(config, prompt_content, model, max_tokens, temperature)
            response_payload = chat_completion.choices[0].message
            response_dict = self._parse_response(response_payload.content)
            self._validate_response(content, response_dict)
            chat_response = CommentResponse(**response_dict)
            return chat_response, response_payload
        except (ValidationError, json.JSONDecodeError, AttributeError, ValueError) as e:
            if retries < max_retries:
                print(f"Attempt {retries + 1} failed for {response_payload.content}, retrying due to {e}...")
                return self.get_comment_response(config, content, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print("Validation error - response")
                raise ValueError(f"Validation error after {max_retries} attempts: {e}")

    def _get_chat_completion(self, config: MetadataConfig, prompt_content: str, model: str, max_tokens: int, temperature: float, retries: int = 0, max_retries: int = 3) -> ChatCompletion:
        client = OpenAI(api_key=os.environ['DATABRICKS_TOKEN'], base_url=config.SETUP_PARAMS['base_url'])
        try:
            return client.chat.completions.create(
                messages=prompt_content,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            if retries < max_retries:
                print(f"Error: {e}. Retrying in {2 ** retries} seconds...")
                exponential_backoff(retries)
                return self._get_chat_completion(config, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print(f"Failed after {max_retries} retries.")
                raise e

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response_dict = json.loads(response)
            if not isinstance(response_dict, dict):
                raise ValueError("Response is not a valid dict")
            return response_dict
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}")

    def _validate_response(self, content: str, response_dict: Dict[str, Any]) -> None:
        if not self._check_list_and_dict_keys_match(content['column_contents']['columns'], response_dict['column_names']):
            raise ValueError("Column names do not match column contents")
    
    @staticmethod
    def _check_list_and_dict_keys_match(dict_list, string_list):
        if isinstance(dict_list, list):
            dict_keys = dict_list
        else:
            try:
                dict_keys = dict_list.keys()
            except: 
                raise TypeError("dict_list is not a list or a dictionary")
        list_matches_keys = all(item in dict_keys for item in string_list)
        keys_match_list = all(key in string_list for key in dict_keys)
        if not (list_matches_keys and keys_match_list):
            return False
        return True



class MetadataGenerator(mlflow.pyfunc.ChatModel, CommentChatCompletionResponse, ABC):
    def __init__(self):
        pass

    @classmethod
    def from_context(cls, context: mlflow.pyfunc.PythonModelContext):
        chat = cls.__new__(cls)
        return chat
    
    @staticmethod
    def _format_prompt(prompt, content, acro_content):
        return create_prompt_template(prompt, content, acro_content)

    def get_openai_client(self):
       return OpenAI(
           api_key=os.environ["DATABRICKS_TOKEN"],
           base_url=os.environ["DATABRICKS_HOST"] + "serving-endpoints")

    def load_context(self, context):
        """Instantiated OpenAI client cannot be added to load_context.
        """
        self.api_key = context.artifacts["api_key"]
        self.base_url = context.artifacts["base_url"]

    def predict(self, context, messages):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        prompt = self.create_prompt(messages)
        response = self.client.chat.completions.create(
            messages=prompt,
            model=context.get("model", "default-model"),
            max_tokens=context.get("max_tokens", 3000),
            temperature=context.get("temperature", 0.1)
        )
        text = response.choices[0].message["content"]

        prompt_tokens = len(self.client.tokenizer.encode(prompt))
        completion_tokens = len(self.client.tokenizer.encode(text))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        
        # Construct the response
        chat_response = {
            "id": f"response_{random.randint(0, 100)}",
            "model": "MyChatModel",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        
        return ChatResponse(**chat_response)

    def create_prompt(self, messages):
        # Convert the list of messages to the format expected by the OpenAI API
        return [{"role": message["role"], "content": message["content"]} for message in messages]


class CommentGenerator(MetadataGenerator, CommentChatCompletionResponse):
    def __init__(self):
        pass


class PIIdentifier(MetadataGenerator, PIChatCompletionResponse):
    def __init__(self):
        pass

# COMMAND ----------

def get_response(config, content: str, prompt_content: str, model: str, max_tokens: str, temperature: str):
    client = OpenAI(
        api_key=os.environ['DATABRICKS_TOKEN'],
        base_url=config.SETUP_PARAMS['base_url']
    )
    chat_completion = client.chat.completions.create(
        messages=prompt_content,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return chat_completion


def exponential_backoff(retries, base_delay=1, max_delay=120, jitter=True):
    """
    Exponential backoff with optional jitter.
    
    :param retries: Number of retries attempted.
    :param base_delay: Initial delay in seconds.
    :param max_delay: Maximum delay in seconds.
    :param jitter: Whether to add random jitter to the delay.
    :return: None
    """
    delay = min(base_delay * (2 ** retries), max_delay)
    if jitter:
        delay = delay / 2 + random.uniform(0, delay / 2)
    time.sleep(delay)


def get_responses(chat_response,
                  config: MetadataConfig) -> Tuple[PIResponse, CommentResponse]:
    print("Getting responses...")
    pi_input_list = None # Hard coding this until full factory is setup.
    responses = {'pi': None, 'comment': None}
    if config.mode == "pi":
        prompt_template = create_prompt_template(pi_input_list, config.SETUP_PARAMS['acro_content'])
        if len(prompt_template) > config.SETUP_PARAMS['max_prompt_length']:
            raise ValueError("The prompt template is too long. Please reduce the number of columns or increase the max_prompt_length.")
        pi_response, message_payload = chat_response.get_pi_response(content=pi_input_list, 
                                                                                  prompt_content=prompt_template['pi'],
                                                                                  model=config.SETUP_PARAMS['model'], 
                                                                                  max_tokens=config.SETUP_PARAMS['max_tokens'], 
                                                                                  temperature=config.SETUP_PARAMS['temperature'])
        responses.append(pi_response)
        responses['pi'] = pi_response
    elif config.mode == "comment":
        prompt_template = create_prompt_template(chat_response.comment_input_data, config.SETUP_PARAMS['acro_content'])
        if len(prompt_template) > config.SETUP_PARAMS['max_prompt_length']:
            raise ValueError("The prompt template is too long. Please reduce the number of columns or increase the max_prompt_length.")
        comment_response, message_payload = chat_response.get_comment_response(config, content=chat_response.comment_input_data, 
                                                                                                 prompt_content=prompt_template['comment'], 
                                                                                                 model=config.SETUP_PARAMS['model'], 
                                                                                                 max_tokens=config.SETUP_PARAMS['max_tokens'], 
                                                                                                 temperature=config.SETUP_PARAMS['temperature'])
        responses['comment'] = comment_response
    else: 
        if config.mode not in ["pi", "comment"]:
                raise ValueError("To be valid, mode must be 'pi' or 'comment'.")
    return responses


def tag_table(table_name: str, tags: Dict[str, str]) -> None:
    """
    Tags a table with the provided tags.

    Args:
        table_name (str): The name of the table to tag.
        tags (Dict[str, str]): A dictionary of tags to apply to the table.
    """
    for key, value in tags.items():
        spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES ('{key}' = '{value}');")


def write_to_log_table(log_data: Dict[str, Any], log_table_name: str) -> None:
    """
    Writes log data to a specified log table.

    Args:
        log_data (Dict[str, Any]): The log data to write.
        log_table_name (str): The name of the log table.
    """
    log_df = spark.createDataFrame([log_data])
    log_df.write.format("delta").mode("append").saveAsTable(log_table_name)


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
    ddl_statement = f"""COMMENT ON TABLE {full_table_name} IS "{comment}";"""
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
    ddl_statement = f"""ALTER TABLE {full_table_name} ALTER COLUMN {column_name} COMMENT "{comment}";"""
    return ddl_statement


@udf
def generate_pi_information_ddl(table_name: str, column_name: str, pi_tag: str, pi_type: str) -> str:
    """
    Generates a DDL statement for ALTER TABLE that will tag a column with information about pi.

    Args:
        table_name (str): The name of the table to create.
        column_name (str): The schema of the table.
        pi_tag (str): The schema of the table.

    Returns:
        str: The DDL statement for adding the pi tag to the table.
    """
    ddl_statement = f"ALTER TABLE {table_name} SET TAGS ('has_pi' = '{pi_type}');"
    return ddl_statement


@udf
def generate_pi_information_ddl(table_name: str, column_name: str, pi_tag: str, pi_type: str) -> str:
    """
    Generates a DDL statement for ALTER TABLE that will tag a column with information about pi.

    Args:
        table_name (str): The name of the table to create.
        column_name (str): The schema of the table.
        pi_tag (str): The schema of the table.

    Returns:
        str: The DDL statement for adding the pi tag to the table.
    """
    ddl_statement = f"ALTER TABLE {table_name} ALTER COLUMN {column_name} SET TAGS ('has_pi' = '{pi_type}');"
    return ddl_statement


def count_df_columns(df):
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
    query = f"""DESCRIBE EXTENDED {config.SETUP_PARAMS['catalog']}.{config.dest_schema}.{table_name} {column_name};"""
    return spark.sql(query)
    

def determine_sampling_ratio(nrows: int, sample_size: int) -> float:
    """
    Takes a number of rows and a ratio, and returns the number of rows to sample.

    Args:
        nrows (int): The number of rows in the DataFrame.
        ratio (int): The ratio to use for sampling to avoid too many rows.

    Returns:
        ratio (float): The number of rows to sample.
    """
    if sample_size < nrows:
        ratio = sample_size / nrows
    else:
        ratio = 1.0        
    print("Sampling ratio:", ratio)
    return ratio


def sample_df(df: DataFrame, nrows: int, sample_size: int = 5) -> DataFrame:
    """
    Sample dataframe to a given size and filter out rows with lots of nulls.

    Args:
        df (DataFrame): The DataFrame to be analyzed.
        nrows (int): number of rows in dataframe
        sample_size (int): The number of rows to sample.
    
    Returns:
        DataFrame: A DataFrame with columns indicating potential PII.
    """
    if nrows < sample_size:
        print(f"Not enough rows for a proper sample. Continuing with inference with {nrows} rows...")
        return df.limit(sample_size)
    larger_sample = sample_size * 100
    sampling_ratio = determine_sampling_ratio(nrows, larger_sample)
    sampled_df = df.sample(withReplacement=False, fraction=sampling_ratio)
    print("nrows of sampled df", sampled_df.count())
    threshold = int(sampled_df.columns.__len__() * 0.5)
    print("threshold", threshold)
    # TODO: change len to function not dunder method
    filtered_df = sampled_df.filter(sampled_df.columns.__len__() - sum(sampled_df[col].isNull().cast("int")
                                                                       for col in sampled_df.columns) >= threshold
                                    )
    result_rows = filtered_df.count()
    print("result rows", result_rows)

    if result_rows < sample_size:
        print("Not enough non-NULL rows:", result_rows, "vs", sample_size)
        print("Returning available rows, despite large proportion of NULLs")
        return df.limit(sample_size)
    
    print(f"Filtering {result_rows} rows down to {sample_size} rows...")
    return filtered_df.limit(sample_size)


def append_table_row(rows: List[Row], full_table_name: str, response: Dict[str, Any], tokenized_full_table_name: str) -> List[Row]:
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
        comment_type='table',
        column_name='None',
        column_content=response.table,
        _created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),     
    )
    rows.append(row)
    return rows


def append_column_rows(rows: List[Row], full_table_name: str, response: Dict[str, Any], tokenized_full_table_name: str) -> List[Row]:
    """
    Appends column rows to the list of rows.

    Args:
        rows (List[Row]): The list of rows to append to.
        full_table_name (str): The full name of the table.
        response (Dict[str, Any]): The response dictionary containing column information.

    Returns:
        List[Row]: The updated list of rows with the new column rows appended.
    """
    for column_name, column_content in zip(response.column_names, response.column_contents):
        if isinstance(column_content, dict):   
            row = Row(
                table=full_table_name,
                tokenized_table=tokenized_full_table_name,        
                comment_type='column',
                column_name=column_name,
                **column_content,
                _created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        elif isinstance(column_content, str):
            row = Row(
                table=full_table_name,
                tokenized_table=tokenized_full_table_name,        
                comment_type='column',
                column_name=column_name,
                column_content=column_content,
                _created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        else:
            raise ValueError("Invalid column contents type, should be dict or string.")
        rows.append(row)
    return rows


def rows_to_df(rows: List[Row]) -> DataFrame:
    """
    Converts a list of rows to a Spark DataFrame.

    Args:
        rows (List[Row]): The list of rows to convert.

    Returns:
        DataFrame: The Spark DataFrame created from the list of rows.
    """
    return spark.createDataFrame(rows)


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


def add_column_ddl_to_pi_df(df: DataFrame, ddl_column: str) -> DataFrame:
    """
    Adds a DDL statement to a DataFrame for PI information.

    Args:
        df (DataFrame): The DataFrame to add the DDL statement to.
        ddl_column (str): The name of the DDL column.

    Returns:
        DataFrame: The updated DataFrame with the DDL statement added.
    """
    return df.withColumn("ddl", generate_pi_information_ddl('tokenized_table', 'column_name', 'column_content'))


def get_current_user() -> str:
    """
    Retrieves the current user.

    Returns:
        str: The current user.
    """
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
    print("df to sql file")
    selected_column_df = df.select(sql_column)
    column_list = [row[sql_column] for row in selected_column_df.collect()]
    uc_volume_path = f"/Volumes/{catalog_name}/{dest_schema_name}/{filename}.sql"
    with open(uc_volume_path, 'w') as file:
        for item in column_list:
            file.write(f"{item}\n")
    return uc_volume_path


def populate_log_table(df, config, current_user, base_path):
        return (df.withColumn("current_user", lit(current_user))
          .withColumn("model", lit(config.SETUP_PARAMS['model']))
          .withColumn("sample_size", lit(config.SETUP_PARAMS['sample_size']))
          .withColumn("max_tokens", lit(config.SETUP_PARAMS['max_tokens']))
          .withColumn("temperature", lit(config.SETUP_PARAMS['temperature']))
          .withColumn("columns_per_call", lit(config.SETUP_PARAMS['columns_per_call']))
          .withColumn("status", lit("No Volume specified..."))
        )


def mark_as_deleted(table_name: str, config: MetadataConfig) -> None:
    """
    Updates the _deleted_at and _updated_at columns to the current timestamp for the specified table.

    Args:
        table_name (str): The name of the table to update.
        config (MetadataConfig): Configuration object containing setup and model parameters.
    """
    print(table_name)
    control_table = f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.{config.SETUP_PARAMS['control_table']}"
    print(control_table)
    update_query = f"""
    UPDATE {control_table}
    SET _deleted_at = current_timestamp(), 
        _updated_at = current_timestamp()
    WHERE table_name = '{table_name}'
    """
    print(update_query)
    spark.sql(update_query)
    print(f"Marked {table_name} as deleted in the control table...")


def log_metadata_generation(df: DataFrame, config: MetadataConfig, table_name: str, volume_name: str) -> None:   
    df.write.mode('append').saveAsTable(f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.metadata_generation_log")
    mark_as_deleted(table_name, config)


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
        catalog (str): The catalog name.
        dest_schema (str): The destination schema name.
        table_names (List[str]): A list of table names.
        base_path: str
        current_user: str
        current_date: str
    """
    df = df.filter(df['table'] == full_table_name)
    print("df filtering...")
    ddl_statements = df.select("ddl").collect()
    table_name = re.sub(r'[^\w\s/]', '_', full_table_name)
    file_path = os.path.join(base_path, f"{table_name}.sql")
    try:
        write_ddl_to_volume(file_path, base_path, ddl_statements)
        df = df.withColumn("status", lit("Success"))
        log_metadata_generation(df, config, full_table_name, base_path)
    except Exception as e:
        print(f"Error writing DDL to volume: {e}. Check if Volume exists and if your permissions are correct.")        
        df = df.withColumn("status", lit("Failed writing to volume..."))
        log_metadata_generation(df, config, full_table_name, base_path)
    

def create_folder_if_not_exists(folder_path):
    """Creates a folder if it doesn't exist.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def write_ddl_to_volume(file_path: str, base_path: str, ddl_statements: List[str]) -> None:
    """
    Writes the DDL statements to a specified file within a given base path.

    Args:
        file_path (str): The path to the file where DDL statements will be written.
        base_path (str): The base path where the file will be created.
        ddl_statements (List[str]): A list of DDL statements to write to the file.
    """
    print("write ddl to volume")
    try:
        create_folder_if_not_exists(base_path)
    except Exception as e:
        print(f"Error creating folder: {e}. Check if Volume exists and if your permissions are correct.")
    with open(file_path, 'w') as file:
        for row in ddl_statements:
            print(f"Writing DDL: {row['ddl']}")
            file.write(f"{row['ddl']}\n")


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
    if config.SETUP_PARAMS['volume_name']:
        base_path = f"/Volumes/{config.SETUP_PARAMS['catalog']}/{config.dest_schema}/{config.SETUP_PARAMS['volume_name']}/{current_user}/{current_date}"
        print(f"Writing DDL for {table_name}...")
        table_df = df[f'{config.mode}_table_df']
        table_df = populate_log_table(table_df, config, current_user, base_path)
        modified_path = re.sub(r'[^\w\s/]', '_', base_path)
        filter_and_write_ddl(table_df, config, modified_path, table_name, current_user, current_date)
        print(f"Writing DDL for {table_name} for columns...")
        column_df = df[f'{config.mode}_column_df']
        "Column df..."
        column_df = populate_log_table(column_df, config, current_user, base_path)
        modified_path = re.sub(r'[^\w\s/]', '_', base_path)
        filter_and_write_ddl(column_df, config, modified_path, table_name, current_user, current_date)
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
    Generates metadata for a given table.

    Args:
        catalog (str): The catalog name.
        schema (str): The schema name.
        table_name (str): The table name.
        model (str): model name
        prompt_template (str): prompt template

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the generated metadata.
    """
    df = spark.read.table(full_table_name)
    nrows = df.count()
    chunked_dfs = chunk_df(df, config.SETUP_PARAMS['columns_per_call'])
    responses = []
    for chunk in chunked_dfs:
        sampled_chunk = sample_df(chunk, nrows, config.SETUP_PARAMS['sample_size'])
        chat_response = CommentChatCompletionResponse(config, sampled_chunk, full_table_name)
        response = get_responses(chat_response, config)
        responses.append(response)
    return responses


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
        if config.mode == "pi":
            response_dict = response['pi']
        elif config.mode == "comment":
            response_dict = response['comment']
        else:
            raise ValueError("Invalid mode. Use 'pi' or 'comment'.")
        table_rows = append_table_row(table_rows, full_table_name, response_dict, tokenized_full_table_name)
        column_rows = append_column_rows(column_rows, full_table_name, response_dict, tokenized_full_table_name)
    ### TODO: add all the table comments to an additional summarizer call rather than taking just the first one.         
    return rows_to_df(column_rows), rows_to_df(table_rows)


def replace_catalog_name(config, full_table_name):
    """
    Replaces __CATALOG_NAME__ in a string with the actual catalog name from a fully scoped table name.

    Args:
        config (MetadataConfig): Configuration object containing setup parameters.
        full_table_name (str): The fully scoped table name.

    Returns:
        str: The string with the catalog name replaced.
    """
    catalog_tokenizable = config.SETUP_PARAMS['catalog_tokenizable']
    catalog_name = full_table_name.split('.')[0]
    
    return catalog_tokenizable.replace('__CATALOG_NAME__', catalog_name)


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
    print("Process and add ddl...")
    column_df, table_df = review_and_generate_metadata(config, table_name)
    dfs = {}
    if config.mode == "comment":
        # TODO: turn this into a class
        table_df = add_ddl_to_table_comment_df(table_df, "ddl")
        column_df = add_ddl_to_column_comment_df(column_df, "ddl")
        # TODO: instead of taking the first table description, fetch all and send to a summarizer.
        dfs['comment_table_df'] = table_df.limit(1)
        dfs['comment_column_df'] = column_df
    elif config.mode == "pi":
        table_df = add_column_ddl_to_pi_df(table_df, "ddl")
        dfs['pi_table_df'] = table_df.limit(1)
        dfs['pi_column_df'] = column_df   
    print(dfs) 
    return dfs


def create_volume(config: MetadataConfig) -> None:
    """
    Creates a schema volume if it does not already exist.

    Args:
        setup_params (Dict[str, Any]): A dictionary containing setup parameters including:
            - catalog (str): The catalog name.
            - dest_schema (str): The destination schema name.
            - volume_name (str): The volume name.
    """
    if config.SETUP_PARAMS["volume_name"]:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {config.SETUP_PARAMS['catalog']}.{config.dest_schema}.{config.SETUP_PARAMS['volume_name']}")


def create_tables(config: MetadataConfig) -> None:
    """
    Creates a schema volume if it does not already exist.

    Args:
        setup_params (Dict[str, Any]): A dictionary containing setup parameters including:
            - catalog (str): The catalog name.
            - dest_schema (str): The destination schema name.
            - control_table (str): The volume name.
    """
    if config.SETUP_PARAMS["control_table"]:
        spark.sql(f"""CREATE TABLE IF NOT EXISTS {config.SETUP_PARAMS['catalog']}.{config.dest_schema}.{config.SETUP_PARAMS['control_table']} (table_name STRING, _updated_at TIMESTAMP, _deleted_at TIMESTAMP)""")


def generate_and_persist_comments(config) -> None:
    """
    Generates and persists comments for tables based on the provided setup and model parameters.

    Args:
        config (MetadataConfig): Configuration object containing setup and model parameters.
    """
    for table in config.table_names:
        print(f"Processing table {table}...")        
        df = process_and_add_ddl(config, table)        
        print(f"Generating and persisting ddl for {table}...")
        create_and_persist_ddl(df, config, table)


def setup_queue(config: MetadataConfig) -> List[str]:
    """
    Checks a control table for any records and returns a list of table names.
    If the queue table is empty, reads a CSV with table names based on the flag set in the config file.

    Args:
        config (MetadataConfig): Configuration object containing setup and model parameters.

    Returns:
        List[str]: A list of table names.
    """
    control_table = f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.{config.SETUP_PARAMS['control_table']}"
    queued_table_names = set()
    if spark.catalog.tableExists(control_table):
        control_df = spark.sql(f"""SELECT table_name FROM {control_table} WHERE _deleted_at IS NULL""")
        queued_table_names = {row["table_name"] for row in control_df.collect()}
    config_table_names = config.table_names
    file_table_names = load_table_names_from_csv(config.SETUP_PARAMS['source_file_path'])
    combined_table_names = list(set().union(queued_table_names, config_table_names, file_table_names))
    combined_table_names = ensure_fully_scoped_table_names(combined_table_names, config.SETUP_PARAMS['catalog'])
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
    print("Upserting table names to control table {table_names}...")
    control_table = f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.{config.SETUP_PARAMS['control_table']}"
    table_names = ensure_fully_scoped_table_names(table_names, config.SETUP_PARAMS['catalog'])
    table_names_df = spark.createDataFrame([(name,) for name in table_names], ["table_name"])
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
    print(csv_file_path)
    df_tables = spark.read.csv(f"file://{os.path.join(os.getcwd(), csv_file_path)}", header=True)    
    table_names = [row["table_name"] for row in df_tables.select("table_name").collect()]
    return table_names


def main(metadata_params):
    config = MetadataConfig(**metadata_params)
    print(config.SETUP_PARAMS['columns_per_call'])
    create_volume(config)
    create_tables(config)
    queue = setup_queue(config)
    if config.SETUP_PARAMS['control_table']:
        upsert_table_names_to_control_table(queue, config)
    config.table_names.extend(queue)
    generate_and_persist_comments(config)

# COMMAND ----------

# MAGIC %md
# MAGIC # Now run the code

# COMMAND ----------

main(METADATA_PARAMS)

# COMMAND ----------

config = MetadataConfig(**METADATA_PARAMS)
df = spark.read.table(f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.metadata_generation_log").orderBy(col('_created_at').desc())
display(df)

# COMMAND ----------

config = MetadataConfig(**METADATA_PARAMS)
df = spark.read.table(f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.metadata_control")
display(df)

# COMMAND ----------

#spark.sql(f"DROP TABLE {config.SETUP_PARAMS['catalog']}.{config.dest_schema}.metadata_control")
