# Databricks notebook source
# MAGIC %md
# MAGIC # Generate comments and create DDL

# COMMAND ----------

# MAGIC %pip install pydantic==2.9.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Change these variables

# COMMAND ----------

base_url = "https://adb-830292400663869.9.azuredatabricks.net/serving-endpoints" # change this to the workspace url you are in
catalog = "eswanson_genai" # should source and dest catalog be the same? Currently yes, could be changed
catalog_tokenizable = "eswanson_genai_{{env}}" ### Change this to match the tokenization approach you need for artifacts in a build pipeline. This will be written into DDL files, so if you want this to match the catalog exactly, or handle tokenization with a different tool.
table_names = ["default.simple_test", "default.simple_test2"]
dest_schema = "test" # dest schema will generally differ from the source schema, allowing different source and dest schemas
#test_schema = "test" # 
volume_name = "generated_metadata" # Either need to be able to create a volume in a schema, or have the volume pre-created.

# COMMAND ----------

# MAGIC %md
# MAGIC # Maybe change these variables

# COMMAND ----------

acro_content = "{'DBX': 'Databricks'}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Be careful changing these - could reduce quality or increase cost

# COMMAND ----------

### A much simpler, bare bones result would probably be attainable from decreasing the sample size, decreasing the max tokens, decreasing the temperature, increasing the columns per call, and potentially changing the model to "databricks-dbrx-instruct".
sample_size=10 # up to a point, higher is more info/better, but it costs more and more is not always better
max_tokens=3000 # mostly a cost vs. quality tradeoff
max_prompt_length=3000 # caps the prompt length. If a prompt is longer than this, the run will fail.
temperature = 0.1 # the 'creativity' of the model, higher is more creative but more hallucinations
columns_per_call=10 # More columns per call may be cheaper and faster overall, but could reduce complexity and quality of responses for columns and lead to unexpected behavior.
model = "databricks-meta-llama-3-1-70b-instruct" # other options that could be explored for example would be "databricks-meta-llama-3-1-405b-instruct", "databricks-dbrx-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC # Don't change these unless you know what you are doing and do it intentionally

# COMMAND ----------

source_file = None # intended as a placeholder, use this once the read table names from csv option is implemented
test = False
generation_mode = "comment" # can also be pi, pi not fully implemented
tagging_mode = "generate_ddl" # generate_ddl will also create a log table
api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

from pydantic import BaseModel, Field, Extra, ValidationError, ConfigDict
from openai import OpenAI
import os
from pydantic.dataclasses import dataclass
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from mlflow.types.llm import TokenUsageStats
import json
import re
from typing import List, Dict, Any, Literal, Tuple
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, Row
from pyspark.sql.functions import col, struct, to_timestamp
from typing import List, Dict, Any
from pyspark.sql import DataFrame
import nest_asyncio
from pyspark.sql import Row
from datetime import datetime
import mlflow
from pyspark.sql.functions import lit
import time
import random

# COMMAND ----------

os.environ["DATABRICKS_TOKEN"]=api_key
os.environ["DATABRICKS_HOST"]=base_url

# COMMAND ----------

def create_prompt_template(content, acro_content):
    return {"pi":
              [
                {
                    "role": "system",
                    "content": f"You are an AI assistant trying to help identify personally identifying information. Please only respond with a dictionary in the format given in the user instructions. Do NOT use content from the examples given in the prompts. The examples are provided to help you understand the format of the prompt. Do not add a note after the dictionary, and do not provide any offensive or dangerous content." 
                },
                {
                    "role": "user",
                    "content": f"""Please look at each column in {content} and identify if the content represents a person, an address, an email, or a potentially valid national ID for a real country and provide a probability that it represents personally identifying information - confidence - scaled from 0 to 1. In addition, provide a classification for PCI or PHI if there is some probability that these are true. 
                    
                    Content will be provided as a Python dictionary in a string, formatted like this example, but the table name and column names might vary: {{"table_name": "finance.restricted.monthly_recurring_revenue", "column_names": ["name", "address", "email", "ssn", "religion"], "column_contents": [["John Johnson", "123 Main St", "jj@msn.com", "664-35-1234", "Buddhist"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "664-35-1234", "Episcopalian"]]}}. Please provide the response as a dictionary in a string. Options for classification are 'phi', 'none', or 'pci'. phi is health information that is tied to pi.

                    pi and pii are synonyms for our purposes, and not used as a legal term but as a way to distinguish individuals from one another. Example: if the content is the example above, then the response should look like {{"table": "pi", "column_names": ["name", "address", "email", "ssn", "religion"], "column_contents": [{{"classification": "pi", "type": "name", "confidence": 0.8}}, {{"classification": "pi", "type": "address", "confidence": 0.7}}, {{"classification": "pi", "type": "email", "confidence": 0.9}}, {{"classification": "pi", "type": "national ID", "confidence": 0.5}}, {{"classification": "none", "type": "none", "confidence": 0.95}}]}}. Modifier: if the content of a column is like {{"appointment_text": "John Johnson has high blood pressure"}}, then the classification should be "phi" and if a column appears to be a credit card number or a bank account number then it should be labeled as "pci". 'type' values allowed include 'name', 'location', 'national ID', 'email', and 'phone'. If the confidence is less than 0.5, then the classification should be 'none'. 
                    
                    Please don't respond with any other content other than the dictionary.
                    """
                }
              ],
              "comment":
              [
                {
                    "role": "system",
                    "content": """You are an AI assistant helping to generate metadata for tables and columns in Databricks. Please respond only with a dictionary in the specified format - {{"table": "finance.restricted.customer_monthly_recurring_revenue", "column_names": ["name", "address", "email", "revenue"], "column_contents": [["John Johnson", "123 Main St", "jj@msn.com", "2024-01-05", NULL], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "2024-01-05", NULL]}}. Do not use the exact content from the examples given in the prompt unless it really makes sense to. Generate descriptions based on the data and column names provided. Ensure the descriptions are detailed but concise, using between 50 to 200 words for the table comments and 20 to 50 words for column comments. If column names are informative, prioritize them; otherwise, rely more on the data content. Please unpack any acronyms, initialisms, and abbreviations unless they are in common parlance like SCUBA. Column contents will only represent a subset of data in the table so please provide information about the data in the sample but but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, any response other than the dictionary will be considered invalid. Make sure the list of column_names in the response content match the dictionary keys in the prompt input in column_contents.
                    
                    Please generate a description for the table and columns in the following string. The content string will be provided as a Python dictionary in a string, formatted like in the examples, but the table name and column names might vary. Please generate table names between 50 and 200 words considering the catalog, schema, table, as well as column names and content. Please generate column descriptions between 10 and 50 words."""
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "finance.restricted.customer_monthly_recurring_revenue", "column_contents": [{"name": ["John Johnson", "Alice Ericks"]}, {"address": ["123 Main St", "6789 Fake Ave"]}, {"email": ["jj@msn.com", "alice.ericks@aol.com"]}, {"revenue": ["$355.45", "$4850.00"]}, {"eap_created": ["2024-01-01", "2024-12-01"]}, {"delete_flag": ["True", NULL]}} and abbreviations and acronyms are here - {"EAP - enterprise architecture platform"}"""
                },
                {   "role": "assistant",
                    "content": """{"table": "Predictable recurring revenue earned from subscriptions in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "column_names": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer's first and last name.", "Customer mailing address including both the number and street name, but not including the city, state, country, or zipcode.", "Customer email address with domain name. This is a common format for email addresses. Domains seen include MSN and AOL. These are not likely domains for company email addresses.", "Monthly recurring revenue from the customer in United States dollars with two decimals for cents.", "Date when the record was created in the Enterprise Architecture Platform or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system."]}""" 
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "enterprise.master_data.customer_master", "column_contents": [{"name": ["John Johnson", "Alice Ericks", "Charlie J. Berens", None]}, {"address": ["123 Main St", "6789 Fake Ave", "42909 Johnsone Street, Dallas, Texas 44411-1111]}, {"cid": ["429184984", "443345555"]}, {"PN": ["(214) 555-0100", "(214) 555-0101"]}, {"email": ["jj@msn.com", "alice.ericks@aol.com"]}, {"dob": ["1980-01-01", "1980-01-02"]}} and abbreviations and acronyms are here - {"EAP: enterprise architecture platform, BU: business unit, PN: phone number, dob: date of birth"}"""
                },
                {   "role": "assistant",
                    "content": """{"table": "Master data for customers. Customer master data is a non-transactional information that identifies and describes a customer within a business's database. Contains customer names, addresses, phone numbers, email addresses, and date of birth. This table appears to be customer master data used enterprise-wide. There appears to be some risk of personally identifying information appearing in this table.", "column_names": ["name", "address", "cid", "PN", "email", "dob"], "column_contents": ["Customer's first and last name. In some cases, middle initial is included so it's possible that this is a free form entry field or that a variety of options are available for name.", "Customer mailing address including both the number and street name, and in some cases, but not all the city, state, and zipcode. It's possible this is free entry or taken from a variety of sources. The one zipcode in the sample data apppears to be the zipcode +4.", "Customer's unique identifier. This is a 9-digit number that appears to be a customer identifier. This is a column that appears to be a customer identifier. There is a small risk that these could be social security numbers in the United States despite not being labeled as such, as the number of digits match and they're in a customer table.", "Customer's phone number, including the area code and formatted with puncuation.", "Customer's email address with domain name.", "Customer's date of birth in the form of a string, but generally formatted as yyyy-mm-dd or yyyy-dd-mm, unclear which."]}""" 
                },
                ### Could add a third 'few-shot example' specific to each workspace or domain it's run in. It's important that you get the schema exactly right.
                {
                    "role": "user",
                    "content": f"""Content is here - {content} and abbreviations are here - {acro_content}"""                    
                }

              ]
            }



class DDLGenerator(ABC):
    def __init__(self):
        pass


class CommentGenerator(mlflow.pyfunc.ChatModel):
    def __init__(self):
        pass

    @classmethod
    def from_context(cls, context: PythonModelContext):
        chat = cls.__new__(cls)
        return chat
    
    @staticmethod
    def _format_prompt(prompt):
        # TODO: add prompt format logic here
        return formatted_prompt(prompt)

    def get_openai_client(self):
       return OpenAI(
           api_key=os.environ["DATABRICKS_TOKEN"],
           base_url=os.environ["DATABRICKS_HOST"] + "serving-endpoints")

    def load_context(self, context):
        """Instantiated OpenAI client cannot be added to load_context.
        """
        self.api_key = context.artifacts["api_key"]
        self.base_url = context.artifacts["base_url"]

    def predict(self, context, messages, params):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        prompt = self.create_prompt(messages)
        response = self.client.chat.completions.create(
            messages=prompt,
            model=params.get("model", "default-model"),
            max_tokens=params.get("max_tokens", 3000),
            temperature=params.get("temperature", 0.1)
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
    

class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    column_names: List[str]

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


class PIResponse(Response):
    model_config = ConfigDict(extra="forbid")

    column_contents: list[dict[str, Any]]


class CommentResponse(Response):
    model_config = ConfigDict(extra="forbid")

    column_contents: list[str]


class DataFrameConverter:
    def __init__(self, df, full_table_name):
        self.df = df
        self.full_table_name = full_table_name
        self.pi_input_list = self.convert_to_pi_input()
        self.comment_input_data = self.convert_to_comment_input()

    def convert_to_pi_input(self) -> Dict[str, Any]:
        return {
            "table_name": self.full_table_name,
            "column_names": self.df.columns,
            "column_contents": [row.asDict() for row in self.df.collect()]
        }

    def convert_to_comment_input(self) -> Dict[str, Any]:
        return {
            "table_name": self.full_table_name,
            "column_contents": self.df.toPandas().to_dict(orient='list')
        }


class PIChatCompletionResponse:

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


class CommentChatCompletionResponse:

    def get_comment_response(self, 
                             content: str, 
                             prompt_content: str, 
                             model: str, 
                             max_tokens: int, 
                             temperature: float,
                             retries: int = 0, 
                             max_retries: int = 5):
        try:
            chat_completion = get_response(content, prompt_content, model, max_tokens, temperature)
            response_payload = chat_completion.choices[0].message
            response = response_payload.content
            response_dict = json.loads(response)
            is_column_names_match = check_list_and_dict_keys_match(content['column_contents'], response_dict['column_names'])
            if not is_column_names_match:
                raise ValueError("Column_names do not match column_contents...")
            if not isinstance(response_dict, dict):
                print('response_dict is not a dict')
                raise ValueError("columns field is not a valid dict")
            chat_response = CommentResponse(**response_dict)
            return chat_response, response_payload
        except (ValidationError, json.JSONDecodeError, AttributeError, ValueError) as e:
            if retries < max_retries:
                print(f"Attempt {retries + 1} failed for {response}, retrying due to {e}...")
                return self.get_comment_response(content, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print("validation error - response")
                raise ValueError(f"Validation error after {max_retries} for {response} with attempts: {e}")


def load_table_names_from_csv(csv_file_path):
    df_tables = spark.read.csv(csv_file_path, header=True)    
    table_names = [row["table_name"] for row in df_tables.select("table_name").collect()]
    return table_names


def check_list_and_dict_keys_match(dict_list, string_list):
    dict_keys = dict_list.keys()
    list_matches_keys = all(item in dict_keys for item in string_list)
    keys_match_list = all(key in string_list for key in dict_keys)
    if not (list_matches_keys and keys_match_list):
        return False
    return True


def get_response(content: str, prompt_content: str, model: str, max_tokens: int, temperature: float):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
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


def get_responses(pi_input_list: List[Dict[str, Any]], 
                  comment_input_data: List[Dict[str, Any]],
                  mode: str,
                  model: str,
                  max_prompt_length: int,
                  acro_content: str,
                  max_tokens: int,
                  temperature: float) -> Tuple[PIResponse, CommentResponse]:
    print("getting responses")
    responses = {'pi': None, 'comment': None}
    if mode == "pi":
        prompt_template = create_prompt_template(pi_input_list, acro_content)
        if len(prompt_template) > max_prompt_length:
            raise ValueError("The prompt template is too long. Please reduce the number of columns or increase the max_prompt_length.")
        pi_response, message_payload = PIChatCompletionResponse().get_pi_response(content=pi_input_list, prompt_content=prompt_template['pi'],model=model, max_tokens=max_tokens, temperature=temperature)
        responses.append(pi_response)
        responses['pi'] = pi_response
    elif mode == "comment":
        prompt_template = create_prompt_template(comment_input_data, acro_content)
        if len(prompt_template) > max_prompt_length:
            raise ValueError("The prompt template is too long. Please reduce the number of columns or increase the max_prompt_length.")
        comment_response, message_payload = CommentChatCompletionResponse().get_comment_response(content=comment_input_data, prompt_content=prompt_template['comment'], model=model, max_tokens=max_tokens, temperature=temperature)
        responses['comment'] = comment_response
    else: 
        if mode not in ["pi", "comment"]:
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


def get_generated_metadata(
    catalog: str, 
    schema: str, 
    table_name: List[str],
    mode: str,
    model: str,
    max_prompt_length: int,
    acro_content: str,
    max_tokens: int,
    temperature: float,
    columns_per_call: int,
    sample_size: int
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
    full_table_name = f"{catalog}.{table_name}"
    df = spark.read.table(full_table_name)
    nrows = df.count()
    sampled_df = sample_df(df, nrows, sample_size)
    chunked_dfs = chunk_df(sampled_df, columns_per_call)
    responses = []
    for chunk in chunked_dfs:
        converter = DataFrameConverter(chunk, full_table_name)
        pi_input_list = converter.pi_input_list
        comment_input_data = converter.comment_input_data
        response = get_responses(pi_input_list, comment_input_data, mode, model, max_prompt_length, acro_content, max_tokens, temperature)
        responses.append(response)
    return responses

def review_and_generate_metadata(
    catalog: str,
    catalog_tokenizable: str,
    schema: str,
    table_names: List[str],
    mode: str,
    model: str,
    max_prompt_length: int,
    acro_content: str,
    max_tokens: int,
    temperature: float,
    columns_per_call: int,
    sample_size: int
    ) -> Tuple[DataFrame, DataFrame]:
    """
    Reviews and generates metadata for a list of tables based on the mode.

    Args:
        catalog (str): The catalog name.
        schema (str): The schema name.
        table_names (List[str]): A list of table names.
        model (str): model name
        mode (str): Mode to determine whether to process 'pi' or 'comment'

    Returns:
        Tuple[DataFrame, DataFrame]: DataFrames containing the generated metadata.
    """
    print("Review and generate metadata...")
    
    catalog = setup_params['catalog']                       
    catalog_tokenizable = setup_params['catalog_tokenizable']
    dest_schema = setup_params['dest_schema']
    table_names = table_names
    volume_name = setup_params['volume_name']
    mode = setup_params['mode']
    model = model_params['model']
    max_prompt_length = model_params['max_prompt_length']
    acro_content = model_params['acro_content']
    max_tokens = model_params['max_tokens']
    temperature = model_params['temperature']
    columns_per_call = setup_params['columns_per_call']
    sample_size = setup_params['sample_size']

    table_rows = []
    column_rows = []
    for table_name in table_names:
        responses = get_generated_metadata(catalog, schema, table_name, mode, model, max_prompt_length, acro_content, max_tokens, temperature, columns_per_call, sample_size)
        for response in responses:
            full_table_name = f"{catalog}.{table_name}"
            tokenized_full_table_name = f"{catalog_tokenizable}.{table_name}"
            if mode == "pi":
                response_dict = response['pi']
            elif mode == "comment":
                response_dict = response['comment']
            else:
                raise ValueError("Invalid mode. Use 'pi' or 'comment'.")
            table_rows = append_table_row(table_rows, full_table_name, response_dict, tokenized_full_table_name)
            column_rows = append_column_rows(column_rows, full_table_name, response_dict, tokenized_full_table_name)
    ### TODO: add all the table comments to an additional summarizer call rather than taking just the first one.         
    return rows_to_df(column_rows), rows_to_df(table_rows)


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


def populate_log_table(df, current_user, model, sample_size, max_tokens, temperature, columns_per_call, base_path):
        return (df.withColumn("current_user", lit(current_user))
          .withColumn("model", lit(model))
          .withColumn("sample_size", lit(sample_size))
          .withColumn("max_tokens", lit(max_tokens))
          .withColumn("temperature", lit(temperature))
          .withColumn("columns_per_call", lit(columns_per_call))
          #.withColumn("_ddl_written_to", lit(base_path))
          .withColumn("status", lit("No Volume specified..."))
        )
                    

def log_metadata_generation(df: DataFrame, catalog: str, dest_schema: str, table_names: List[str], volume_name: str) -> None:    
    #display(df)
    df.write.mode('append').saveAsTable(f"{catalog}.{dest_schema}.metadata_generation_log")


def filter_and_write_ddl(df: DataFrame, 
                     catalog: str,
                     dest_schema: str, 
                     table_name: List[str], 
                     base_path: str, 
                     current_user: str, 
                     current_date: str, 
                     model, 
                     max_prompt_length, 
                     acro_content, 
                     max_tokens, 
                     temperature, 
                     columns_per_call, 
                     sample_size) -> DataFrame:
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
    df = df.filter(df['table'] == f"{catalog}.{table_name}")
    ddl_statements = df.select("ddl").collect()
    table_name = re.sub(r'[^\w\s/]', '_', table_name)
    file_path = os.path.join(base_path, f"{table_name}.sql")
    try:
        write_ddl_to_volume(file_path, base_path, ddl_statements)
        df = df.withColumn("status", lit("Success"))
        log_metadata_generation(df, catalog, dest_schema, [table], base_path)
    except Exception as e:
        print(f"Error writing DDL to volume: {e}. Check if Volume exists and if your permissions are correct.")        
        df = df.withColumn("status", lit("Failed writing to volume..."))
        log_metadata_generation(df, catalog, dest_schema, [table], base_path)
    

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
            #print(f"Writing DDL: {row['ddl']}")
            file.write(f"{row['ddl']}\n")


def create_and_persist_ddl(df: DataFrame,
                           setup_params: Dict[str, Any],
                           model_params: Dict[str, Any],
                           table_names: List[str]
                           ) -> None:
    """
    Writes the DDL statements from the DataFrame to a volume as SQL files.

    Args:
        df (DataFrame): The DataFrame containing the DDL statements.
        catalog (str): The catalog name.
        dest_schema (str): The destination schema name.
        table_names (List[str]): A list of table names.
        volume_name (str): The volume name.
    """

    catalog = setup_params['catalog']                       
    catalog_tokenizable = setup_params['catalog_tokenizable']
    dest_schema = setup_params['dest_schema']
    table_names = table_names
    volume_name = setup_params['volume_name']
    mode = setup_params['mode']
    model = model_params['model']
    max_prompt_length = model_params['max_prompt_length']
    acro_content = model_params['acro_content']
    max_tokens = model_params['max_tokens']
    temperature = model_params['temperature']
    columns_per_call = setup_params['columns_per_call']
    sample_size = setup_params['sample_size']


    print("Running create and persist ddl...")
    current_user = get_current_user()
    current_date = datetime.now().strftime('%Y%m%d')
    if volume_name:
        base_path = f"/Volumes/{catalog}/{dest_schema}/{volume_name}/{current_user}/{current_date}"
        for table_name in table_names:
            print(f"Writing DDL for {table_name}...")
            table_df = df[f'{mode}_table_df']            
            table_df = populate_log_table(table_df, current_user, model, sample_size, max_tokens, temperature, columns_per_call, volume_name)
            modified_path = re.sub(r'[^\w\s/]', '_', base_path)
            filter_and_write_ddl(table_df, catalog, dest_schema, table_name, modified_path, current_user, current_date, model, max_prompt_length, acro_content, max_tokens, temperature, columns_per_call, sample_size)
            column_df = df[f'{mode}_column_df']
            column_df = populate_log_table(column_df, current_user, model, sample_size, max_tokens, temperature, columns_per_call, volume_name)
            modified_path = re.sub(r'[^\w\s/]', '_', base_path)
            filter_and_write_ddl(column_df, catalog, dest_schema, table_name, modified_path, current_user, current_date, model, max_prompt_length, acro_content, max_tokens, temperature, columns_per_call, sample_size)
    else:
        print("No volume name provided. Not writing DDL to volume...")
        table_df = populate_log_table(df['comment_table_df'], current_user, model, sample_size, max_tokens, temperature, columns_per_call, volume_name)
        log_metadata_generation(table_df, catalog, dest_schema, table_names, volume_name)
        column_df = populate_log_table(df['comment_column_df'], current_user, model, sample_size, max_tokens, temperature, columns_per_call, volume_name)
        log_metadata_generation(column_df, catalog, dest_schema, table_names, volume_name)


def process_and_add_ddl(setup_params: Dict[str, Any], model_params: Dict[str, Any], tabe_names: List[str]) -> DataFrame:
    """
    Processes the metadata, splits the DataFrame based on 'table' values, applies DDL functions, and returns a unioned DataFrame.

    Args:
        catalog (str): The catalog name data is being read from and written to.
        dest_schema (str): The destination schema name.
        table_names (List[str]): A list of table names.
        model (str): The model name.

    Returns:
        DataFrame: The unioned DataFrame with DDL statements added.
    """
    print("Process and add ddl...")
    column_df, table_df = review_and_generate_metadata(setup_params, model_params, table_names)
    dfs = {}
    if mode == "comment":
        # TODO: turn this into a class
        table_df = add_ddl_to_table_comment_df(table_df, "ddl")
        column_df = add_ddl_to_column_comment_df(column_df, "ddl")
        dfs['comment_table_df'] = table_df.limit(1)
        dfs['comment_column_df'] = column_df
    elif mode == "pi":
        table_df = add_column_ddl_to_pi_df(table_df, "ddl")
        dfs['pi_table_df'] = table_df.limit(1)
        dfs['pi_column_df'] = column_df    
    return dfs


def create_schema(setup_params: Dict[str, Any]) -> None:
    if setup_params["volume_name"]:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {setup_params['catalog']}.{setup_params['dest_schema']}.{setup_params['volume_name']}")


def generate_and_persist_comments(setup_params: Dict[str, Any], model_params: Dict[str, Any]) -> None:
    """
    Generates and persists comments for tables based on the provided setup and model parameters.

    Args:
        setup_params (Dict[str, Any]): Dictionary containing setup parameters.
        model_params (Dict[str, Any]): Dictionary containing model parameters.
    """
    create_schema(setup_params)
    for table in setup_params["table_names"]:
        print(f"Processing table {table}...")        
        df = process_and_add_ddl(setup_params, model_params, table_names=[table])        
        print(f"Generating and persisting ddl for {table}...")
        create_and_persist_ddl(df, setup_params, model_params, table_names=[table])

# COMMAND ----------


setup_params = {
    "catalog": catalog,
    "catalog_tokenizable": catalog_tokenizable,
    "dest_schema": dest_schema,
    "table_names": table_names,
    "generation_mode": generation_mode,
    "model": model,
    "max_prompt_length": max_prompt_length,
    "volume_name": volume_name,
    "acro_content": acro_content,
    "columns_per_call": columns_per_call,
    "sample_size": sample_size,
}

model_params = {
    "max_tokens": max_tokens,
    "temperature": temperature,
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Now run the code

# COMMAND ----------

generate_and_persist_comments(setup_params, model_params)

# COMMAND ----------

df = spark.read.table(f"{catalog}.{dest_schema}.metadata_generation_log")

# COMMAND ----------

display(df)
