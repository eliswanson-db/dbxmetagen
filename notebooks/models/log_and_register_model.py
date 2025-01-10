# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from src.dbxmetagen.processing import split_table_names, instantiate_metadata_objects
from src.dbxmetagen.prompts import Prompt, PromptFactory, PIPrompt, CommentPrompt
from mlflow.models import set_model

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("dest_schema", "metadata_results")
dbutils.widgets.text("table_names", "")
dbutils.widgets.text("mode", "comment")
dbutils.widgets.text("model_name", "comment_generator")
dbutils.widgets.text("base_url", "https://adb-830292400663869.9.azuredatabricks.net")

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
dest_schema = dbutils.widgets.get("dest_schema")
table_names = split_table_names(dbutils.widgets.get("table_names"))
mode = dbutils.widgets.get("mode")
base_url = dbutils.widgets.get("base_url")
model_name = dbutils.widgets.get("model_name")
full_model_name = f"{catalog_name}.{dest_schema}.{model_name}"

# COMMAND ----------

import os
from src.dbxmetagen.comment_generator import CommentGeneratorModel
from src.dbxmetagen.config import MetadataConfig
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow
import mlflow
from src.dbxmetagen.metadata_generator import CommentGenerator
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from mlflow.types.llm import ChatResponse, ChatChoice, ChatChoiceLogProbs
from src.dbxmetagen.prompts import Prompt
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse
from openai import OpenAI
import pandas as pd
from openai.types.chat.chat_completion import ChatCompletion
from mlflow.types.llm import ChatResponse, ChatChoice, ChatChoiceLogProbs, ChatMessage

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

### Set key for authenticating to AI Gateway
api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"]=api_key
os.environ["DATABRICKS_HOST"]=base_url

# COMMAND ----------

METADATA_PARAMS = instantiate_metadata_objects(catalog_name, dest_schema, table_names, mode, base_url)
config = MetadataConfig(**METADATA_PARAMS)
config.SETUP_PARAMS.update({'add_metadata': False})
comment_model = CommentGeneratorModel()
comment_model.from_context(config)

# COMMAND ----------

def convert_to_chat_response(chat_completion):
    choices = [
        ChatChoice(
            index=choice.index,
            message=ChatMessage(role=choice.message.role, content=choice.message.content) if isinstance(choice.message, ChatCompletionMessage) else choice.message,
            finish_reason=choice.finish_reason
        )
        for choice in chat_completion.choices
    ]
    
    return ChatResponse(
        id=chat_completion.id,
        object=chat_completion.object,
        created=chat_completion.created,
        model=chat_completion.model,
        choices=choices,
        usage=chat_completion.usage.to_dict() if hasattr(chat_completion.usage, 'to_dict') else chat_completion.usage
    )

# COMMAND ----------

model_input = [ 
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "hr.employees.employee_performance_reviews", "column_contents": {"index": [0,1,2,3,4], "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "data": [["E123", "2023-06-15", "4.5", "Excellent work throughout the year", "Yes"], ["E456", "2023-06-15", "3.2", "Needs improvement in meeting deadlines", "No"], ["E789", "2023-06-15", "4.8", "Outstanding performance and leadership", "Yes"], ["E101", "2023-06-15", "2.9", "Struggles with teamwork", "No"], ["E112", "2023-06-15", "3.7", "Consistently meets expectations", "Yes"]], "column_metadata": {"employee_id": {"col_name": "employee_id", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "4", "max_col_len": "4"}, "review_date": {"col_name": "review_date", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "10", "max_col_len": "10"}, "performance_score": {"col_name": "performance_score", "data_type": "string", "num_nulls": "0", "distinct_count": "50", "avg_col_len": "3", "max_col_len": "3"}, "manager_comments": {"col_name": "manager_comments", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "30", "max_col_len": "100"}, "promotion_recommendation": {"col_name": "promotion_recommendation", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "3", "max_col_len": "3"}}}} and abbreviations and acronyms are here - {"EID - employee ID"}"""
                },
                {   
                    "role": "assistant",
                    "content": """{"table": "Employee performance reviews conducted annually. This table includes employee IDs, review dates, performance scores, manager comments, and promotion recommendations.", "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "column_contents": ["Unique identifier for each employee. This field is always populated and has 100 distinct values. The average and maximum column lengths are both 4, indicating a consistent format for employee IDs.", "Date when the performance review was conducted. This field is always populated and has only one distinct value in the sample, suggesting that all reviews were conducted on the same date. The average and maximum column lengths are both 10, consistent with the date format 'YYYY-MM-DD'.", "Performance score given by the manager, typically on a scale of 1 to 5. This field is always populated and has 50 distinct values. The average and maximum column lengths are both 3, indicating a consistent format for performance scores.", "Comments provided by the manager during the performance review. This field is always populated and has 100 distinct values, one for each employee, so these are fairly unique comments for each employee. The average column length is 30 and the maximum column length is 100, indicating a wide range of comment lengths, though given the skew there are probably a large number of very short comments.", "Recommendation for promotion based on the performance review. This field is always populated and has two distinct values: 'Yes' and 'No'. The average and maximum column lengths are both 3, indicating a consistent format for promotion recommendations."]}"""
                },
                {
                    "role": "user",
                    "content": "Content is here and abbreviations are here."                  
                }
              ]


predicted_choices = comment_model.predict(model_input=model_input, params=None)
chat_response = convert_to_chat_response(predicted_choices)
chat_response_dict = chat_response.__dict__ if hasattr(chat_response, '__dict__') else chat_response
if not isinstance(chat_response_dict, dict):
    chat_response_dict = chat_response_dict.__dict__
chat_response_df = pd.DataFrame([chat_response_dict])
signature = infer_signature(model_input, chat_response_df)

# COMMAND ----------

eval_data =  [
                #[
                {
                    "role": "system",
                    "content": """You are an AI assistant helping to generate metadata for tables and columns in Databricks."""
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "hr.employees.employee_performance_reviews", "column_contents": {"index": [0,1,2,3,4], "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "data": [["E123", "2023-06-15", "4.5", "Excellent work throughout the year", "Yes"], ["E456", "2023-06-15", "3.2", "Needs improvement in meeting deadlines", "No"], ["E789", "2023-06-15", "4.8", "Outstanding performance and leadership", "Yes"], ["E101", "2023-06-15", "2.9", "Struggles with teamwork", "No"], ["E112", "2023-06-15", "3.7", "Consistently meets expectations", "Yes"]], "column_metadata": {"employee_id": {"col_name": "employee_id", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "4", "max_col_len": "4"}, "review_date": {"col_name": "review_date", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "10", "max_col_len": "10"}, "performance_score": {"col_name": "performance_score", "data_type": "string", "num_nulls": "0", "distinct_count": "50", "avg_col_len": "3", "max_col_len": "3"}, "manager_comments": {"col_name": "manager_comments", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "30", "max_col_len": "100"}, "promotion_recommendation": {"col_name": "promotion_recommendation", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "3", "max_col_len": "3"}}}} and abbreviations and acronyms are here - {"EID - employee ID"}"""
                },
                {   
                    "role": "assistant",
                    "content": """{"table": "Employee performance reviews conducted annually. This table includes employee IDs, review dates, performance scores, manager comments, and promotion recommendations.", "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "column_contents": ["Unique identifier for each employee. This field is always populated and has 100 distinct values. The average and maximum column lengths are both 4, indicating a consistent format for employee IDs.", "Date when the performance review was conducted. This field is always populated and has only one distinct value in the sample, suggesting that all reviews were conducted on the same date. The average and maximum column lengths are both 10, consistent with the date format 'YYYY-MM-DD'.", "Performance score given by the manager, typically on a scale of 1 to 5. This field is always populated and has 50 distinct values. The average and maximum column lengths are both 3, indicating a consistent format for performance scores.", "Comments provided by the manager during the performance review. This field is always populated and has 100 distinct values, one for each employee, so these are fairly unique comments for each employee. The average column length is 30 and the maximum column length is 100, indicating a wide range of comment lengths, though given the skew there are probably a large number of very short comments.", "Recommendation for promotion based on the performance review. This field is always populated and has two distinct values: 'Yes' and 'No'. The average and maximum column lengths are both 3, indicating a consistent format for promotion recommendations."]}"""
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "finance.restricted.customer_monthly_recurring_revenue", "column_contents": {"index": [0,1], "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$355.45", "2024-01-01", "True"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$4850.00", "2024-12-01", "False"]}, "column_metadata": {"name": {"col_name": "name", "data_type": "string", "num_nulls": "0", "distinct_count": "5", "avg_col_len": "16", "max_col_len": "23"}, "address": {"col_name": "address", "data_type": "string", "num_nulls": "0", "distinct_count": "46", "avg_col_len": "4", "max_col_len": "4"}, "email": {"col_name": "email", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "15", "max_col_len": "15"}, "revenue": {"col_name": "revenue", "data_type": "string", "num_nulls": "0", "distinct_count": "10", "avg_col_len": "11", "max_col_len": "11"}, "eap_created": {"col_name": "eap_created", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "11", "max_col_len": "11"}, "delete_flag": {"col_name": "delete_flag", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "11", "max_col_len": "11"}}}} and abbreviations and acronyms are here - {"EAP - enterprise architecture platform"}"""               
                }
            #]
        ]

prediction =  [
            """{"table": "Predictable recurring revenue earned from customers in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer's first and last name.", "Customer mailing address including both the number and street name, but not including the city, state, country, or zipcode. Stored as a string and populated in all cases. At least 46 distinct values.", "Customer email address with domain name. This is a common format for email addresses. Domains seen include MSN and AOL. These are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "Monthly recurring revenue from the customer in United States dollars with two decimals for cents. This field is never null, and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the Enterprise Architecture Platform or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column - based on the sample and the metadata it appears that every value is set to False, but as a string rather than as a boolean value."]}""" ,
        ]



with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        registered_model_name="dbxmetagen.default.comment_generator_model",
        artifact_path="comment_generator",
        model_config=config.MODEL_PARAMS,
        python_model=comment_model,
        code_paths=["../src/dbxmetagen/"],
        pip_requirements=["../requirements.txt"],
        #input_example=model_input
        signature=signature
    )


"""Will need to figure out how to get the evaluation to work with a list of dicts as the prompt."""

    
latest_model_version = get_latest_model_version("dbxmetagen.default.comment_generator_model")

loaded = mlflow.pyfunc.load_model(f"models:/dbxmetagen.default.comment_generator_model/{latest_model_version}")
print(loaded.model_config)
loaded.unwrap_python_model().predict(model_input=model_input, params=None)
