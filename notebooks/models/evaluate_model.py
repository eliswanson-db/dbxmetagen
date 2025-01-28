# Databricks notebook source
# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from src.dbxmetagen.processing import split_table_names, instantiate_metadata_objects
from src.dbxmetagen.prompts import Prompt, PromptFactory, PIPrompt, CommentPrompt
from mlflow.models import set_model

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("dest_schema", "default")
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
from openai import OpenAI
import pandas as pd
from openai.types.chat.chat_completion import ChatCompletion
from mlflow.types.llm import ChatResponse, ChatChoice, ChatChoiceLogProbs, ChatMessage
from src.dbxmetagen.prompts import Prompt
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse
from src.dbxmetagen.model_logging import convert_to_chat_response, get_latest_model_version

# COMMAND ----------

api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"]=api_key
os.environ["DATABRICKS_HOST"]=base_url
mlflow.set_registry_uri("databricks-uc")
METADATA_PARAMS = instantiate_metadata_objects(catalog_name, dest_schema, table_names, mode, base_url)
config = MetadataConfig(**METADATA_PARAMS)
config.SETUP_PARAMS.update({'add_metadata': False}) # add_metadata makes a spark call so it has to be excluded from the model logging
comment_model = CommentGeneratorModel()
comment_model.from_context(config)

# COMMAND ----------

### Working on implementing evaluation below

# COMMAND ----------

latest_model_version = get_latest_model_version(f"{full_model_name}")

# COMMAND ----------

import pandas as pd

data = {
  "request": [
      "What is the difference between reduceByKey and groupByKey in Spark?",
      {
          "messages": [
              {
                  "role": "user",
                  "content": "How can you minimize data shuffling in Spark?"
              }
          ]
      },
      {
          "query": "Explain broadcast variables in Spark. How do they enhance performance?",
          "history": [
              {
                  "role": "user",
                  "content": "What are broadcast variables?"
              },
              {
                  "role": "assistant",
                  "content": "Broadcast variables allow the programmer to keep a read-only variable cached on each machine."
              }
          ]
      }
  ],

  "expected_response": [
    "expected response for first question",
    "expected response for second question",
    "expected response for third question"
  ]
}

eval_dataset = pd.DataFrame(data)

# COMMAND ----------

evaluation_results = mlflow.evaluate(
    data=eval_set_df,  # pandas DataFrame with just the evaluation set
    model = f"models:/{full_model_name}/{latest_model_version}",  # 1 is the version number
    model_type="databricks-agent",
)
