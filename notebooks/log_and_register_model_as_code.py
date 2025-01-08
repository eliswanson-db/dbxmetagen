# Databricks notebook source
import mlflow

from src.dbxmetagen.add_model import AddModel

from mlflow.models import infer_signature
import pandas as pd


model = AddModel()

mlflow.set_registry_uri('databricks-uc')

model_input = pd.DataFrame([{'x': 1.0, 'y': 2.0}])

signature = infer_signature(model_input)
model_output = model.predict(model_input=model_input, params=None)

signature = infer_signature(model_input, model_output)


mlflow.pyfunc.log_model(
    registered_model_name="dbxmetagen.default.add_n_model",
    artifact_path="add_n_model",
    python_model=model,
    code_paths=["../src/dbxmetagen/"],
    signature=signature
)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model("models:/dbxmetagen.default.add_n_model/6")

# COMMAND ----------

model_output = model.predict(model_input=model_input, params=None)

# COMMAND ----------

print(model_output)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("dest_schema", "metadata_results")
dbutils.widgets.text("table_names", "")
dbutils.widgets.text("mode", "comment")
dbutils.widgets.text("model_name", "comment_generator")
dbutils.widgets.text("base_url", "https://adb-830292400663869.9.azuredatabricks.net")

# COMMAND ----------

import mlflow

from src.dbxmetagen.comment_generator import CommentGeneratorModel
import mlflow
from src.dbxmetagen.metadata_generator import CommentGenerator
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from src.dbxmetagen.prompts import Prompt
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse
from src.dbxmetagen.processing import split_table_names, instantiate_metadata_objects
from src.dbxmetagen.prompts import Prompt, PromptFactory, PIPrompt, CommentPrompt
from mlflow.models import set_model

from mlflow.models import infer_signature
import pandas as pd

mlflow.set_registry_uri('databricks-uc')


METADATA_PARAMS = instantiate_metadata_objects(catalog_name, dest_schema, table_names, mode, base_url)
metadata_config = MetadataConfig(**METADATA_PARAMS)
metadata_config.SETUP_PARAMS.update({'add_metadata': False})
keys_to_extract = ['model', 'temperature', 'max_tokens']
config = {key: metadata_config.SETUP_PARAMS[key] for key in keys_to_extract}


model = CommentGeneratorModel()


model_input = [
                {   
                    "role": "assistant",
                    "content": """{"table": "Employee performance reviews conducted annually. This table includes employee IDs, review dates, performance scores, manager comments, and promotion recommendations.", "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "column_contents": ["Unique identifier for each employee. This field is always populated and has 100 distinct values. The average and maximum column lengths are both 4, indicating a consistent format for employee IDs.", "Date when the performance review was conducted. This field is always populated and has only one distinct value in the sample, suggesting that all reviews were conducted on the same date. The average and maximum column lengths are both 10, consistent with the date format 'YYYY-MM-DD'.", "Performance score given by the manager, typically on a scale of 1 to 5. This field is always populated and has 50 distinct values. The average and maximum column lengths are both 3, indicating a consistent format for performance scores.", "Comments provided by the manager during the performance review. This field is always populated and has 100 distinct values, one for each employee, so these are fairly unique comments for each employee. The average column length is 30 and the maximum column length is 100, indicating a wide range of comment lengths, though given the skew there are probably a large number of very short comments.", "Recommendation for promotion based on the performance review. This field is always populated and has two distinct values: 'Yes' and 'No'. The average and maximum column lengths are both 3, indicating a consistent format for promotion recommendations."]}"""
                },
                {
                    "role": "user",
                    "content": "Content is here and abbreviations are here."                  
                }
              ]




#signature = infer_signature(model_input)
model_output = model.predict(model_input=model_input, params=None)

signature = infer_signature(model_input, model_output)


mlflow.pyfunc.log_model(
    registered_model_name="dbxmetagen.default.comment_generator_model",
    artifact_path="comment_generator",
    python_model=model,
    code_paths=["../src/dbxmetagen/"],
    signature=signature
)

# COMMAND ----------


