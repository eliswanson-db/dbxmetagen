# Databricks notebook source
# MAGIC %md
# MAGIC # GenAI-Assisted Metadata Utility (a.k.a `dbxmetagen`)

# COMMAND ----------

# MAGIC %md
# MAGIC #`dbxmetagen` Overview
# MAGIC ### This is a utility to help generate high quality descriptions for tables and columns to enhance enterprise search and data governance, identify and classify PI, improve Databricks Genie performance for Text-2-SQL, and generally help curate a high quality metadata layer and data dictionary for enterprise data.
# MAGIC
# MAGIC While Databricks does offer high quality [AI Generated Documentation](https://docs.databricks.com/en/comments/ai-comments.html), and PI identification these are not always customizable to customers' needs or integrable into devops workflows without additional effort. Prompts and model choice are not adjustable by customers, and there are a variety of customization options that customers have asked for. This utility, `dbxmetagen`, helps generate table and column descriptions at scale, as well as identifying and classifying various forms of sensitive information. Eventually Databricks utilities will undoubtedly be more flexible, but this solution accelerator can allow customers to close the gap in a customizable fashion until then.
# MAGIC
# MAGIC Please review the readme for full details and documentation.
# MAGIC
# MAGIC ###Disclaimer
# MAGIC AI generated comments are not always accurate and comment DDLs should be reviewed prior to modifying your tables. Databricks strongly recommends human review of AI-generated comments to check for inaccuracies. While the model has been guided to avoids generating harmful or inappropriate descriptions, you can mitigate this risk by setting up [AI Guardrails](https://docs.databricks.com/en/ai-gateway/index.html#ai-guardrails) in the AI Gateway where you connect your LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC # Library installs

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Library imports

# COMMAND ----------

import sys

sys.path.append("../")

# COMMAND ----------

import os
import json
from src.dbxmetagen.prompts import Prompt, PIPrompt, CommentPrompt, PromptFactory
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import (
    PIResponse,
    CommentResponse,
    Response,
    MetadataGenerator,
    CommentGenerator,
    PIIdentifier,
    MetadataGeneratorFactory,
)
from src.dbxmetagen.processing import split_table_names, sanitize_email
from src.dbxmetagen.error_handling import exponential_backoff
from src.dbxmetagen.main import main

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up widgets

# COMMAND ----------

dbutils.widgets.dropdown("cleanup_control_table", "false", ["true", "false"])
dbutils.widgets.dropdown("mode", "comment", ["comment", "pi"])
dbutils.widgets.text("env", "")
dbutils.widgets.text("table_names", "")

# Configuration widgets - these allow app parameters to override variables.yml defaults
dbutils.widgets.text("catalog_name", "")
dbutils.widgets.dropdown("allow_data", "true", ["true", "false"])
dbutils.widgets.text("sample_size", "5")
dbutils.widgets.dropdown("apply_ddl", "false", ["true", "false"])
dbutils.widgets.text("model", "")
dbutils.widgets.text("temperature", "0.1")
dbutils.widgets.text("schema_name", "")
dbutils.widgets.text("volume_name", "")
dbutils.widgets.text("max_tokens", "4096")
dbutils.widgets.text("columns_per_call", "5")
dbutils.widgets.dropdown("add_metadata", "true", ["true", "false"])
dbutils.widgets.dropdown("allow_data_in_comments", "true", ["true", "false"])
dbutils.widgets.dropdown("include_deterministic_pi", "true", ["true", "false"])
dbutils.widgets.text("ddl_output_format", "sql")
dbutils.widgets.text("reviewable_output_format", "tsv")

# COMMAND ----------

table_names = dbutils.widgets.get("table_names")
mode = dbutils.widgets.get("mode")
env = dbutils.widgets.get("env")
cleanup_control_table = dbutils.widgets.get("cleanup_control_table")

# Get configuration parameters from widgets (app overrides)
widget_config = {
    "catalog_name": dbutils.widgets.get("catalog_name"),
    "allow_data": dbutils.widgets.get("allow_data").lower() == "true",
    "sample_size": (
        int(dbutils.widgets.get("sample_size"))
        if dbutils.widgets.get("sample_size")
        else None
    ),
    "apply_ddl": dbutils.widgets.get("apply_ddl").lower() == "true",
    "model": dbutils.widgets.get("model"),
    "temperature": (
        float(dbutils.widgets.get("temperature"))
        if dbutils.widgets.get("temperature")
        else None
    ),
    "schema_name": dbutils.widgets.get("schema_name"),
    "volume_name": dbutils.widgets.get("volume_name"),
    "max_tokens": (
        int(dbutils.widgets.get("max_tokens"))
        if dbutils.widgets.get("max_tokens")
        else None
    ),
    "columns_per_call": (
        int(dbutils.widgets.get("columns_per_call"))
        if dbutils.widgets.get("columns_per_call")
        else None
    ),
    "add_metadata": dbutils.widgets.get("add_metadata").lower() == "true",
    "allow_data_in_comments": dbutils.widgets.get("allow_data_in_comments").lower()
    == "true",
    "include_deterministic_pi": dbutils.widgets.get("include_deterministic_pi").lower()
    == "true",
    "ddl_output_format": dbutils.widgets.get("ddl_output_format"),
    "reviewable_output_format": dbutils.widgets.get("reviewable_output_format"),
}

# Filter out empty string values to let variables.yml defaults take precedence
widget_config = {k: v for k, v in widget_config.items() if v != "" and v is not None}
context_json = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
)
context = json.loads(context_json)
job_id = context.get("tags", {}).get("jobId", None)
current_user = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
notebook_variables = {
    "table_names": table_names,
    "mode": mode,
    "env": env,
    "current_user": current_user,
    "cleanup_control_table": cleanup_control_table,
    "job_id": job_id,
}

# Merge widget configuration overrides
notebook_variables.update(widget_config)
api_key = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["DATABRICKS_TOKEN"] = api_key

# COMMAND ----------

main(notebook_variables)
