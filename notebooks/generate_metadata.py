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
from src.dbxmetagen.databricks_utils import (
    setup_databricks_environment,
    get_job_context,
)
from src.dbxmetagen.main import main

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up widgets and environment

# COMMAND ----------

dbutils.widgets.dropdown("cleanup_control_table", "false", ["true", "false"])
dbutils.widgets.dropdown("mode", "comment", ["comment", "pi"])
dbutils.widgets.text("env", "")
dbutils.widgets.text("table_names", "")

# COMMAND ----------

# Get widget values and set up environment
table_names = dbutils.widgets.get("table_names")
mode = dbutils.widgets.get("mode")
env = dbutils.widgets.get("env")
cleanup_control_table = dbutils.widgets.get("cleanup_control_table")

# Set up Databricks environment variables and get current user
current_user = setup_databricks_environment(dbutils)

# Get job context if running in a job
job_id = get_job_context(dbutils)

notebook_variables = {
    "table_names": table_names,
    "mode": mode,
    "env": env,
    "current_user": current_user,
    "cleanup_control_table": cleanup_control_table,
    "job_id": job_id,
}

# COMMAND ----------

main(notebook_variables)
