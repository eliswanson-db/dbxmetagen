# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys

sys.path.append("../")
import os

from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.processing import split_table_names, sanitize_email
from src.dbxmetagen.error_handling import exponential_backoff
from src.dbxmetagen.ddl_regenerator import (
    process_metadata_file,
    load_metadata_file,
    replace_comment_in_ddl,
    replace_pii_tags_in_ddl,
)

current_user = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
dbutils.widgets.text("reviewed_file_name", "")
dbutils.widgets.text("mode", "comment")
file_name = dbutils.widgets.get("reviewed_file_name")
mode = dbutils.widgets.get("mode")
review_variables = {
    "reviewed_file_name": file_name,
    "current_user": current_user,
    "mode": mode
}

# COMMAND ----------


def main(kwargs, input_file):
    spark_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
    config = MetadataConfig(**kwargs)
    if 'ml' not in spark_version and 'excel' in (config.review_input_file_type, config.reviewable_output_format):
        raise ValueError("Excel writes in dbxmetagen are not supported on standard runtimes. Please change your output file type to tsv or sql if appropriate.")
    process_metadata_file(config, input_file)


# COMMAND ----------

main(review_variables, file_name)

# COMMAND ----------


