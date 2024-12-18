# Databricks notebook source
# MAGIC %md
# MAGIC # GenAI-Assisted Metadata Generation (a.k.a `dbxmetagen`)

# COMMAND ----------

# MAGIC %md
# MAGIC #`dbxmetagen` Overview
# MAGIC ### This is a utlity to help generate high quality descriptions for tables and columns to enhance enterprise search and data governance, improve Databricks Genie performance for Text-2-SQL, and generally help curate a high quality metadata layer for enterprise data.
# MAGIC
# MAGIC While Databricks does offer [AI Generated Documentation](https://docs.databricks.com/en/comments/ai-comments.html), this is not sustainable at scale as a human must manually select and approve AI generated metadata. This utility, `dbxmetagen`, helps generate table and column descriptions at scale. 
# MAGIC
# MAGIC ###Disclaimer
# MAGIC AI generated comments are not always accurate and comment DDLs should be reviewed prior to modifying your tables. Databricks strongly recommends human review of AI-generated comments to check for inaccuracies. While the model has been guided to avoids generating harmful or inappropriate descriptions, you can mitigate this risk by setting up [AI Guardrails](https://docs.databricks.com/en/ai-gateway/index.html#ai-guardrails) in the AI Gateway where you connect your LLM. 
# MAGIC
# MAGIC ###Solution Overview:
# MAGIC There are a few key sections in this notebook: 
# MAGIC - Library installs and setup using the config referenced in `dbxmetagen/src/config.py`
# MAGIC - Function definitions for:
# MAGIC   - Retrieving table and column information from the list of tables provided in `table_names.csv`
# MAGIC   - Sampling data from those tables, with exponential backoff, to help generate more accurate metadata, especially for columns with categorical data, that will also indicate the structure of the data. This is particularly helpful for [Genie](https://www.databricks.com/product/ai-bi/genie). This sampling also checks for nulls.
# MAGIC   - Use of `Pydantic` to ensure that LLM metadata generation conforms to a particular format. This is also used for DDL generation to ensure that the DDL is always runnable. 
# MAGIC   - Creation of a log table keeping track of tables read/modified
# MAGIC   - Creation of DDL scripts, one for each table, that have the DDL commands to `ALTER TABLE` to add comments to table and columns. This is to help integrate with your CI/CD processes, in case you do not have access in a production environment
# MAGIC - Application of the functions above to generate metadata and DDL for the list of tables provided in `dbxmetagen/table_names.csv` 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Pydantic if needed

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

#%pip install -U pydantic==2.9.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

import os
from src.dbxmetagen.prompts import create_prompt_template
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse, CommentResponse, Response, MetadataGenerator, CommentGenerator, PIIdentifier, MetadataGeneratorFactory
from src.dbxmetagen.processing import (
    tag_table,
    write_to_log_table,
    count_df_columns,
    chunk_df,
    get_extended_metadata_for_column,
    sample_df,
    append_table_row,
    append_column_rows,
    rows_to_df,
    add_ddl_to_column_comment_df,
    add_ddl_to_table_comment_df,
    add_column_ddl_to_pi_df,
    get_current_user,
    df_to_sql_file,
    populate_log_table,
    mark_as_deleted,
    log_metadata_generation,
    filter_and_write_ddl,
    create_folder_if_not_exists,
    write_ddl_to_volume,
    create_and_persist_ddl,
    get_generated_metadata,
    review_and_generate_metadata,
    replace_catalog_name,
    apply_comment_ddl,
    process_and_add_ddl,
    add_ddl_to_dfs,
    summarize_table_content,
    setup_ddl,
    create_tables,
    instantiate_metadata_objects,
    generate_and_persist_comments,
    setup_queue,
    ensure_fully_scoped_table_names,
    upsert_table_names_to_control_table,
    load_table_names_from_csv,
    split_table_names,
    instantiate_metadata,
    generate_table_comment_ddl,
    generate_column_comment_ddl,
    generate_pi_information_ddl,
    main
)
from src.dbxmetagen.error_handling import exponential_backoff

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up widgets

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("dest_schema", "metadata_results")
dbutils.widgets.text("table_names", "")
dbutils.widgets.text("mode", "comment")
dbutils.widgets.text("base_url", "https://adb-830292400663869.9.azuredatabricks.net")

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
dest_schema = dbutils.widgets.get("dest_schema")
table_names = split_table_names(dbutils.widgets.get("table_names"))
mode = dbutils.widgets.get("mode")
base_url = dbutils.widgets.get("base_url")

# COMMAND ----------

### Set instance variables as needed.
### Variables set in the config will be overridden by the values passed in the notebook through widgets.
METADATA_PARAMS = instantiate_metadata_objects(catalog_name, dest_schema, table_names, mode, base_url)

# COMMAND ----------

### Set key for authenticating to AI Gateway
api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"]=api_key

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now run the code - this may take some time depending on how many tables/columns you are running and what your tokens/sec throughput on the LLM is. 

# COMMAND ----------

main(METADATA_PARAMS)

# COMMAND ----------


