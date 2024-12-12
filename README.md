# DBX Metadata Generation

# GenAI-Assisted Metadata Generation (a.k.a `dbxmetagen`)

# `dbxmetagen` Overview
### This is a utility to help generate high quality descriptions for tables and columns to enhance enterprise search and data governance, improve Databricks Genie performance for Text-2-SQL, and generally help curate a high quality metadata layer for enterprise data.

While Databricks does offer [AI Generated Documentation](https://docs.databricks.com/en/comments/ai-comments.html), this is not sustainable at scale as a human must manually select and approve AI generated metadata. This utility, `dbxmetagen`, helps generate table and column descriptions at scale. 

### Disclaimer
AI generated comments are not always accurate and comment DDLs should be reviewed prior to modifying your tables. Databricks strongly recommends human review of AI-generated comments to check for inaccuracies. While the model has been guided to avoids generating harmful or inappropriate descriptions, you can mitigate this risk by setting up [AI Guardrails](https://docs.databricks.com/en/ai-gateway/index.html#ai-guardrails) in the AI Gateway where you connect your LLM. 

### Solution Overview:
There are a few key sections in this notebook: 
- Library installs and setup using the config referenced in `dbxmetagen/src/config.py`
- Function definitions for:
  - Retrieving table and column information from the list of tables provided in `table_names.csv`
  - Sampling data from those tables, with exponential backoff, to help generate more accurate metadata, especially for columns with categorical data, that will also indicate the structure of the data. This is particularly helpful for [Genie](https://www.databricks.com/product/ai-bi/genie). This sampling also checks for nulls.
  - Use of `Pydantic` to ensure that LLM metadata generation conforms to a particular format. This is also used for DDL generation to ensure that the DDL is always runnable. 
  - Creation of a log table keeping track of tables read/modified
  - Creation of DDL scripts, one for each table, that have the DDL commands to `ALTER TABLE` to add comments to table and columns. This is to help integrate with your CI/CD processes, in case you do not have access in a production environment
- Application of the functions above to generate metadata and DDL for the list of tables provided in `dbxmetagen/table_names.csv` 

### Setup
1. Clone the Repo into Databricks or locally
1. If cloned into Repos in Databricks, can run the notebook using an all-purpose cluster without further deployment.
   1. Alternatively, run the notebook deploy.py, open the web terminal, copy-paste the path and command from deploy.py and run it in the web terminal. This will run an asset bundle-based deploy in the Databricks UI web terminal.
1. If cloned locally, recommend using asset bundle build to create and run a workflow.
1. Either create a catalog or use an existing one.
1. Set the config.py file in src/dbxmetagen to whatever settings you need. If you want to make changes to variables in your project, change them in the notebook widget.
   1. Make sure to check the options for add_metadata and apply_ddl and set them correctly. Add metadata will run a describe extended on every column and use the metadata in table descriptions, though ANALYZE ... COLUMNS will need to have been run to get useful information from this.
   2. You also can adjust sample_size, columns_per_call, and ACRO_CONTENT.
1. In notebooks/table_names.csv, keep the first row as _table_name_ and add the list of tables you want metadata to be generated for. Add them as <schema>.<table> if they are in the same catalog that you define your catalog in the config.py file separately, or you can use a three-level namespace for these table names.

### Current status
1. Tested on DBR 15.4ML LTS
1. Currently creates ALTER scripts and puts in a volume. Tested in a databricks workspace.
1. Some print-based logging to make understanding what's happening and debugging easy in the UI

### Discussion points:
1. Throttling - the PPT endpoints will throttle eventually. Likely this will occur wehn running backfills.
1. Sampling - setting a reasonable sample size for data will serve to provide input from column contents without leading to swamping of column names.
1. Chunking - running a smaller number of columns at once will result in more attention paid and more tokens PER column but will probably cost slightly more and take longer.
1. One of the easiest ways to speed this up and get terser answers is to ramp up the columns per call - compare 5 and 50 for example.
1. Larger chunks will result in simpler comments with less creativity and elaboration.

### Future Items
1. Fix sampling - Done
1. Add flag for inclusion of metadata - Done
1. Add flag for direct commenting of tables - Done
1. Add flag for dry run of direct commenting - Done
1. Summarizer step for table comments - Done
1. Adjust prompts and few-shot examples to reduce errors and improve comments - Done
1. Add a retry for get_response with a double injection reminder to only respond with the provided schema.
1. Register as a UC model to allow tracking and iteration of prompts
1. Expand detail in audit logs
1. register a pyfunc so that we can iterate on prompts
1. Separate out table comments into its own Response
1. Add async
1. any utility of agent framework?
1. Fix input data classes
