# DBX Metadata Generation

### Setup
1. Clone the Repo into Databricks or locally
1. If cloned into Repos in Databricks, can run the notebook using an all-purpose cluster without further deployment.
1. If cloned locally, recommend using asset bundle build to create and run a workflow.
1. Either create a catalog and schema, or use an existing one.
1. Set the config.py file in dbxmetagen/src to whatever settings you need.

### Current status
1. Tested on DBR 15.4ML LTS
1. Currently creates ALTER scripts and puts in a volume. Tested in a databricks workspace.
1. Some print-based logging to make understanding what's happening and debugging easy in the UI

### Discussion points:
1. Throttling - the PPT endpoints will throttle eventually. Likely this will occur wehn running backfills.
1. Sampling - setting a reasonable sample size for data will serve to provide input from column contents without leading to swamping of column names.
1. Chunking - running a smaller number of columns at once will result in more attention paid and more tokens PER column but will probably cost slightly more and take longer.
1. One of the easiest ways to speed this up and get terser answers is to ramp up the columns per call - compare 5 and 50 for example.

### Future Items
1. Adjust prompts and few-shot examples to reduce errors
1. Add a retry for get_response with a double injection reminder to only respond with the provided schema.
1. Register as a UC model to allow tracking and iteration of prompts
1. Expand detail in audit logs
1. Change table comment generation to use the table name and be longer
