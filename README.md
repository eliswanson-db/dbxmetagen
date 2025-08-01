# dbxmetagen: GenAI-Assisted Metadata Generation for Databricks

## Expanded Documentation with Full Variable Reference

This document provides a comprehensive guide to **dbxmetagen**, integrating all information from the original README and expanding it with details from the `variables.yml` configuration. Every option, workflow, and advanced usage pattern is included to ensure full transparency and control for users.

## Table of Contents

- [Project Overview](#project-overview)
- [Disclaimer](#disclaimer)
- [Solution Overview](#solution-overview)
- [User Guide](#user-guide)
  - [Personas](#personas)
  - [Workflow Diagrams](#workflow-diagrams)
- [Minimal Setup](#minimal-setup)
- [Full Setup Instructions](#full-setup-instructions)
- [Configuration Reference](#configuration-reference)
- [Workflow and Usage Patterns](#workflow-and-usage-patterns)
- [Current Status](#current-status)
- [Discussion Points & Recommendations](#discussion-points--recommendations)
- [Details of Comment Generation and PI Identification](#details-of-comment-generation-and-pi-identification)
- [Performance Details and Skew](#performance-details-and-skew)
- [Under Development](#under-development)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

**dbxmetagen** is a utility for generating high-quality descriptions for tables and columns in Databricks, enhancing enterprise search, governance, and Databricks Genie performance. It can identify and classify personal information (PI) into PII, PHI, and PCI. The tool is highly configurable, supporting bulk operations, SDLC integration, and fine-grained control over privacy and output formats.

## Disclaimer

- **AI-generated comments must be human-reviewed.**
- Generated comments may include data samples or metadata, depending on settings.
- Use [AI Guardrails](https://docs.databricks.com/en/ai-gateway/index.html#ai-guardrails) to mitigate risks.
- Compliance (e.g., HIPAA) is the user's responsibility.
- Unless configured otherwise, dbxmetagen inspects data and sends it to the specified model endpoint. There are a wide variety of options to control this behavior in detail.

## Solution Overview

- **Configuration-driven:** All settings are managed via `variables.yml`.
- **AI-assisted:** Both comment generation and PI identification and classification use both AI-based and deterministic or data engineering approaches, combining multiple sophisticated approaches to get you quality metadata.
- **Data Sampling:** Controls over sample size, inclusion of data, and metadata.
- **Validation:** Uses `Pydantic` and structured outputs for schema enforcement.
- **Logging:** Tracks processed tables and results.
- **DDL Generation:** Produces `ALTER TABLE` statements for integration.
- **Manual Overrides:** Supports CSV-based overrides for granular control.

## User Guide

### Entry points
Both primary entry points for this application are Databricks notebooks.

- **dbxmetagen/src/notebooks/generate_metadata** This is the primary entry point for the application, allowing both comment generation and PI identification and classification.
- **dbxmetagen/src/notebooks/sync_reviewed_ddl** This utility allows re-integration of reviewed and edited run logs in tsv or excel format to be used to apply DDL to tables.

### Personas

- **Data Engineer:** Sets up and maintains the tool.
- **Data Steward:** Reviews and approves generated metadata.
- **Data Scientist/Analyst:** Uses enriched metadata.
- **Compliance Officer:** Ensures regulatory requirements are met.

### Workflow Diagrams

- **Simple Workflow:** Clone repo, configure `variables.yml`, update `notebooks/table_names.csv`, run notebook.

<img src="images/personas.png" alt="User Personas" width="400" top-margin="50">

- **Advanced Workflow:** Adjust PI definitions, acronyms, secondary options; use asset bundles or web terminal for deployment; leverage manual overrides.

<br/>
<br/>

<img src="images/workflow1.png" alt="User Process Flow - simple" width="1000"> 

<br/>
<br/>

<img src="images/simple_dab_workflow.png" alt="User Process Flow - simple" width="1200"> 

<br/>
<br/>

## Minimal Setup

1. Clone the repo into Databricks (Git Folder or Workspace).
2. Update `variables.yml` (host, catalog_name, etc.).
3. Update `notebooks/table_names.csv`.
4. Set notebook widget for comment or PI mode and run the notebook.


## Full Setup Instructions

1. **Clone the repository** into Databricks or locally.
2. **Configure variables:** Adjust `variables.yml` and notebook widgets.
3. **Deploy:** Run notebook directly or use `deploy.py` for asset bundle deployment.
4. **Review and adjust settings:** Tweak options in `variables.yml` and `config.py`.
5. **Add table names** in `notebooks/table_names.csv`.
6. **Run and review outputs:** Generated DDL scripts are stored in the `generated_metadata` volume.

## Details

1. Some errors are silent by design, but they always show up in the log table, so review it if you are not seeing the outputs you expect.
1. If you get an error partway through a run, the control table will still keep in memory the tables you entered the first time that haven't run yet, so you should be able to remove the table names from table_names.csv and run it again and it should pick up any unfinished tables. If you don't, you'll see that they all get run again anyways. This checkpointing is a feature.
1. To make sure that column constraints are interpretable, make sure the constraint names relate to the column that they constrain, as constraints are pulled from the table metadata, not the column metadata, and they describe the constraint name, not the columns with the constraint.

## Configuration Reference

Below is a table summarizing all configuration variables, their descriptions, and defaults. Each variable can be set in `variables.yml` to control workflow, privacy, output, and model behavior.

| Variable | Description | Default |
|----------|-------------|---------|
| catalog_name | Target catalog for data, models, and files. If source tables are here, only schema.table needed; otherwise, fully scoped names required. | dbxmetagen |
| host | Base URL host. Overridden by asset bundles if used. | https://adb-830292400663869.9.azuredatabricks.net/ |
| allow_data | If false, no data sent to LLMs, no data in comments, and no data-based metadata. Reduces output quality. Also sets sample_size=0, allow_data_in_comments=false, include_possible_data_fields_in_metadata=false. | false |
| sample_size | Number of rows to sample per chunk for prompt generation. 0 disables data sampling. | 10 |
| disable_medical_information_value | If true, all medical info is treated as PHI for maximum security. | true |
| allow_data_in_comments | If true, allows data to appear in comments. Does not prevent data from being used as input. | true |
| add_metadata | If true, uses metadata from information schema and DESCRIBE EXTENDED ... COMPUTE STATISTICS. May slow process and can leak data via min/max. | true |
| include_datatype_from_metadata | If true, includes datatype in comment generation. | false |
| include_possible_data_fields_in_metadata | If true, includes fields from extended metadata that may leak PII/PHI. Useful for Genie. | true |
| catalog_tokenizable | Tokenizable catalog name; supports string formatting for environment. | __CATALOG_NAME__ |
| format_catalog | If true, formats bracketed variables in catalog name; otherwise, keeps as literal. | false |
| model | LLM endpoint for model calls. Recommend databricks-claude-3-7-sonnet or databricks-meta-llama-3-3-70b-instruct. | databricks-meta-llama-3-3-70b-instruct |
| job_table_names | Default table names if no host overrides. | default.simple_test |
| apply_ddl | If true, applies DDL directly to environment (alters tables). | false |
| ddl_output_format | Output format for DDL: SQL file (default) or TSV. | excel |
| reviewable_output_format | Format for full run output for reviewability. | excel |
| include_deterministic_pi | If true, runs presidio analyzer before LLM-based PI identification. | true |
| spacy_model_names | spaCy models to use in presidio. Only single model supported currently. | en_core_web_lg |
| pi_classification_rules | Rules for PI classification, injected into prompts. See below for full default. | See below |
| allow_manual_override | If true, allows manual overrides via CSV. | true |
| override_csv_path | Path for manual override CSV. | metadata_overrides.csv |
| tag_none_fields | If true, tags fields with no PI; otherwise, leaves untagged. | true |
| max_prompt_length | Maximum prompt length for LLM. | 4096 |
| word_limit_per_cell | Maximum number of words per cell from source tables (truncates longer values). | 100 |
| limit_prompt_based_on_cell_len | If true, truncates cells longer than word_limit_per_cell. | true |
| columns_per_call | Number of columns sent to LLM per chunk. | 5 |
| max_tokens | Maximum tokens for model output. | 4096 |
| temperature | Temperature parameter for LLM. | 0.1 |
| acro_content | Acronyms used in data, provided as a dictionary. | {"DBX":"Databricks","WHO":"World Health Organization","GMC":"Global Marketing Code"} |
| schema_name | Primary schema for outputs. Not the source schema. | metadata_results |
| volume_name | Volume for storing DDL files. | generated_metadata |
| registered_model_name | Registered model name. | default |
| model_type | Model type. | default |
| table_names_source | Path variable to table names. | csv_file_path |
| source_file_path | Path to source file for table names. | table_names.csv |
| current_user | User deploying the bundle. | ${workspace.current_user.userName} |
| current_working_directory | Working directory or bundle root. | /Users/${var.current_user}/.bundle/${bundle.name}/${bundle.target} |
| control_table | Control table name. | metadata_control_{} |
| dry_run | If true, performs a dry run (not yet implemented). Setting volume_name to null and apply_ddl to false is an effective dry run. | false |
| pi_column_field_names | Field names for PI columns (not yet implemented). | default |
| review_input_file_type | Input file type for reviewed DDL (excel or tsv). | tsv |
| review_output_file_type | Output file type for reviewed DDL (sql, excel, or tsv). | excel |
| review_apply_ddl | If true, applies all DDL after running 'sync_reviewed_ddl'. | false |

### Default PI Classification Rules

```
PII (Personally Identifiable Information): Any data that can identify an individual, such as name, address, or social security number. PHI (Protected Health Information): Health-related data that includes any of the 18 HIPAA-identified elements, like medical records or treatment histories. PCI (Payment Card Information): Data related to payment card transactions, including card numbers, expiration dates, and security codes. Non-PII/PHI/PCI: Data that cannot be used to identify an individual or is not related to health or payment cards, such as general demographics or anonymized statistics.
###
To identify these types:
###
PII: Look for column names or data containing personal details like 'full_name', 'address', 'ssn', 'email', or 'phone_number'.
PHI: Search for health-related terms in column names or data, such as 'medical_record_number', 'diagnosis', 'treatment_date', or any of the 18 HIPAA identifiers.
PCI: Identify columns or data with payment card-related terms like 'card_number', 'expiration_date', 'cvv', or 'cardholder_name'.
Non-PII/PHI/PCI: Look for general, non-identifying information like 'zip_code' (without other identifiers), 'age_group', or 'product_category'.
###
Tables with PII vs columns with PII:
###
If a column has only PII, then it should be considered only PII, even if other columns in the table have medical information in them. A specific example of this is that if a column only contains name, or address, then it should be marked as pii, not as phi. However, the column containing medical information would be considered PHI because it pertains to the health status, provision of health care, or payment for health care that can be linked to an individual.

If a table has columns with both PII and PHI, or with PII and medical information, then the table itself has PHI.
```


## Workflow and Usage Patterns

### Privacy and Security Controls

- **allow_data, allow_data_in_comments, sample_size, include_possible_data_fields_in_metadata:** Together, these control whether data is sent to LLMs, appears in comments, or is used in metadata. Setting `allow_data` to false disables all data-related features for maximum privacy.
- **disable_medical_information_value:** Converts all medical info to PHI for stricter compliance.
- **add_metadata, include_datatype_from_metadata:** Control how much metadata is used, balancing richness against privacy.
- **tag_none_fields:** Controls whether columns classified as 'None' for PI are explicitly tagged.

### Output and Review

- **ddl_output_format, reviewable_output_format, review_input_file_type, review_output_file_type:** Control output formats for DDL and review files, supporting SQL, TSV, and Excel.
- **volume_name:** Specifies where generated files are stored.
- **apply_ddl, review_apply_ddl:** Control whether DDL is applied directly or only generated for review.

Each run will export an 'exportable run log' to a folder named the same. This run log can be TSV or Excel, and will include every table that was run during that run. 
Exportable run logs can be modified and input to the review process. Do this by checking the variables 'review_input_file_type', 'review_output_file_type', 'review_apply_ddl', and 'column_with_reviewed_ddl' and setting them to desired values. Note that 'review_apply_ddl' will apply ddl to tables. 

Put the file you want to review in 'reviewed_outputs' folder in your user folder and put the filename in the sync_reviewed_ddl notebook widget, and run the notebook. A file will be updated and output.

### Model and Prompt Tuning

- **model, temperature, max_prompt_length, max_tokens, columns_per_call:** Tune LLM endpoint and prompt parameters for desired output quality and performance.
- **acro_content:** Provide domain-specific acronyms for better metadata generation.
- **include_deterministic_pi, spacy_model_names:** Integrate deterministic PI detection (e.g., Presidio) before LLM-based classification.

### Manual Overrides and Advanced Control

- **allow_manual_override, override_csv_path:** Allow manual specification of metadata via CSV, overriding automated results.
- **control_table, dry_run:** Support for checkpointing, re-running incomplete tables, and dry-run testing.
- **pi_column_field_names:** (Planned) Custom field names for PI columns.

### Environment and Deployment

- **catalog_name, schema_name, host, catalog_tokenizable, format_catalog:** Control where data and metadata are stored and how environment variables are handled.
- **job_table_names, table_names_source, source_file_path:** Control sources for table lists.
- **current_user, current_working_directory:** Support for multi-user and multi-environment deployments.

## Advanced Features and Implementation Details

### 1. Data Processing and Sampling

- **DataFrames are chunked** according to `columns_per_call` for scalable prompt generation.
- **Sampling logic** ensures that only a representative, non-null-heavy subset of data is sent to the LLM, controlled by `sample_size` and filtered for nulls.
- **Truncation of data cells** is enforced by `word_limit_per_cell` and `limit_prompt_based_on_cell_len`, ensuring prompts stay within token limits and privacy constraints.

### 2. Metadata Extraction and Prompt Generation

- **Extended metadata** (from `DESCRIBE EXTENDED`) is filtered and included/excluded in prompts based on `add_metadata`, `include_datatype_from_metadata`, and `include_possible_data_fields_in_metadata`.
- **Prompt construction** is performed by specialized classes (e.g., `CommentPrompt`, `PIPrompt`, `CommentNoDataPrompt`), which format input for the LLM and enforce output structure.
- **Acronym expansion** is supported via the `acro_content` dictionary, unpacking abbreviations in generated comments.

### 3. PI Detection and Classification

- **Deterministic PI detection** (via Presidio) can be run before LLM-based identification (`include_deterministic_pi`), and results are incorporated into prompt context.
- **Classification logic** is enforced both at the column and table level, with rules for PHI/PII/PCI inheritance and prioritization.
- **Manual PI tagging** is possible through CSV overrides, with `allow_manual_override` and `override_csv_path`.

### 4. DDL and Output Generation

- **DDL statements** for comments and PI tagging are generated as SQL, TSV, or Excel files, with output paths constructed dynamically based on user, date, and run context.
- **Direct DDL application** is controlled by `apply_ddl`; otherwise, DDL is written to files for review and manual application.
- **Logging** is performed at each step, with logs written to Delta tables for auditability.

### 5. Table and Control Flow Management

- **Control tables** are used to track processing status, support checkpointing, and enable resuming incomplete runs.
- **Table name scoping** ensures all tables are referenced with full catalog.schema.table names, with utility functions to sanitize and expand names as needed.
- **Queue management** allows for batch or incremental processing of table lists, merging sources from control tables, config, and CSV files.

### 6. Output Review and Export

- **Reviewable outputs** are exported in the desired format (Excel/TSV/SQL), and can be re-imported for DDL application if `review_apply_ddl` is enabled. 
- **Syncing reviewed outputs** Put reviewed 'exportable_run_log' into the 'reviewed_outputs' folder, update any review variables such as input and output format in variables.yml, add the file name to the file_name widget in the sync notebook and run the notebook. 
- **Export utilities** ensure directories are created as needed, and output files are validated post-write. Note that if DDL is generated on one DBR, such as 16.4, you may be unable to apply it with other DBRs, such as 14.3. This is a common source of user error.
- **Modify** Modify the 'column_content' column for comments, or the 'type' and 'classification' columns for PI classification. Set your options for review, including review_input_file_type, review_output_file_type, and review_apply_ddl.

## Current Status

- Tested on DBR 16.4 LTS, 14.3 LTS, and 15.4 LTS, as well as the ML versions.
- Views only work on 16.4. Pre-16.4, alternative DDL is used that only works on tables.
- Excel writes for metadata generator or sync_reviewed_ddl only work on ML runtimes. If you must use a standard runtime, leverage tsv.

## Discussion Points & Recommendations

- **Throttling:** Default endpoints may throttle during large or concurrent jobs.
- **Sampling:** Balance sample size for accuracy and performance.
- **Chunking:** Fewer columns per call yield richer comments but may increase cost/time.
- **Security:** Default endpoints are not HIPAA-compliant; configure secure endpoints as needed.
- **PI Mode:** Use more rows and smaller chunks for better PI detection.

## Details of Comment Generation and PI Identification

- **PHI Classification:** All medical columns are treated as PHI if `disable_medical_information_value` is true.
- **Column-level vs Table-level:** Columns are classified individually; tables inherit the highest classification from their columns.
- **Manual Overrides:** Allow explicit tagging or comment overrides via CSV.
- **Summarization:** Table comments are generated by summarizing column comments.

## Performance Details and Skew

- **Medical Columns:** Often classified as PHI for safety.
- **Chunking and Sampling:** Affect both performance and quality of generated metadata.

## Under Development

- Prompt registration and model evaluation using agent evals.
- Support for multiple spaCy models and custom PI column field names.

## License

This project is licensed under the Databricks DB License.

## Analysis of packages used

Below is a detailed table summarizing the status, open source availability, and licensing details for each Python library used in this project:

| Library (Version)           | Status / Description                                                                 | Open Source | License Type & Details                                                                                          |
|-----------------------------|-------------------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------|
| **mlflow (2.18.0)**         | Machine learning lifecycle platform (tracking, packaging, deployment, registry)     | Yes         | **Apache 2.0** — Permissive, allows commercial use, modification, distribution, patent use[1][2].             |
| **openai (1.56.1)**         | Official Python client for OpenAI API                                               | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use[3].                      |
| **cloudpickle (3.1.0)**     | Enhanced pickling for Python objects; used for serialization                        | Yes         | **BSD 3-Clause** — Permissive, allows commercial use, modification, distribution[4].                         |
| **pydantic (2.9.2)**        | Data validation and settings management using Python type annotations                | Yes         | **BSD 3-Clause** — Permissive, allows commercial use, modification, distribution[5].                         |
| **ydata-profiling (4.12.1)**| Automated data profiling and exploratory data analysis                              | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use[6].                      |
| **databricks-langchain (0.0.3)** | Integration between Databricks and LangChain for LLM apps                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use[7].                     |
| **openpyxl (3.1.5)**        | Read/write Excel 2010 xlsx/xlsm/xltx/xltm files                                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use[8].                     |
| **spacy (3.8.7)**           | Industrial-strength NLP toolkit                                                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use[9].                     |
| **presidio_analyzer (2.2.358)** | PII/PHI/PCI detection and analysis toolkit                                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                         |
| **presidio_anonymizer (2.2.358)** | PII/PHI/PCI anonymization toolkit                                           | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                         |

### Additional Details

- **All libraries listed are open source** and widely used in the Python/data science ecosystem.
- **Licensing is permissive** for all entries, with Apache 2.0, BSD 3-Clause, and MIT being the most common. These allow for commercial use, modification, and redistribution, with minimal restrictions.
- **MLflow** is governed by Apache 2.0, which is slightly more restrictive than MIT but still highly permissive[1][2].
- **OpenAI's Python client** and most data science/utility libraries here use the MIT license, which is very permissive[3][6][7][8][9].
- **Pydantic** and **cloudpickle** use BSD 3-Clause, which is similar to MIT but with an additional clause regarding the use of the names of the copyright holders[4][5].

### Notes

- All versions referenced are current or recent as of July 2025.
- For some libraries (e.g., presidio_analyzer and presidio_anonymizer), the MIT license is standard for Microsoft open source projects, though always confirm in the project repository for the specific release.

This table should provide a clear overview for compliance, integration, and open source policy review.

[1] https://github.com/mlflow/mlflow/blob/master/LICENSE.txt

[2] https://pypi.org/project/mlflow/

[3] https://github.com/openai/openai-quickstart-python/blob/master/LICENSE

[4] https://release-monitoring.org/project/python-cloudpickle

[5] https://github.com/conda-forge/pydantic-feedstock/blob/main/LICENSE.txt

[6] https://github.com/ydataai/ydata-profiling/blob/develop/LICENSE

[7] https://pypi.org/project/databricks-langchain/

[8] https://layers.openembedded.org/layerindex/recipe/120206/

[9] https://github.com/ipipan/spacy-pl/blob/master/LICENSE

[10] http://mlflow.org/releases/2.18.0

[11] https://security.snyk.io/package/pip/mlflow/2.18.0

[12] http://mlflow.org

[13] https://launchpad.net/ubuntu/+source/cloudpickle/3.1.0-1

[14] https://github.com/FullStackWithLawrence/pydantic-example/blob/main/LICENSE

[15] https://github.com/mlflow/mlflow/blob/master/CHANGELOG.md

[16] https://github.com/openai/openai-python/blob/main/LICENSE

[17] https://github.com/ydataai/ydata-profiling/releases

[18] https://hiplab.mc.vanderbilt.edu/git/steve/databricks-cli/src/commit/e236c8a54546a08e5415fcf83ab4d200769dde1d/LICENSE

[19] https://github.com/LaneDoyle/OpenPyxl-Examples/blob/master/LICENSE

[20] https://huggingface.co/spacy/pl_core_news_lg/blob/main/LICENSES_SOURCES

## Acknowledgements

Special thanks to the Databricks community and contributors.

**For full technical details, advanced workflows, and troubleshooting, refer to the documentation and comments in your repository and `variables.yml`. Every configuration option is surfaced here to enable maximum control and compliance for your enterprise metadata workflows.**
