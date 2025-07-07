# dbxmetagen: GenAI-Assisted Metadata Generation for Databricks

## Expanded Documentation with Full Variable Reference

This document provides a comprehensive guide to **dbxmetagen**, integrating all information from the original README and expanding it with details from the `variables.yml` configuration. Every option, workflow, and advanced usage pattern is included to ensure full transparency and control for users. The following sections are further enriched with implementation details, advanced configuration, and workflow logic based on the provided source code.

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
- [Advanced Features and Implementation Details](#advanced-features-and-implementation-details)
- [Current Status](#current-status)
- [Discussion Points & Recommendations](#discussion-points--recommendations)
- [Details of Comment Generation and PI Identification](#details-of-comment-generation-and-pi-identification)
- [Performance Details and Skew](#performance-details-and-skew)
- [Under Development](#under-development)
- [License](#license)
- [Library Analysis: Status, Open Source Nature, and Licensing](#library-analysis-status-open-source-nature-and-licensing)
- [Acknowledgements](#acknowledgements)

## Project Overview

**dbxmetagen** is a utility for generating high-quality descriptions for tables and columns in Databricks, enhancing enterprise search, governance, and Databricks Genie performance. It can identify and classify personal information (PI) into PII, PHI, and PCI. The tool is highly configurable, supporting bulk operations, SDLC integration, and fine-grained control over privacy and output formats.

## Disclaimer

- **AI-generated comments must be human-reviewed.**
- Generated comments may include data samples or metadata, depending on settings.
- Use [AI Guardrails](https://docs.databricks.com/en/ai-gateway/index.html#ai-guardrails) to mitigate risks.
- Compliance (e.g., HIPAA) is the user's responsibility.
- Unless configured otherwise, dbxmetagen inspects data and sends it to the specified model endpoint.

## Solution Overview

- **Configuration-driven:** All settings are managed via `variables.yml`.
- **Data Sampling:** Controls over sample size, inclusion of data, and metadata.
- **Validation:** Uses `Pydantic` for schema enforcement.
- **Logging:** Tracks processed tables and results.
- **DDL Generation:** Produces `ALTER TABLE` statements for integration.
- **Manual Overrides:** Supports CSV-based overrides for granular control.

## User Guide

### Personas

- **Data Engineer:** Sets up and maintains the tool.
- **Data Steward:** Reviews and approves generated metadata.
- **Data Scientist/Analyst:** Uses enriched metadata.
- **Compliance Officer:** Ensures regulatory requirements are met.

### Workflow Diagrams

- **Simple Workflow:** Clone repo, configure `variables.yml`, update `notebooks/table_names.csv`, run notebook.
- **Advanced Workflow:** Adjust PI definitions, acronyms, secondary options; use asset bundles or web terminal for deployment; leverage manual overrides.

## Minimal Setup

1. Clone the repo into Databricks (Git Folder or Workspace).
2. Update `variables.yml` (host, catalog_name, etc.).
3. Set notebook widget for comment or PI mode and run the notebook.
4. Update `notebooks/table_names.csv`.

## Full Setup Instructions

1. **Clone the repository** into Databricks or locally.
2. **Configure variables:** Adjust `variables.yml` and notebook widgets.
3. **Deploy:** Run notebook directly or use `deploy.py` for asset bundle deployment.
4. **Review and adjust settings:** Tweak options in `variables.yml` and `config.py`.
5. **Add table names** in `notebooks/table_names.csv`.
6. **Run and review outputs:** Generated DDL scripts are stored in the `generated_metadata` volume.

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
- **Export utilities** ensure directories are created as needed, and output files are validated post-write.

## Current Status

- Tested on DBR 14.3 ML LTS, 15.4 ML LTS.
- Default: Generates `ALTER TABLE` scripts, stores in a volume.
- Print-based logging for debugging and transparency.
- Control tables and logs are created and maintained for all runs.

## Discussion Points & Recommendations

- **Throttling:** Default endpoints may throttle during large or concurrent jobs.
- **Sampling:** Balance sample size for accuracy and performance.
- **Chunking:** Fewer columns per call yield richer comments but may increase cost/time.
- **Security:** Default endpoints are not HIPAA-compliant; configure secure endpoints as needed.
- **PI Mode:** Use more rows and smaller chunks for better PI detection.
- **Manual overrides** are recommended for critical or regulated columns.

## Details of Comment Generation and PI Identification

- **PHI Classification:** All medical columns are treated as PHI if `disable_medical_information_value` is true.
- **Column-level vs Table-level:** Columns are classified individually; tables inherit the highest classification from their columns.
- **Manual Overrides:** Allow explicit tagging or comment overrides via CSV.
- **Summarization:** Table comments are generated by summarizing column comments.
- **Prompt templates** enforce strict output formats and include detailed instructions to the LLM for both comment and PI modes.
- **Metadata enrichment** includes column tags, table tags, constraints, and comments, all extracted from Databricks information schema.

## Performance Details and Skew

- **Medical Columns:** Often classified as PHI for safety.
- **Chunking and Sampling:** Affect both performance and quality of generated metadata.
- **Null filtering and truncation** are implemented to improve prompt quality and reduce token waste.
- **Token length checks** are performed to avoid exceeding model limits.

## Under Development

- Prompt registration and model evaluation.
- Deterministic PI identification/classification checks.
- Support for multiple spaCy models and custom PI column field names.
- Enhanced dry-run and rollback logic.
- Robust error handling and logging improvements.

## License

This project is licensed under the Databricks DB License.

## Library Analysis: Status, Open Source Nature, and Licensing

| Library (Version)           | Status / Description                                                                 | Open Source | License Type & Details                                                                                          |
|-----------------------------|-------------------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------|
| **mlflow (2.18.0)**         | Machine learning lifecycle platform (tracking, packaging, deployment, registry)     | Yes         | **Apache 2.0** — Permissive, allows commercial use, modification, distribution, patent use.             |
| **openai (1.56.1)**         | Official Python client for OpenAI API                                               | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                      |
| **cloudpickle (3.1.0)**     | Enhanced pickling for Python objects; used for serialization                        | Yes         | **BSD 3-Clause** — Permissive, allows commercial use, modification, distribution.                         |
| **pydantic (2.9.2)**        | Data validation and settings management using Python type annotations                | Yes         | **BSD 3-Clause** — Permissive, allows commercial use, modification, distribution.                         |
| **ydata-profiling (4.12.1)**| Automated data profiling and exploratory data analysis                              | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                      |
| **databricks-langchain (0.0.3)** | Integration between Databricks and LangChain for LLM apps                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                     |
| **openpyxl (3.1.5)**        | Read/write Excel 2010 xlsx/xlsm/xltx/xltm files                                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                     |
| **spacy (3.8.7)**           | Industrial-strength NLP toolkit                                                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                     |
| **presidio_analyzer (2.2.358)** | PII/PHI/PCI detection and analysis toolkit                                     | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                         |
| **presidio_anonymizer (2.2.358)** | PII/PHI/PCI anonymization toolkit                                           | Yes         | **MIT** — Permissive, allows commercial use, modification, distribution, private use.                         |

## Acknowledgements

Special thanks to the Databricks community and contributors.

**For full technical details, advanced workflows, and troubleshooting, refer to the documentation and comments in your repository and `variables.yml`. Every configuration option is surfaced here to enable maximum control and compliance for your enterprise metadata workflows.**

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/40121587/3fc6db76-7136-42e9-90bb-a37a499116b2/paste.txt
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/40121587/649a0a74-c562-4629-9ec6-77a4905dca9d/paste-2.txt