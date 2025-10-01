import os
from pyspark.sql import SparkSession
from src.dbxmetagen.error_handling import validate_csv
from src.dbxmetagen.processing import (
    setup_ddl,
    create_tables,
    setup_queue,
    upsert_table_names_to_control_table,
    generate_and_persist_metadata,
    get_generated_metadata,
    get_generated_metadata_data_aware,
    get_control_table,
)
from src.dbxmetagen.user_utils import sanitize_user_identifier
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.deterministic_pi import ensure_spacy_model


def main(kwargs):
    spark = SparkSession.builder.getOrCreate()

    # Get Spark version in a serverless-compatible way
    try:
        # Try traditional approach first (works on regular clusters)
        spark_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
        print(f"Using traditional spark version: {spark_version}")
    except Exception as e:
        # Fallback for serverless compute
        print(
            f"Traditional spark version lookup failed (likely serverless): {str(e)[:100]}..."
        )

        # On serverless, Excel is not supported due to networking restrictions
        # Use version string without "ml" to trigger Excel blocking in compatibility check
        spark_version = spark.version
        print(f"Using serverless fallback: {spark_version} (Excel disabled)")
    if not validate_csv("./metadata_overrides.csv"):
        raise Exception(
            "Invalid metadata_overrides.csv file. Please check the format of your metadata_overrides configuration file..."
        )

    config = MetadataConfig(**kwargs)
    if config.include_deterministic_pi and config.mode == "pi":
        ensure_spacy_model(config.spacy_model_names)
    if "ml" not in spark_version and "excel" in (
        config.ddl_output_format,
        config.review_output_file_type,
    ):
        raise ValueError(
            "Excel writes in dbxmetagen are not supported on standard runtimes. Please change your output file type to tsv or sql if appropriate."
        )
    print("DATABRICKS_HOST", os.environ.get("DATABRICKS_HOST"))
    if not os.environ.get("DATABRICKS_HOST"):
        os.environ["DATABRICKS_HOST"] = config.base_url
    # os.environ["DATABRICKS_HOST"] = config.base_url
    setup_ddl(config)
    create_tables(config)
    config.table_names = setup_queue(config)
    if config.control_table:
        upsert_table_names_to_control_table(config.table_names, config)
    print("Running generate on...", config.table_names)
    generate_and_persist_metadata(config)
    # Get the unique temp table name for this specific job run
    temp_table = config.get_temp_metadata_log_table_name()
    control_table = get_control_table(config)
    control_table_full = f"{config.catalog_name}.{config.schema_name}.{control_table}"

    # Clean up this job's unique temp table using DROP (safe since each job has its own table)
    try:
        spark.sql(f"""DROP TABLE IF EXISTS {temp_table}""")
        print(f"Cleaned up temp table: {temp_table}")
    except Exception as e:
        print(f"Temp table cleanup failed: {e}")

    # For control table, use DELETE FROM since multiple jobs might share it (depending on config)
    try:
        spark.sql(f"""DELETE FROM {control_table_full}""")
        print(f"Cleaned up control table: {control_table_full}")
    except Exception as e:
        # If table doesn't exist, that's fine - it means cleanup already happened
        print(f"Control table cleanup skipped (table may not exist): {e}")
