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
    sanitize_email,
    get_control_table
)
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.deterministic_pi import ensure_spacy_model

def main(kwargs):
    spark = SparkSession.builder.getOrCreate()
    spark_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
    if not validate_csv('./metadata_overrides.csv'):
        raise Exception("Invalid metadata_overrides.csv file. Please check the format of your metadata_overrides configuration file...")

    config = MetadataConfig(**kwargs)
    if config.include_deterministic_pi and config.mode == "pi":
        ensure_spacy_model(config.spacy_model_names)
    if 'ml' not in spark_version and 'excel' in (config.ddl_output_format, config.review_output_file_type):
        raise ValueError("Excel writes in dbxmetagen are not supported on standard runtimes. Please change your output file type to tsv or sql if appropriate.")
    os.environ["DATABRICKS_HOST"]=config.base_url
    setup_ddl(config)
    create_tables(config)
    config.table_names = setup_queue(config)
    if config.control_table:
        upsert_table_names_to_control_table(config.table_names, config)    
    print("Running generate on...", config.table_names)
    generate_and_persist_metadata(config)
    spark.sql(f"""DROP TABLE IF EXISTS {config.catalog_name}.{config.schema_name}.{config.mode}_temp_metadata_generation_log_{sanitize_email(config.current_user)}""")
    control_table = get_control_table(config)
    spark.sql(f"""DROP TABLE IF EXISTS {config.catalog_name}.{config.schema_name}.{control_table}""")
    
    
