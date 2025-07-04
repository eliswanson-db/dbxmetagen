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
    sanitize_email
)
from src.dbxmetagen.config import MetadataConfig

def main(kwargs):
    spark = SparkSession.builder.getOrCreate()
    if not validate_csv('./metadata_overrides.csv'):
        raise Exception("Invalid metadata_overrides.csv file. Please check the format of your metadata_overrides configuration file...")

    config = MetadataConfig(**kwargs)
    os.environ["DATABRICKS_HOST"]=config.base_url
    setup_ddl(config)
    create_tables(config)
    queue = setup_queue(config)
    if config.control_table:
        upsert_table_names_to_control_table(queue, config)
    config.table_names = list(set(config.table_names).union(set(queue)))
    print("Running generate on...", config.table_names)
    generate_and_persist_metadata(config)
    spark.sql(f"""DROP TABLE IF EXISTS {config.catalog_name}.{config.schema_name}.{config.mode}_temp_metadata_generation_log_{sanitize_email(config.current_user)}""")
    
