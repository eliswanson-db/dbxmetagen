# Databricks notebook source
config = MetadataConfig(**METADATA_PARAMS)
df = spark.read.table(f"{config.SETUP_PARAMS['catalog']}.{config.dest_schema}.metadata_control")
display(df)

# COMMAND ----------

#spark.sql(f"DROP TABLE {config.SETUP_PARAMS['catalog']}.{config.dest_schema}.metadata_control")
