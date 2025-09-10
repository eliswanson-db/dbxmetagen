# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Tables for Cost Benchmarking
# MAGIC
# MAGIC This notebook generates synthetic tables for cost benchmarking scenarios.

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

import sys

sys.path.append("../../src")

from table_generation.delta_table_generator import BenchmarkTableGenerator
from pyspark.sql import SparkSession
import time

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("schema_name", "cost_benchmarks")
dbutils.widgets.text("scenario_name", "")
dbutils.widgets.text("num_tables", "5")
dbutils.widgets.text("num_columns", "10")
dbutils.widgets.text("num_rows", "10000")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
scenario_name = dbutils.widgets.get("scenario_name")
num_tables = int(dbutils.widgets.get("num_tables"))
num_columns = int(dbutils.widgets.get("num_columns"))
num_rows = int(dbutils.widgets.get("num_rows"))

# COMMAND ----------

# Initialize generator
spark = SparkSession.builder.getOrCreate()
generator = BenchmarkTableGenerator(spark, catalog_name, schema_name)

print(f"Generating tables for scenario: {scenario_name}")
print(f"- Catalog: {catalog_name}")
print(f"- Schema: {schema_name}")
print(f"- Number of tables: {num_tables}")
print(f"- Columns per table: {num_columns}")
print(f"- Rows per table: {num_rows}")

# COMMAND ----------

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
print(f"Schema {catalog_name}.{schema_name} ready")

# COMMAND ----------

# Generate tables
start_time = time.time()

try:
    table_names = generator.create_tables_for_scenario(
        scenario_name=scenario_name,
        num_tables=num_tables,
        num_columns=num_columns,
        num_rows=num_rows,
        include_sensitive_data=True,
    )

    generation_time = time.time() - start_time

    print(
        f"Successfully generated {len(table_names)} tables in {generation_time:.2f} seconds"
    )
    print("Generated tables:")
    for table_name in table_names:
        print(f"  - {table_name}")

    # Store metadata about generated tables
    total_columns = num_tables * num_columns
    total_rows = num_tables * num_rows

    metadata_record = spark.createDataFrame(
        [
            {
                "scenario_name": scenario_name,
                "num_tables": num_tables,
                "num_columns_per_table": num_columns,
                "num_rows_per_table": num_rows,
                "total_columns": total_columns,
                "total_rows": total_rows,
                "table_names": ",".join(table_names),
                "generation_time_seconds": generation_time,
                "generation_timestamp": spark.sql(
                    "SELECT current_timestamp()"
                ).collect()[0][0],
            }
        ]
    )

    metadata_table = f"{catalog_name}.{schema_name}.table_generation_metadata"
    metadata_record.write.format("delta").mode("append").saveAsTable(metadata_table)

    print(f"Metadata saved to: {metadata_table}")

except Exception as e:
    print(f"Error generating tables: {e}")
    import traceback

    traceback.print_exc()
    dbutils.notebook.exit("ERROR: " + str(e))

# COMMAND ----------

print(f"Table generation completed successfully for scenario: {scenario_name}")
