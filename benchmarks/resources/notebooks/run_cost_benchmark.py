# Databricks notebook source
# MAGIC %md
# MAGIC # Run Cost Benchmark
# MAGIC
# MAGIC This notebook runs dbxmetagen on cost benchmark scenarios and measures performance.

# COMMAND ----------

import sys

sys.path.append("../../src")
sys.path.append("../../../")

from evaluation.cost_evaluator import CostEvaluator
from src.dbxmetagen.main import main
from src.dbxmetagen.config import MetadataConfig
import time
import json

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("schema_name", "cost_benchmarks")
dbutils.widgets.text("scenario_name", "")
dbutils.widgets.text("num_workers", "2")
dbutils.widgets.text("mode", "pi")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
scenario_name = dbutils.widgets.get("scenario_name")
num_workers = int(dbutils.widgets.get("num_workers"))
mode = dbutils.widgets.get("mode")

# COMMAND ----------

# Initialize cost evaluator
spark = SparkSession.builder.getOrCreate()
evaluator = CostEvaluator(spark, catalog_name, "benchmarks")

print(f"Running cost benchmark:")
print(f"- Scenario: {scenario_name}")
print(f"- Catalog: {catalog_name}")
print(f"- Schema: {schema_name}")
print(f"- Workers: {num_workers}")
print(f"- Mode: {mode}")

# COMMAND ----------

# Get cluster configuration
cluster_info = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "unknown")
node_type = spark.conf.get("spark.databricks.clusterUsageTags.nodeTypeId", "unknown")

cluster_config = {
    "cluster_id": cluster_info,
    "num_workers": num_workers,
    "node_type": node_type,
    "mode": mode,
    "scenario": scenario_name,
}

print(f"Cluster configuration: {json.dumps(cluster_config, indent=2)}")

# COMMAND ----------

# Get table metadata for this scenario
metadata_query = f"""
SELECT * FROM {catalog_name}.{schema_name}.table_generation_metadata 
WHERE scenario_name = '{scenario_name}'
ORDER BY generation_timestamp DESC
LIMIT 1
"""

try:
    metadata_df = spark.sql(metadata_query)
    if metadata_df.count() == 0:
        raise Exception(f"No metadata found for scenario: {scenario_name}")

    metadata_row = metadata_df.collect()[0]
    table_metadata = {
        "num_tables": metadata_row["num_tables"],
        "num_columns_per_table": metadata_row["num_columns_per_table"],
        "num_rows_per_table": metadata_row["num_rows_per_table"],
        "total_columns": metadata_row["total_columns"],
        "total_rows": metadata_row["total_rows"],
        "table_names": metadata_row["table_names"],
    }

    # Add to cluster config for comprehensive tracking
    cluster_config.update(table_metadata)

    print(f"Table metadata: {json.dumps(table_metadata, indent=2)}")

except Exception as e:
    print(f"Error getting table metadata: {e}")
    dbutils.notebook.exit("ERROR: " + str(e))

# COMMAND ----------

# Start benchmark recording
run_id = evaluator.record_benchmark_start(
    scenario_name=scenario_name,
    cluster_config=cluster_config,
    table_metadata=table_metadata,
)

print(f"Started benchmark run: {run_id}")

# COMMAND ----------

# Run dbxmetagen
start_time = time.time()

try:
    # Get current user
    current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]

    # Prepare dbxmetagen configuration
    kwargs = {
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "mode": mode,
        "table_names": table_metadata["table_names"],
        "current_user": current_user,
        "apply_ddl": False,
        "reviewable_output_format": "tsv",
        "env": "benchmark",
    }

    print("Starting dbxmetagen execution...")
    print(f"Processing tables: {table_metadata['table_names']}")

    # Run dbxmetagen
    main(kwargs)

    execution_time = time.time() - start_time

    print(f"dbxmetagen completed in {execution_time:.2f} seconds")

    # Calculate processing metrics
    processing_metrics = evaluator.calculate_throughput_metrics(
        total_tables=table_metadata["num_tables"],
        total_columns=table_metadata["total_columns"],
        total_rows=table_metadata["total_rows"],
        execution_time_seconds=execution_time,
    )

    print("Processing metrics:")
    for key, value in processing_metrics.items():
        print(f"  {key}: {value}")

    # Estimate cost
    cost_estimate_data = evaluator.estimate_cost(cluster_config, execution_time)

    print("Cost estimates:")
    for key, value in cost_estimate_data.items():
        print(f"  {key}: {value}")

    # Record completion
    evaluator.record_benchmark_completion(
        run_id=run_id,
        execution_time_seconds=execution_time,
        processing_metrics=processing_metrics,
        cost_estimate=cost_estimate_data.get("estimated_cost_usd", 0),
    )

    print(f"Benchmark run {run_id} completed successfully!")

except Exception as e:
    print(f"Error running dbxmetagen: {e}")
    import traceback

    traceback.print_exc()

    # Record failure
    evaluator.record_benchmark_completion(
        run_id=run_id,
        execution_time_seconds=time.time() - start_time,
        processing_metrics={"error": str(e)},
        cost_estimate=0,
    )

    dbutils.notebook.exit("ERROR: " + str(e))

# COMMAND ----------

print(f"Cost benchmark run completed successfully for scenario: {scenario_name}")
print(f"Run ID: {run_id}")
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Estimated cost: ${cost_estimate_data.get('estimated_cost_usd', 0):.4f}")
