# Databricks notebook source
# MAGIC %md
# MAGIC # Performance Evaluation for dbxmetagen
# MAGIC
# MAGIC This notebook evaluates the performance of dbxmetagen against benchmark data.

# COMMAND ----------

# MAGIC %pip install scikit-learn pandas

# COMMAND ----------

import sys

sys.path.append("../../src")

from evaluation.performance_evaluator import PerformanceEvaluator
from pyspark.sql import SparkSession
import os
import glob

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("schema_name", "benchmarks")
dbutils.widgets.text("benchmark_table", "")
dbutils.widgets.text("results_table", "")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
benchmark_table = dbutils.widgets.get("benchmark_table")
results_table = dbutils.widgets.get("results_table")

# COMMAND ----------

# Initialize evaluator
spark = SparkSession.builder.getOrCreate()
evaluator = PerformanceEvaluator(spark)

print(f"Evaluating performance:")
print(f"- Catalog: {catalog_name}")
print(f"- Schema: {schema_name}")
print(f"- Benchmark table: {benchmark_table}")
print(f"- Results table: {results_table}")

# COMMAND ----------

# Find the most recent dbxmetagen output file
output_base_path = f"/Volumes/{catalog_name}/performance_benchmarks/generated_metadata"

# Get current user to find their output folder
current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]
user_email = current_user.replace("@", "_at_").replace(".", "_")

print(f"Looking for output files for user: {current_user} (sanitized: {user_email})")

# Find the most recent output file
search_pattern = (
    f"{output_base_path}/{user_email}/*/exportable_run_logs/review_metadata_pi_*.tsv"
)
print(f"Searching for files matching: {search_pattern}")

# Use dbutils to list files since we're in Databricks
try:
    # Get all possible date folders
    user_folders = dbutils.fs.ls(
        f"/Volumes/{catalog_name}/performance_benchmarks/generated_metadata/{user_email}/"
    )

    latest_file = None
    latest_timestamp = None

    for folder in user_folders:
        if folder.isDir():
            logs_path = folder.path + "exportable_run_logs/"
            try:
                files = dbutils.fs.ls(logs_path)
                for file in files:
                    if file.name.startswith(
                        "review_metadata_pi_"
                    ) and file.name.endswith(".tsv"):
                        # Extract timestamp from filename
                        timestamp_str = file.name.replace(
                            "review_metadata_pi_", ""
                        ).replace(".tsv", "")
                        if latest_timestamp is None or timestamp_str > latest_timestamp:
                            latest_timestamp = timestamp_str
                            latest_file = file.path
            except:
                continue

    if latest_file:
        print(f"Found latest output file: {latest_file}")
    else:
        raise Exception("No dbxmetagen output files found")

except Exception as e:
    print(f"Error finding output files: {e}")
    # Fallback - try to find any TSV file
    latest_file = None

# COMMAND ----------

if latest_file is None:
    print("Could not find output file automatically. Please check the path manually.")
    dbutils.notebook.exit("ERROR: No output file found")

# Convert Databricks path to local path for pandas
local_file_path = latest_file.replace("dbfs:", "")

print(f"Using output file: {local_file_path}")

# COMMAND ----------

# Run performance evaluation
try:
    report = evaluator.generate_performance_report(
        dbxmetagen_output_path=local_file_path,
        benchmark_table=benchmark_table,
        report_output_table=results_table,
    )

    if "error" in report:
        print(f"Evaluation failed: {report['error']}")
        dbutils.notebook.exit("ERROR: " + report["error"])
    else:
        print("Performance evaluation completed successfully!")
        print(f"Total records evaluated: {report['total_records_evaluated']}")
        print(f"Missing predictions: {report['missing_predictions_count']}")

        # Display key metrics
        metrics = report["performance_metrics"]
        print("\n=== PERFORMANCE METRICS ===")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.3f}")
        print(f"F1 Score (weighted): {metrics['f1_weighted']:.3f}")
        print(f"Precision (macro): {metrics['precision_macro']:.3f}")
        print(f"Recall (macro): {metrics['recall_macro']:.3f}")

        # Display misclassification patterns
        patterns = report["misclassification_analysis"]["misclassification_patterns"]
        print("\n=== MISCLASSIFICATION PATTERNS ===")
        print(f"PHI misclassified as PII: {patterns['phi_misclassified_as_pii']}")
        print(f"PII misclassified as PHI: {patterns['pii_misclassified_as_phi']}")
        print(f"Sensitive data unclassified: {patterns['sensitive_data_unclassified']}")
        print(
            f"Non-sensitive incorrectly classified: {patterns['nonsensitive_incorrectly_classified']}"
        )

        # Display per-class metrics
        class_metrics = report["misclassification_analysis"]["per_class_metrics"]
        print("\n=== PER-CLASS METRICS ===")
        for class_name, metrics in class_metrics.items():
            print(
                f"{class_name}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, Support={metrics['support']}"
            )

        print(f"\nDetailed results saved to: {results_table}")

except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback

    traceback.print_exc()
    dbutils.notebook.exit("ERROR: " + str(e))

# COMMAND ----------

print("Performance evaluation notebook completed successfully!")
