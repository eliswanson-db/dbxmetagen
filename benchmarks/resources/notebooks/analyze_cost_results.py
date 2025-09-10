# Databricks notebook source
# MAGIC %md
# MAGIC # Analyze Cost Benchmark Results
# MAGIC
# MAGIC This notebook analyzes the results from cost benchmarking runs.

# COMMAND ----------

import sys

sys.path.append("../../src")

from evaluation.cost_evaluator import CostEvaluator
from pyspark.sql import SparkSession
import json

# COMMAND ----------

# Get parameters
dbutils.widgets.text("catalog_name", "dbxmetagen")
dbutils.widgets.text("schema_name", "benchmarks")
dbutils.widgets.text("results_table", "")
dbutils.widgets.text("report_table", "")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
results_table = dbutils.widgets.get("results_table")
report_table = dbutils.widgets.get("report_table")

# COMMAND ----------

# Initialize evaluator
spark = SparkSession.builder.getOrCreate()
evaluator = CostEvaluator(spark, catalog_name, schema_name)

print(f"Analyzing cost benchmark results:")
print(f"- Results table: {results_table}")
print(f"- Report table: {report_table}")

# COMMAND ----------

# Create results table if it doesn't exist (reference the benchmark_runs table)
benchmark_runs_table = f"{catalog_name}.{schema_name}.benchmark_runs"

# Check if we have any completed runs
completed_runs_count = spark.sql(
    f"""
    SELECT COUNT(*) as count 
    FROM {benchmark_runs_table} 
    WHERE status = 'COMPLETED'
"""
).collect()[0]["count"]

print(f"Found {completed_runs_count} completed benchmark runs")

if completed_runs_count == 0:
    print("No completed benchmark runs found. Cannot generate analysis.")
    dbutils.notebook.exit("NO_DATA: No completed runs found")

# COMMAND ----------

# Generate comprehensive cost report
try:
    report = evaluator.generate_cost_report(
        results_table=benchmark_runs_table, report_output_table=report_table
    )

    print("=== COST ANALYSIS REPORT ===")
    print(f"Report generated at: {report['report_timestamp']}")
    print(f"Total scenarios analyzed: {len(report['scenarios_analyzed'])}")

    # Display summary statistics
    if "summary_statistics" in report:
        stats = report["summary_statistics"]
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total benchmark runs: {stats['total_benchmark_runs']}")
        print(
            f"Average execution time: {stats['avg_execution_time_seconds']:.2f} seconds"
        )
        print(f"Min execution time: {stats['min_execution_time_seconds']:.2f} seconds")
        print(f"Max execution time: {stats['max_execution_time_seconds']:.2f} seconds")
        print(f"Average estimated cost: ${stats['avg_cost_estimate']:.4f}")
        print(f"Total estimated cost: ${stats['total_estimated_cost']:.4f}")

    # Display scenario analysis
    print(f"\n=== SCENARIO ANALYSIS ===")
    for scenario in report["scenarios_analyzed"]:
        print(f"\nScenario: {scenario['scenario_name']}")
        print(f"  Total runs: {scenario['total_runs']}")

        if "best_performance" in scenario:
            best_perf = scenario["best_performance"]
            print(
                f"  Best performance: {best_perf['tables_per_second']:.2f} tables/sec ({best_perf['num_workers']} workers)"
            )

        if "best_cost" in scenario:
            best_cost = scenario["best_cost"]
            print(
                f"  Best cost: ${best_cost['cost_estimate']:.4f} ({best_cost['num_workers']} workers)"
            )

        if "best_efficiency" in scenario:
            best_eff = scenario["best_efficiency"]
            print(
                f"  Best efficiency: {best_eff['cost_efficiency']:.2f} tables/sec/$ ({best_eff['num_workers']} workers)"
            )

    print(f"\nDetailed report saved to: {report_table}")

except Exception as e:
    print(f"Error generating cost report: {e}")
    import traceback

    traceback.print_exc()
    dbutils.notebook.exit("ERROR: " + str(e))

# COMMAND ----------

# Generate additional insights
print("\n=== ADDITIONAL INSIGHTS ===")

# Worker scaling analysis
worker_analysis_query = f"""
SELECT 
    JSON_EXTRACT_SCALAR(cluster_config, '$.num_workers') as num_workers,
    JSON_EXTRACT_SCALAR(cluster_config, '$.scenario') as scenario_type,
    AVG(execution_time_seconds) as avg_execution_time,
    AVG(cost_estimate) as avg_cost,
    COUNT(*) as run_count
FROM {benchmark_runs_table}
WHERE status = 'COMPLETED'
GROUP BY 
    JSON_EXTRACT_SCALAR(cluster_config, '$.num_workers'),
    JSON_EXTRACT_SCALAR(cluster_config, '$.scenario')
ORDER BY scenario_type, CAST(num_workers AS INT)
"""

try:
    worker_analysis = spark.sql(worker_analysis_query).toPandas()

    if not worker_analysis.empty:
        print("\nWorker scaling analysis:")
        for _, row in worker_analysis.iterrows():
            scenario_parts = (
                row["scenario_type"].split("_") if row["scenario_type"] else ["unknown"]
            )
            scenario_base = (
                "_".join(scenario_parts[:-1])
                if len(scenario_parts) > 1
                else row["scenario_type"]
            )

            print(
                f"  {scenario_base} with {row['num_workers']} workers: "
                f"{row['avg_execution_time']:.1f}s avg time, "
                f"${row['avg_cost']:.4f} avg cost "
                f"({row['run_count']} runs)"
            )

except Exception as e:
    print(f"Error in worker analysis: {e}")

# COMMAND ----------

# Performance scaling insights
performance_query = f"""
SELECT 
    JSON_EXTRACT_SCALAR(processing_metrics, '$.total_tables') as total_tables,
    JSON_EXTRACT_SCALAR(processing_metrics, '$.total_columns') as total_columns,
    JSON_EXTRACT_SCALAR(processing_metrics, '$.total_rows') as total_rows,
    JSON_EXTRACT_SCALAR(cluster_config, '$.num_workers') as num_workers,
    execution_time_seconds,
    cost_estimate,
    JSON_EXTRACT_SCALAR(processing_metrics, '$.tables_per_second') as tables_per_second,
    JSON_EXTRACT_SCALAR(processing_metrics, '$.columns_per_second') as columns_per_second
FROM {benchmark_runs_table}
WHERE status = 'COMPLETED'
ORDER BY CAST(total_tables AS INT), CAST(total_columns AS INT), CAST(total_rows AS INT)
"""

try:
    performance_df = spark.sql(performance_query).toPandas()

    if not performance_df.empty:
        print("\n=== PERFORMANCE SCALING INSIGHTS ===")

        # Group by data size characteristics
        for _, row in performance_df.iterrows():
            size_desc = (
                f"{row['total_tables']}T×{row['total_columns']}C×{row['total_rows']}R"
            )
            print(
                f"{size_desc} ({row['num_workers']}w): "
                f"{float(row['tables_per_second']):.2f} tables/sec, "
                f"{float(row['columns_per_second']):.1f} cols/sec, "
                f"{row['execution_time_seconds']:.1f}s, "
                f"${row['cost_estimate']:.4f}"
            )

except Exception as e:
    print(f"Error in performance analysis: {e}")

# COMMAND ----------

print("\nCost analysis completed successfully!")
print(f"Comprehensive report available in: {report_table}")
print(f"Raw benchmark data available in: {benchmark_runs_table}")
