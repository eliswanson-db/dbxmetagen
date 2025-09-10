"""
Cost evaluation utilities for dbxmetagen benchmarking.
"""

from typing import Dict, Optional
import time
from datetime import datetime
from pyspark.sql import SparkSession
import json


class CostEvaluator:
    """Evaluates cost and performance metrics for different cluster configurations."""

    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        """
        Initialize the cost evaluator.

        Args:
            spark: SparkSession instance
            catalog: Catalog for storing results
            schema: Schema for storing results
        """
        self.spark = spark
        self.catalog = catalog
        self.schema = schema

    def record_benchmark_start(
        self,
        scenario_name: str,
        cluster_config: Dict[str, any],
        table_metadata: Dict[str, any],
    ) -> str:
        """
        Record the start of a benchmark run.

        Args:
            scenario_name: Name of the benchmarking scenario
            cluster_config: Configuration of the cluster
            table_metadata: Metadata about tables being processed

        Returns:
            Benchmark run ID
        """
        run_id = f"{scenario_name}_{int(time.time())}"

        benchmark_record = {
            "run_id": run_id,
            "scenario_name": scenario_name,
            "start_time": datetime.now(),
            "cluster_config": json.dumps(cluster_config),
            "table_metadata": json.dumps(table_metadata),
            "status": "STARTED",
        }

        # Store initial record
        self._store_benchmark_record(benchmark_record)

        return run_id

    def record_benchmark_completion(
        self,
        run_id: str,
        execution_time_seconds: float,
        processing_metrics: Dict[str, any],
        cost_estimate: Optional[float] = None,
    ) -> None:
        """
        Record the completion of a benchmark run.

        Args:
            run_id: Benchmark run ID
            execution_time_seconds: Total execution time
            processing_metrics: Metrics about the processing
            cost_estimate: Estimated cost if available
        """
        completion_record = {
            "run_id": run_id,
            "end_time": datetime.now(),
            "execution_time_seconds": execution_time_seconds,
            "processing_metrics": json.dumps(processing_metrics),
            "cost_estimate": cost_estimate,
            "status": "COMPLETED",
        }

        self._update_benchmark_record(completion_record)

    def calculate_throughput_metrics(
        self,
        total_tables: int,
        total_columns: int,
        total_rows: int,
        execution_time_seconds: float,
    ) -> Dict[str, float]:
        """
        Calculate throughput metrics for the benchmark run.

        Args:
            total_tables: Number of tables processed
            total_columns: Number of columns processed
            total_rows: Total number of rows across all tables
            execution_time_seconds: Execution time in seconds

        Returns:
            Throughput metrics
        """
        if execution_time_seconds <= 0:
            return {}

        return {
            "tables_per_second": total_tables / execution_time_seconds,
            "columns_per_second": total_columns / execution_time_seconds,
            "rows_per_second": total_rows / execution_time_seconds,
            "execution_time_minutes": execution_time_seconds / 60,
            "total_tables": total_tables,
            "total_columns": total_columns,
            "total_rows": total_rows,
        }

    def estimate_cost(
        self,
        cluster_config: Dict[str, any],
        execution_time_seconds: float,
        dbu_cost_per_hour: float = 0.30,  # Approximate cost per DBU hour
    ) -> Dict[str, float]:
        """
        Estimate the cost of running dbxmetagen.

        Args:
            cluster_config: Cluster configuration
            execution_time_seconds: Execution time
            dbu_cost_per_hour: Cost per DBU hour

        Returns:
            Cost estimates
        """
        num_workers = cluster_config.get("num_workers", 0)
        driver_dbus = 1  # Assuming 1 DBU for driver
        worker_dbus_each = 1  # Assuming 1 DBU per worker

        total_dbus = driver_dbus + (num_workers * worker_dbus_each)
        execution_hours = execution_time_seconds / 3600

        estimated_cost = total_dbus * execution_hours * dbu_cost_per_hour

        return {
            "total_dbus": total_dbus,
            "execution_hours": execution_hours,
            "estimated_cost_usd": estimated_cost,
            "cost_per_table": estimated_cost / cluster_config.get("total_tables", 1),
            "cost_per_column": estimated_cost / cluster_config.get("total_columns", 1),
        }

    def compare_cluster_performance(
        self, scenario_name: str, results_table: str
    ) -> Dict[str, any]:
        """
        Compare performance across different cluster configurations.

        Args:
            scenario_name: Scenario to analyze
            results_table: Table containing benchmark results

        Returns:
            Comparative analysis
        """
        query = f"""
        SELECT 
            run_id,
            scenario_name,
            cluster_config,
            execution_time_seconds,
            processing_metrics,
            cost_estimate,
            start_time,
            end_time
        FROM {results_table}
        WHERE scenario_name = '{scenario_name}'
        AND status = 'COMPLETED'
        ORDER BY start_time DESC
        """

        df = self.spark.sql(query).toPandas()

        if df.empty:
            return {"error": "No completed benchmarks found for scenario"}

        # Parse JSON columns
        df["cluster_config_parsed"] = df["cluster_config"].apply(json.loads)
        df["processing_metrics_parsed"] = df["processing_metrics"].apply(json.loads)

        # Extract key metrics
        comparisons = []
        for _, row in df.iterrows():
            cluster_config = row["cluster_config_parsed"]
            processing_metrics = row["processing_metrics_parsed"]

            comparison = {
                "run_id": row["run_id"],
                "num_workers": cluster_config.get("num_workers", 0),
                "execution_time_seconds": row["execution_time_seconds"],
                "tables_per_second": processing_metrics.get("tables_per_second", 0),
                "columns_per_second": processing_metrics.get("columns_per_second", 0),
                "cost_estimate": row["cost_estimate"],
                "cost_efficiency": (
                    processing_metrics.get("tables_per_second", 0)
                    / (row["cost_estimate"] or 1)
                ),
            }
            comparisons.append(comparison)

        # Find optimal configuration
        best_performance = max(comparisons, key=lambda x: x["tables_per_second"])
        best_cost = min(comparisons, key=lambda x: x["cost_estimate"] or float("inf"))
        best_efficiency = max(comparisons, key=lambda x: x["cost_efficiency"])

        return {
            "scenario_name": scenario_name,
            "total_runs": len(comparisons),
            "all_runs": comparisons,
            "best_performance": best_performance,
            "best_cost": best_cost,
            "best_efficiency": best_efficiency,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def generate_cost_report(
        self, results_table: str, report_output_table: str
    ) -> Dict[str, any]:
        """
        Generate comprehensive cost analysis report.

        Args:
            results_table: Table with benchmark results
            report_output_table: Table to save report

        Returns:
            Cost analysis report
        """
        # Get all scenarios
        scenarios_df = self.spark.sql(
            f"""
            SELECT DISTINCT scenario_name 
            FROM {results_table} 
            WHERE status = 'COMPLETED'
        """
        ).toPandas()

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "scenarios_analyzed": [],
            "summary_statistics": {},
        }

        all_runs = []

        for scenario in scenarios_df["scenario_name"]:
            scenario_analysis = self.compare_cluster_performance(
                scenario, results_table
            )
            if "error" not in scenario_analysis:
                report["scenarios_analyzed"].append(scenario_analysis)
                all_runs.extend(scenario_analysis["all_runs"])

        # Calculate summary statistics
        if all_runs:
            execution_times = [run["execution_time_seconds"] for run in all_runs]
            costs = [run["cost_estimate"] for run in all_runs if run["cost_estimate"]]

            report["summary_statistics"] = {
                "total_benchmark_runs": len(all_runs),
                "avg_execution_time_seconds": sum(execution_times)
                / len(execution_times),
                "min_execution_time_seconds": min(execution_times),
                "max_execution_time_seconds": max(execution_times),
                "avg_cost_estimate": sum(costs) / len(costs) if costs else 0,
                "total_estimated_cost": sum(costs) if costs else 0,
            }

        # Save report to Delta table
        self._save_cost_report(report, report_output_table)

        return report

    def _store_benchmark_record(self, record: Dict[str, any]) -> None:
        """Store initial benchmark record."""
        table_name = f"{self.catalog}.{self.schema}.benchmark_runs"

        # Convert to Spark DataFrame
        df = self.spark.createDataFrame([record])

        # Save to Delta table (append mode)
        df.write.format("delta").mode("append").saveAsTable(table_name)

    def _update_benchmark_record(self, completion_record: Dict[str, any]) -> None:
        """Update benchmark record with completion data."""
        table_name = f"{self.catalog}.{self.schema}.benchmark_runs"

        # Use MERGE to update the existing record
        temp_view = "completion_update"
        completion_df = self.spark.createDataFrame([completion_record])
        completion_df.createOrReplaceTempView(temp_view)

        merge_sql = f"""
        MERGE INTO {table_name} t
        USING {temp_view} s ON t.run_id = s.run_id
        WHEN MATCHED THEN UPDATE SET
            end_time = s.end_time,
            execution_time_seconds = s.execution_time_seconds,
            processing_metrics = s.processing_metrics,
            cost_estimate = s.cost_estimate,
            status = s.status
        """

        self.spark.sql(merge_sql)

    def _save_cost_report(self, report: Dict[str, any], output_table: str) -> None:
        """Save cost analysis report to Delta table."""
        try:
            report_record = {
                "report_id": f"cost_report_{int(time.time())}",
                "report_timestamp": datetime.now(),
                "report_data": json.dumps(report),
            }

            df = self.spark.createDataFrame([report_record])
            df.write.format("delta").mode("append").saveAsTable(output_table)

            print(f"Cost report saved to {output_table}")

        except Exception as e:
            print(f"Error saving cost report: {e}")  # noqa: W0703
