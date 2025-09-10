"""
Performance evaluation utilities for dbxmetagen benchmarking.
"""

from typing import Dict, List, Tuple
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from datetime import datetime


class PerformanceEvaluator:
    """Evaluates dbxmetagen performance against benchmark data."""

    def __init__(self, spark: SparkSession):
        """
        Initialize the performance evaluator.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def load_dbxmetagen_output(self, file_path: str) -> pd.DataFrame:
        """
        Load dbxmetagen TSV output file.

        Args:
            file_path: Path to the TSV file

        Returns:
            DataFrame with dbxmetagen output
        """
        try:
            df = pd.read_csv(file_path, sep="\t")
            return df
        except Exception as e:
            print(f"Error loading dbxmetagen output: {e}")  # noqa: W0703
            return pd.DataFrame()

    def load_benchmark_data(self, table_name: str) -> pd.DataFrame:
        """
        Load benchmark data from Unity Catalog table.

        Args:
            table_name: Full table name (catalog.schema.table)

        Returns:
            DataFrame with benchmark data
        """
        try:
            spark_df = self.spark.table(table_name)
            return spark_df.toPandas()
        except Exception as e:
            print(f"Error loading benchmark data: {e}")  # noqa: W0703
            return pd.DataFrame()

    def prepare_data_for_evaluation(
        self, dbxmetagen_output: pd.DataFrame, benchmark_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare and align data for evaluation.

        Args:
            dbxmetagen_output: Output from dbxmetagen
            benchmark_data: Ground truth benchmark data

        Returns:
            Tuple of (merged_df, missing_predictions)
        """
        # Normalize column names and create join keys
        dbxmetagen_output["join_key"] = (
            dbxmetagen_output["catalog"].astype(str)
            + "."
            + dbxmetagen_output["schema"].astype(str)
            + "."
            + dbxmetagen_output["table"].astype(str)
            + "."
            + dbxmetagen_output["column"].astype(str)
        )

        benchmark_data["join_key"] = (
            benchmark_data["catalog"].astype(str)
            + "."
            + benchmark_data["schema"].astype(str)
            + "."
            + benchmark_data["table"].astype(str)
            + "."
            + benchmark_data["column"].astype(str)
        )

        # Merge datasets
        merged_df = benchmark_data.merge(
            dbxmetagen_output[["join_key", "classification", "type"]],
            on="join_key",
            how="left",
            suffixes=("_actual", "_predicted"),
        )

        # Find missing predictions
        missing_predictions = merged_df[merged_df["classification"].isna()][
            "join_key"
        ].tolist()

        # Fill missing predictions with "UNCLASSIFIED"
        merged_df["classification"] = merged_df["classification"].fillna("UNCLASSIFIED")
        merged_df["type"] = merged_df["type"].fillna("UNCLASSIFIED")

        return merged_df, missing_predictions

    def evaluate_classification_performance(
        self,
        merged_df: pd.DataFrame,
        classification_column: str = "data_classification",
    ) -> Dict[str, float]:
        """
        Evaluate classification performance metrics.

        Args:
            merged_df: Merged dataframe with actual and predicted values
            classification_column: Column name for ground truth classification

        Returns:
            Dictionary with performance metrics
        """
        y_true = merged_df[classification_column]
        y_pred = merged_df["classification"]

        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate metrics with different averaging strategies
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall_weighted = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
        }

    def analyze_misclassification_patterns(
        self,
        merged_df: pd.DataFrame,
        classification_column: str = "data_classification",
    ) -> Dict[str, any]:
        """
        Analyze specific misclassification patterns.

        Args:
            merged_df: Merged dataframe with actual and predicted values
            classification_column: Column name for ground truth classification

        Returns:
            Dictionary with misclassification analysis
        """
        y_true = merged_df[classification_column]
        y_pred = merged_df["classification"]

        # Confusion matrix
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Specific pattern analysis
        patterns = {}

        # PHI misclassified as PII
        phi_as_pii = len(
            merged_df[
                (merged_df[classification_column] == "PHI")
                & (merged_df["classification"] == "PII")
            ]
        )
        patterns["phi_misclassified_as_pii"] = phi_as_pii

        # PII misclassified as PHI
        pii_as_phi = len(
            merged_df[
                (merged_df[classification_column] == "PII")
                & (merged_df["classification"] == "PHI")
            ]
        )
        patterns["pii_misclassified_as_phi"] = pii_as_phi

        # Sensitive data not classified
        sensitive_unclassified = len(
            merged_df[
                (merged_df[classification_column].isin(["PHI", "PII", "PCI"]))
                & (merged_df["classification"] == "UNCLASSIFIED")
            ]
        )
        patterns["sensitive_data_unclassified"] = sensitive_unclassified

        # Non-sensitive data incorrectly classified
        nonsensitive_classified = len(
            merged_df[
                (~merged_df[classification_column].isin(["PHI", "PII", "PCI"]))
                & (merged_df["classification"].isin(["PHI", "PII", "PCI"]))
            ]
        )
        patterns["nonsensitive_incorrectly_classified"] = nonsensitive_classified

        # Per-class precision and recall
        class_metrics = {}
        for label in labels:
            tp = len(
                merged_df[
                    (merged_df[classification_column] == label)
                    & (merged_df["classification"] == label)
                ]
            )
            fp = len(
                merged_df[
                    (merged_df[classification_column] != label)
                    & (merged_df["classification"] == label)
                ]
            )
            fn = len(
                merged_df[
                    (merged_df[classification_column] == label)
                    & (merged_df["classification"] != label)
                ]
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": len(merged_df[merged_df[classification_column] == label]),
            }

        return {
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": labels,
            "misclassification_patterns": patterns,
            "per_class_metrics": class_metrics,
        }

    def generate_performance_report(
        self,
        dbxmetagen_output_path: str,
        benchmark_table: str,
        report_output_table: str,
    ) -> Dict[str, any]:
        """
        Generate comprehensive performance evaluation report.

        Args:
            dbxmetagen_output_path: Path to dbxmetagen TSV output
            benchmark_table: Full name of benchmark table
            report_output_table: Table to save detailed results

        Returns:
            Complete evaluation report
        """
        # Load data
        dbxmetagen_output = self.load_dbxmetagen_output(dbxmetagen_output_path)
        benchmark_data = self.load_benchmark_data(benchmark_table)

        if dbxmetagen_output.empty or benchmark_data.empty:
            return {"error": "Failed to load required data"}

        # Prepare data
        merged_df, missing_predictions = self.prepare_data_for_evaluation(
            dbxmetagen_output, benchmark_data
        )

        # Calculate metrics
        performance_metrics = self.evaluate_classification_performance(merged_df)
        misclassification_analysis = self.analyze_misclassification_patterns(merged_df)

        # Prepare report
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_records_evaluated": len(merged_df),
            "missing_predictions_count": len(missing_predictions),
            "performance_metrics": performance_metrics,
            "misclassification_analysis": misclassification_analysis,
            "missing_predictions": missing_predictions[:100],  # Limit for display
        }

        # Save detailed results to Delta table
        self._save_detailed_results(merged_df, report_output_table)

        return report

    def _save_detailed_results(
        self, merged_df: pd.DataFrame, output_table: str
    ) -> None:
        """
        Save detailed evaluation results to Delta table.

        Args:
            merged_df: Merged evaluation dataframe
            output_table: Output table name
        """
        try:
            # Add evaluation metadata
            merged_df["evaluation_timestamp"] = datetime.now()
            merged_df["is_correct"] = (
                merged_df["data_classification"] == merged_df["classification"]
            )

            # Convert to Spark DataFrame and save
            spark_df = self.spark.createDataFrame(merged_df)
            spark_df.write.format("delta").mode("overwrite").saveAsTable(output_table)

            print(f"Detailed results saved to {output_table}")

        except Exception as e:
            print(f"Error saving detailed results: {e}")  # noqa: W0703
