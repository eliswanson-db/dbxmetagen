# dbxmetagen Benchmarking Suite

This benchmarking suite provides comprehensive performance and cost evaluation capabilities for dbxmetagen.

## Overview

The benchmarking suite consists of two main types of benchmarks:

1. **Performance Benchmarking**: Evaluates the accuracy and classification performance of dbxmetagen against ground truth data
2. **Cost Benchmarking**: Measures execution time and cost across different cluster configurations and data volumes

## Structure

```
benchmarks/
├── src/
│   ├── benchmarks/           # Core benchmarking utilities
│   ├── evaluation/          # Performance and cost evaluation modules
│   └── table_generation/    # Synthetic table generation utilities
├── resources/
│   ├── jobs/               # Databricks job configurations
│   └── notebooks/          # Driver notebooks for benchmarking
├── data/                   # Sample benchmark data
└── results/               # Benchmark results and reports
```

## Performance Benchmarking

### Prerequisites

1. Create a schema called `performance_benchmarks` in your target catalog
2. Create a ground truth table with the following columns:
   - `catalog`: Catalog name
   - `schema`: Schema name  
   - `table`: Table name
   - `column`: Column name
   - `data_classification`: Ground truth classification (PII, PHI, PCI, etc.)
   - `data_subclassification`: Ground truth sub-classification

### Running Performance Benchmarks

1. Deploy the performance benchmark job:
   ```bash
   databricks bundle deploy --target <environment>
   ```

2. Run the job:
   ```bash
   databricks jobs run-now --job-name "dbxmetagen_performance_benchmark_<environment>"
   ```

### Evaluation Metrics

The performance benchmark evaluates:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Both macro and weighted averages
- **Misclassification Analysis**: Specific patterns like PHI→PII, PII→PHI, sensitive→unclassified
- **Per-class Metrics**: Individual performance for each classification type

## Cost Benchmarking

### Test Scenarios

The cost benchmark runs three scenarios with both 2-worker and 4-worker clusters:

1. **Scenario 1**: 5 tables × 10 columns × 10k rows
2. **Scenario 2**: 5 tables × 10 columns × 1M rows  
3. **Scenario 3**: 5 tables × 100 columns × 10k rows

### Running Cost Benchmarks

1. Deploy the cost benchmark job:
   ```bash
   databricks bundle deploy --target <environment>
   ```

2. Run the job:
   ```bash
   databricks jobs run-now --job-name "dbxmetagen_cost_benchmark_<environment>"
   ```

### Cost Analysis

The cost benchmark provides:
- **Throughput Metrics**: Tables/second, columns/second, rows/second
- **Cost Estimates**: Based on DBU consumption and execution time
- **Scaling Analysis**: Performance comparison across worker counts
- **Efficiency Metrics**: Cost per unit of work processed

## Key Components

### Table Generator (`BenchmarkTableGenerator`)
- Generates realistic synthetic tables with various data types
- Includes sensitive data patterns (PII, PHI, PCI)
- Supports configurable table dimensions and row counts

### Performance Evaluator (`PerformanceEvaluator`)
- Loads and compares dbxmetagen output against ground truth
- Calculates comprehensive classification metrics
- Provides detailed misclassification pattern analysis

### Cost Evaluator (`CostEvaluator`)
- Tracks benchmark execution timing and resource usage
- Estimates costs based on cluster configuration
- Provides comparative analysis across scenarios

## Usage Examples

### Creating Custom Benchmark Tables

```python
from benchmarks.src.table_generation.delta_table_generator import BenchmarkTableGenerator

generator = BenchmarkTableGenerator(spark, "my_catalog", "benchmark_schema")
tables = generator.create_tables_for_scenario(
    scenario_name="custom_test",
    num_tables=3,
    num_columns=20,
    num_rows=50000,
    include_sensitive_data=True
)
```

### Running Performance Evaluation

```python
from benchmarks.src.evaluation.performance_evaluator import PerformanceEvaluator

evaluator = PerformanceEvaluator(spark)
report = evaluator.generate_performance_report(
    dbxmetagen_output_path="/path/to/output.tsv",
    benchmark_table="catalog.schema.ground_truth",
    report_output_table="catalog.schema.evaluation_results"
)
```

### Analyzing Cost Results

```python
from benchmarks.src.evaluation.cost_evaluator import CostEvaluator

evaluator = CostEvaluator(spark, "catalog", "schema")
analysis = evaluator.compare_cluster_performance(
    scenario_name="scenario_1",
    results_table="catalog.schema.benchmark_runs"
)
```

## Configuration

### Job Configuration

The benchmark jobs are configured via Databricks Asset Bundles. Key parameters:

- `catalog_name`: Target catalog for benchmarking
- `current_user`: User email for notifications
- Cluster configurations are defined in the job YAML files

### Environment Variables

Set these in your `variables.yml`:

```yaml
catalog_name: "your_catalog"
current_user: "your.email@company.com"
```

## Output Tables

### Performance Benchmarking
- `{catalog}.benchmarks.performance_evaluation_results`: Detailed evaluation results
- Output includes record-level predictions vs. ground truth

### Cost Benchmarking
- `{catalog}.benchmarks.benchmark_runs`: Individual benchmark run records
- `{catalog}.benchmarks.cost_analysis_report`: Aggregated cost analysis
- `{catalog}.cost_benchmarks.table_generation_metadata`: Generated table metadata

## Best Practices

1. **Performance Benchmarking**:
   - Use representative data that matches your actual use cases
   - Include diverse classification types in ground truth data
   - Run benchmarks after any model or configuration changes

2. **Cost Benchmarking**:
   - Run cost benchmarks in isolated environments
   - Use consistent cluster configurations for fair comparison
   - Consider running multiple iterations for statistical significance

3. **General**:
   - Monitor resource usage during benchmarking
   - Clean up generated tables after cost benchmarking if desired
   - Review detailed logs for debugging failed runs

## Troubleshooting

### Common Issues

1. **Missing Output Files**: Check that dbxmetagen completed successfully and output files exist in the expected Volume path
2. **Permission Errors**: Ensure proper permissions on target catalogs and schemas
3. **Resource Constraints**: Increase cluster size or timeout values for large datasets

### Debugging

- Check notebook output and error logs in the Databricks job run details
- Verify table existence and permissions using SQL queries
- Use the evaluation modules directly in notebooks for step-by-step debugging

## Contributing

When extending the benchmarking suite:

1. Follow the existing module structure
2. Add comprehensive docstrings and type hints
3. Include error handling and logging
4. Update this README with new functionality
5. Test with various data sizes and configurations
