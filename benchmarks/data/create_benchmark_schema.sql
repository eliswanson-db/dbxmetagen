-- Setup script for dbxmetagen benchmarking infrastructure
-- Run this script to create the required schemas and initial tables

-- Performance benchmarking schema and ground truth table
CREATE SCHEMA IF NOT EXISTS ${catalog_name}.performance_benchmarks;
CREATE SCHEMA IF NOT EXISTS ${catalog_name}.benchmarks;
CREATE SCHEMA IF NOT EXISTS ${catalog_name}.cost_benchmarks;

-- Create volume for storing generated metadata files
CREATE VOLUME IF NOT EXISTS ${catalog_name}.performance_benchmarks.generated_metadata;

-- Create ground truth table for performance evaluation
CREATE OR REPLACE TABLE ${catalog_name}.performance_benchmarks.ground_truth_classifications (
    catalog STRING,
    schema STRING,
    table_name STRING,  
    column_name STRING,
    data_classification STRING COMMENT 'Ground truth classification: PII, PHI, PCI, NON_SENSITIVE',
    data_subclassification STRING COMMENT 'Detailed sub-classification',
    confidence_score DOUBLE COMMENT 'Confidence in the ground truth label',
    notes STRING COMMENT 'Additional notes about the classification',
    created_timestamp TIMESTAMP DEFAULT current_timestamp(),
    created_by STRING DEFAULT current_user()
) USING DELTA
COMMENT 'Ground truth data for evaluating dbxmetagen classification performance';

-- Create results table for performance evaluations  
CREATE OR REPLACE TABLE ${catalog_name}.benchmarks.performance_evaluation_results (
    evaluation_id STRING,
    join_key STRING COMMENT 'catalog.schema.table.column identifier',
    catalog STRING,
    schema STRING,
    table_name STRING,
    column_name STRING,
    data_classification_actual STRING COMMENT 'Ground truth classification',
    data_subclassification_actual STRING COMMENT 'Ground truth sub-classification', 
    classification_predicted STRING COMMENT 'dbxmetagen predicted classification',
    type_predicted STRING COMMENT 'dbxmetagen predicted type',
    is_correct BOOLEAN COMMENT 'Whether prediction matches ground truth',
    evaluation_timestamp TIMESTAMP,
    evaluation_user STRING
) USING DELTA
COMMENT 'Detailed results from performance evaluations';

-- Create table for cost benchmark tracking
CREATE OR REPLACE TABLE ${catalog_name}.benchmarks.benchmark_runs (
    run_id STRING PRIMARY KEY,
    scenario_name STRING,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    execution_time_seconds DOUBLE,
    cluster_config STRING COMMENT 'JSON string with cluster configuration',
    table_metadata STRING COMMENT 'JSON string with table metadata',
    processing_metrics STRING COMMENT 'JSON string with processing metrics',
    cost_estimate DOUBLE COMMENT 'Estimated cost in USD',
    status STRING COMMENT 'STARTED, COMPLETED, FAILED',
    created_timestamp TIMESTAMP DEFAULT current_timestamp()
) USING DELTA
COMMENT 'Tracks individual cost benchmark runs';

-- Create table for cost analysis reports
CREATE OR REPLACE TABLE ${catalog_name}.benchmarks.cost_analysis_report (
    report_id STRING PRIMARY KEY,
    report_timestamp TIMESTAMP,
    report_data STRING COMMENT 'JSON string with complete analysis report',
    created_timestamp TIMESTAMP DEFAULT current_timestamp()
) USING DELTA
COMMENT 'Stores cost analysis reports';

-- Grant permissions (adjust as needed for your environment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA ${catalog_name}.performance_benchmarks TO `your-group`;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA ${catalog_name}.benchmarks TO `your-group`;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA ${catalog_name}.cost_benchmarks TO `your-group`;

SELECT 
    'Benchmarking infrastructure created successfully!' as status,
    '${catalog_name}.performance_benchmarks' as performance_schema,
    '${catalog_name}.benchmarks' as results_schema,
    '${catalog_name}.cost_benchmarks' as cost_benchmarks_schema;
