-- Sample script to create ground truth data for performance benchmarking
-- Modify this according to your specific catalog and schema setup

CREATE OR REPLACE TABLE ${catalog_name}.performance_benchmarks.ground_truth_classifications (
    catalog STRING,
    schema STRING,
    table_name STRING,  
    column_name STRING,
    data_classification STRING,
    data_subclassification STRING,
    confidence_score DOUBLE,
    created_timestamp TIMESTAMP,
    created_by STRING
) USING DELTA;

-- Sample ground truth data - replace with your actual benchmark data
INSERT INTO ${catalog_name}.performance_benchmarks.ground_truth_classifications VALUES
-- PII Examples
('${catalog_name}', 'performance_benchmarks', 'customer_data', 'email', 'PII', 'EMAIL', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'customer_data', 'phone_number', 'PII', 'PHONE', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'customer_data', 'ssn', 'PII', 'SSN', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'customer_data', 'full_name', 'PII', 'NAME', 1.0, current_timestamp(), current_user()),

-- PHI Examples  
('${catalog_name}', 'performance_benchmarks', 'patient_records', 'patient_name', 'PHI', 'PATIENT_NAME', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'patient_records', 'medical_record_number', 'PHI', 'MRN', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'patient_records', 'diagnosis', 'PHI', 'DIAGNOSIS', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'patient_records', 'prescription', 'PHI', 'PRESCRIPTION', 1.0, current_timestamp(), current_user()),

-- PCI Examples
('${catalog_name}', 'performance_benchmarks', 'payment_data', 'credit_card_number', 'PCI', 'CREDIT_CARD', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'payment_data', 'cardholder_name', 'PCI', 'CARDHOLDER_NAME', 1.0, current_timestamp(), current_user()),

-- Non-sensitive Examples
('${catalog_name}', 'performance_benchmarks', 'product_catalog', 'product_id', 'NON_SENSITIVE', 'IDENTIFIER', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'product_catalog', 'product_name', 'NON_SENSITIVE', 'PRODUCT_INFO', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'product_catalog', 'price', 'NON_SENSITIVE', 'FINANCIAL_NON_SENSITIVE', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'sales_summary', 'total_sales', 'NON_SENSITIVE', 'AGGREGATE_METRIC', 1.0, current_timestamp(), current_user()),
('${catalog_name}', 'performance_benchmarks', 'sales_summary', 'region', 'NON_SENSITIVE', 'GEOGRAPHIC_AGGREGATE', 1.0, current_timestamp(), current_user());

-- View the created ground truth data
SELECT 
    data_classification,
    data_subclassification,
    COUNT(*) as count
FROM ${catalog_name}.performance_benchmarks.ground_truth_classifications 
GROUP BY data_classification, data_subclassification
ORDER BY data_classification, data_subclassification;
