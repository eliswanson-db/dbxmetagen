# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy Training Quickstart for dbxmetagen
# MAGIC
# MAGIC This notebook demonstrates how to train and optimize DSPy prompts for the dbxmetagen CommentGenerator.
# MAGIC
# MAGIC ## Overview
# MAGIC - Set up DSPy integration with existing dbxmetagen system
# MAGIC - Create training data from real Databricks tables
# MAGIC - Train and optimize prompts using DSPy
# MAGIC - Compare performance between traditional and DSPy approaches
# MAGIC - Save and load optimized models
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - DSPy library installed (`pip install dspy-ai>=2.4.0`)
# MAGIC - OpenAI API key configured
# MAGIC - Access to Databricks tables for training data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

# Install DSPy if not already installed
%pip install dspy-ai>=2.4.0

# COMMAND ----------

import os
import json
import pandas as pd
from typing import List, Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max as spark_max, min as spark_min

# Import dbxmetagen components
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.dspy_comment_generator import (
    DSPyCommentGeneratorModel,
    DSPyCommentGeneratorFactory
)
from src.dbxmetagen.enhanced_factory import (
    EnhancedMetadataGeneratorFactory,
    DSPyMigrationHelper
)

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

print("✅ Setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC
# MAGIC Configure your OpenAI API key and model settings.

# COMMAND ----------

# Configure OpenAI API key
# Option 1: Set as environment variable (recommended)
# os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Option 2: Use Databricks secrets (most secure)
# os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("your-scope", "openai-api-key")

# Create configuration for training
config = MetadataConfig(
    mode="comment",
    model="databricks-claude-3-7-sonnet",  # Use cheaper model for training
    max_tokens=2000,
    temperature=0.3,  # Lower temperature for more consistent training
    allow_data_in_comments=True,
    max_prompt_length=50000,
    limit_prompt_based_on_cell_len=True,
    word_limit_per_cell=50,
    add_metadata=True,
    include_datatype_from_metadata=True,
    include_possible_data_fields_in_metadata=False,
    include_existing_table_comment=True,
    acro_content={"ID": "Identifier", "FK": "Foreign Key", "PK": "Primary Key"}
)

print("✅ Configuration created!")
print(f"Model: {config.model}")
print(f"Max tokens: {config.max_tokens}")
print(f"Temperature: {config.temperature}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Training Data from Real Tables
# MAGIC
# MAGIC We'll create training examples by sampling from your actual Databricks tables.

# COMMAND ----------

def create_training_data_from_table(catalog: str, schema: str, table: str, sample_size: int = 2) -> Dict[str, Any]:
    """
    Create training data from a real Databricks table.
    
    Args:
        catalog: Catalog name
        schema: Schema name  
        table: Table name
        sample_size: Number of sample rows to include
        
    Returns:
        Dictionary with training data structure
    """
    full_table_name = f"{catalog}.{schema}.{table}"
    
    try:
        # Get table sample
        df = spark.table(full_table_name)
        sample_df = df.limit(sample_size)
        pandas_sample = sample_df.toPandas()
        
        # Get column metadata
        column_metadata = {}
        for column in df.columns:
            col_df = df.select(
                count("*").alias("total_count"),
                count(col(column)).alias("non_null_count"),
                # Only calculate string length stats for string columns
                *([avg(length(col(column))).alias("avg_len"), spark_max(length(col(column))).alias("max_len")] 
                  if dict(df.dtypes)[column] == 'string' else [])
            )
            stats = col_df.collect()[0]
            
            column_metadata[column] = {
                "col_name": column,
                "data_type": dict(df.dtypes)[column],
                "num_nulls": str(stats["total_count"] - stats["non_null_count"]),
                "distinct_count": str(df.select(column).distinct().count()),
            }
            
            # Add string-specific stats
            if dict(df.dtypes)[column] == 'string' and stats.get("avg_len"):
                column_metadata[column]["avg_col_len"] = str(int(stats["avg_len"] or 0))
                column_metadata[column]["max_col_len"] = str(stats["max_len"] or 0)
        
        # Create training data structure
        training_data = {
            "table_name": full_table_name,
            "table_data": {
                "index": list(range(len(pandas_sample))),
                "columns": list(pandas_sample.columns),
                "data": pandas_sample.values.tolist()
            },
            "column_metadata": column_metadata,
            "abbreviations": config.acro_content
        }
        
        return training_data
        
    except Exception as e:
        print(f"⚠️ Error processing table {full_table_name}: {e}")
        return None

# Example: Create training data from sample tables
# Replace these with your actual table names
sample_tables = [
    # ("your_catalog", "your_schema", "your_table"),
    # Add more tables here for better training data
]

training_examples_raw = []

# If you don't have sample tables, let's create mock data for demonstration
if not sample_tables:
    print("📝 Creating mock training data for demonstration...")
    
    # Mock training example 1: Customer table
    training_examples_raw.append({
        "table_name": "sales.customers.customer_profile",
        "table_data": {
            "index": [0, 1],
            "columns": ["customer_id", "first_name", "last_name", "email", "registration_date", "is_active"],
            "data": [
                ["CUST001", "John", "Doe", "john.doe@email.com", "2023-01-15", True],
                ["CUST002", "Jane", "Smith", "jane.smith@email.com", "2023-02-20", True]
            ]
        },
        "column_metadata": {
            "customer_id": {"col_name": "customer_id", "data_type": "string", "num_nulls": "0", "distinct_count": "1000", "avg_col_len": "7", "max_col_len": "7"},
            "first_name": {"col_name": "first_name", "data_type": "string", "num_nulls": "5", "distinct_count": "800", "avg_col_len": "6", "max_col_len": "15"},
            "last_name": {"col_name": "last_name", "data_type": "string", "num_nulls": "2", "distinct_count": "900", "avg_col_len": "7", "max_col_len": "20"},
            "email": {"col_name": "email", "data_type": "string", "num_nulls": "0", "distinct_count": "1000", "avg_col_len": "20", "max_col_len": "50"},
            "registration_date": {"col_name": "registration_date", "data_type": "date", "num_nulls": "0", "distinct_count": "365"},
            "is_active": {"col_name": "is_active", "data_type": "boolean", "num_nulls": "0", "distinct_count": "2"}
        },
        "abbreviations": config.acro_content
    })
    
    # Mock training example 2: Transaction table
    training_examples_raw.append({
        "table_name": "finance.transactions.payment_history", 
        "table_data": {
            "index": [0, 1],
            "columns": ["transaction_id", "customer_id", "amount", "currency", "payment_method", "transaction_date", "status"],
            "data": [
                ["TXN001", "CUST001", 99.99, "USD", "credit_card", "2023-03-01", "completed"],
                ["TXN002", "CUST002", 149.99, "USD", "bank_transfer", "2023-03-02", "completed"]
            ]
        },
        "column_metadata": {
            "transaction_id": {"col_name": "transaction_id", "data_type": "string", "num_nulls": "0", "distinct_count": "5000", "avg_col_len": "6", "max_col_len": "6"},
            "customer_id": {"col_name": "customer_id", "data_type": "string", "num_nulls": "0", "distinct_count": "1000", "avg_col_len": "7", "max_col_len": "7"},
            "amount": {"col_name": "amount", "data_type": "decimal", "num_nulls": "0", "distinct_count": "500"},
            "currency": {"col_name": "currency", "data_type": "string", "num_nulls": "0", "distinct_count": "3", "avg_col_len": "3", "max_col_len": "3"},
            "payment_method": {"col_name": "payment_method", "data_type": "string", "num_nulls": "0", "distinct_count": "4", "avg_col_len": "10", "max_col_len": "15"},
            "transaction_date": {"col_name": "transaction_date", "data_type": "date", "num_nulls": "0", "distinct_count": "365"},
            "status": {"col_name": "status", "data_type": "string", "num_nulls": "0", "distinct_count": "5", "avg_col_len": "8", "max_col_len": "12"}
        },
        "abbreviations": config.acro_content
    })
else:
    # Create training data from real tables
    for catalog, schema, table in sample_tables:
        training_data = create_training_data_from_table(catalog, schema, table)
        if training_data:
            training_examples_raw.append(training_data)

print(f"✅ Created {len(training_examples_raw)} raw training examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Expected Responses
# MAGIC
# MAGIC We need to generate high-quality expected responses for training. We'll use the traditional generator to create baseline responses, then manually review and improve them.

# COMMAND ----------

# Create traditional generator for baseline responses
traditional_generator = EnhancedMetadataGeneratorFactory.create_generator(
    config, use_dspy=False
)

print("🔄 Generating expected responses using traditional generator...")

training_examples = []

for i, raw_example in enumerate(training_examples_raw):
    print(f"Processing example {i+1}/{len(training_examples_raw)}: {raw_example['table_name']}")
    
    try:
        # Create prompt message format
        content = json.dumps({
            "table_name": raw_example["table_name"],
            "column_contents": raw_example["table_data"],
            "column_metadata": raw_example["column_metadata"]
        })
        
        messages = [{
            "role": "user", 
            "content": f"Content is here - {content} and abbreviations are here - {json.dumps(raw_example['abbreviations'])}"
        }]
        
        # Generate response using traditional generator  
        response = traditional_generator.predict_chat_response(messages)
        
        # Convert response to expected format
        expected_response = {
            "table": response.table,
            "columns": response.columns, 
            "column_contents": response.column_contents
        }
        
        # Create training example
        training_example = DSPyCommentGeneratorFactory.create_training_example(
            table_name=raw_example["table_name"],
            table_data=raw_example["table_data"],
            column_metadata=raw_example["column_metadata"], 
            expected_response=json.dumps(expected_response),
            abbreviations=raw_example["abbreviations"]
        )
        
        training_examples.append(training_example)
        print(f"  ✅ Generated response with {len(expected_response['column_contents'])} column descriptions")
        
    except Exception as e:
        print(f"  ⚠️ Error generating response: {e}")
        continue

print(f"\n✅ Created {len(training_examples)} complete training examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Review and Improve Training Examples
# MAGIC
# MAGIC Let's review the generated responses and improve them if needed.

# COMMAND ----------

# Display training examples for review
for i, example in enumerate(training_examples):
    print(f"\n{'='*60}")
    print(f"TRAINING EXAMPLE {i+1}")
    print(f"{'='*60}")
    print(f"Table: {example['table_name']}")
    print(f"Columns: {example['table_data']['columns']}")
    
    # Parse expected response
    expected = json.loads(example['expected_response'])
    print(f"\nTable Description:")
    print(f"  {expected['table']}")
    
    print(f"\nColumn Descriptions:")
    for j, (col, desc) in enumerate(zip(expected['columns'], expected['column_contents'])):
        print(f"  {j+1}. {col}: {desc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train DSPy Model
# MAGIC
# MAGIC Now let's train the DSPy model using our training examples.

# COMMAND ----------

# Create DSPy model
print("🔄 Creating DSPy model...")
dspy_model = DSPyCommentGeneratorFactory.create_generator(config)

# Split training data (use 80% for training, 20% for validation)
train_size = int(0.8 * len(training_examples))
train_examples = training_examples[:train_size]
val_examples = training_examples[train_size:] if len(training_examples) > 1 else None

print(f"📊 Training with {len(train_examples)} examples")
if val_examples:
    print(f"📊 Validating with {len(val_examples)} examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run DSPy Optimization
# MAGIC
# MAGIC This is where the magic happens! DSPy will optimize the prompts based on our training data.

# COMMAND ----------

print("🚀 Starting DSPy optimization...")
print("⏱️ This may take 5-15 minutes depending on the number of examples and API response times...")

try:
    # Run optimization
    optimized_module = dspy_model.optimize_prompts(
        train_examples=train_examples,
        validation_examples=val_examples,
        num_threads=2  # Adjust based on your API rate limits
    )
    
    print("✅ DSPy optimization completed successfully!")
    
    # Save optimized prompts
    model_path = "/tmp/dspy_optimized_comment_model.json"
    dspy_model.save_optimized_prompts(model_path)
    print(f"💾 Optimized model saved to: {model_path}")
    
except Exception as e:
    print(f"❌ Optimization failed: {e}")
    print("This is often due to API rate limits or connectivity issues.")
    print("Try reducing num_threads or checking your OpenAI API key.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Optimized Model
# MAGIC
# MAGIC Let's test our optimized model on new data and compare it with the traditional approach.

# COMMAND ----------

# Create test data (different from training data)
test_example = {
    "table_name": "hr.employees.employee_details",
    "table_data": {
        "index": [0, 1], 
        "columns": ["employee_id", "department", "salary", "hire_date", "manager_id"],
        "data": [
            ["EMP001", "Engineering", 85000, "2022-05-15", "MGR001"],
            ["EMP002", "Marketing", 65000, "2023-01-10", "MGR002"]
        ]
    },
    "column_metadata": {
        "employee_id": {"col_name": "employee_id", "data_type": "string", "num_nulls": "0", "distinct_count": "500", "avg_col_len": "6", "max_col_len": "6"},
        "department": {"col_name": "department", "data_type": "string", "num_nulls": "0", "distinct_count": "12", "avg_col_len": "10", "max_col_len": "15"}, 
        "salary": {"col_name": "salary", "data_type": "integer", "num_nulls": "0", "distinct_count": "300"},
        "hire_date": {"col_name": "hire_date", "data_type": "date", "num_nulls": "0", "distinct_count": "400"},
        "manager_id": {"col_name": "manager_id", "data_type": "string", "num_nulls": "5", "distinct_count": "50", "avg_col_len": "6", "max_col_len": "6"}
    },
    "abbreviations": config.acro_content
}

# Create test message
test_content = json.dumps({
    "table_name": test_example["table_name"],
    "column_contents": test_example["table_data"], 
    "column_metadata": test_example["column_metadata"]
})

test_messages = [{
    "role": "user",
    "content": f"Content is here - {test_content} and abbreviations are here - {json.dumps(test_example['abbreviations'])}"
}]

print("🧪 Testing both traditional and DSPy models on new data...")

# COMMAND ----------

# Test traditional model
print("\n" + "="*50)
print("TRADITIONAL MODEL RESPONSE")
print("="*50)

try:
    traditional_response = traditional_generator.predict_chat_response(test_messages)
    print(f"Table: {traditional_response.table}")
    print(f"\nColumns: {traditional_response.columns}")
    print(f"\nColumn Descriptions:")
    for i, (col, desc) in enumerate(zip(traditional_response.columns, traditional_response.column_contents)):
        print(f"  {i+1}. {col}: {desc}")
except Exception as e:
    print(f"❌ Traditional model error: {e}")
    traditional_response = None

# COMMAND ----------

# Test DSPy model
print("\n" + "="*50)
print("DSPY OPTIMIZED MODEL RESPONSE") 
print("="*50)

try:
    dspy_response = dspy_model.predict_chat_response(test_messages)
    print(f"Table: {dspy_response.table}")
    print(f"\nColumns: {dspy_response.columns}")
    print(f"\nColumn Descriptions:")
    for i, (col, desc) in enumerate(zip(dspy_response.columns, dspy_response.column_contents)):
        print(f"  {i+1}. {col}: {desc}")
except Exception as e:
    print(f"❌ DSPy model error: {e}")
    dspy_response = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Compare Performance
# MAGIC
# MAGIC Let's do a detailed comparison between the two approaches.

# COMMAND ----------

if traditional_response and dspy_response:
    print("\n" + "="*60)
    print("DETAILED COMPARISON")
    print("="*60)
    
    # Use migration helper to compare
    comparison = DSPyMigrationHelper.compare_responses(
        traditional_response, 
        dspy_response,
        metrics=["length", "completeness", "structure"]
    )
    
    print("📊 Comparison Results:")
    print(f"  Traditional response length: {comparison['metrics']['length']['traditional']} chars")
    print(f"  DSPy response length: {comparison['metrics']['length']['dspy']} chars")
    print(f"  Length difference: {comparison['metrics']['length']['difference']} chars")
    
    print(f"\n📋 Structure Check:")
    struct_metrics = comparison['metrics']['structure']
    print(f"  Both have table descriptions: {struct_metrics['traditional_has_table'] and struct_metrics['dspy_has_table']}")
    print(f"  Both have column lists: {struct_metrics['traditional_has_columns'] and struct_metrics['dspy_has_columns']}")
    print(f"  Both have column contents: {struct_metrics['traditional_has_contents'] and struct_metrics['dspy_has_contents']}")
    
    print(f"\n✅ Completeness Check:")
    trad_complete = comparison['metrics']['completeness']['traditional']
    dspy_complete = comparison['metrics']['completeness']['dspy']
    
    print(f"  Traditional - Column count match: {trad_complete.get('column_count_match', 'N/A')}")
    print(f"  DSPy - Column count match: {dspy_complete.get('column_count_match', 'N/A')}")

else:
    print("⚠️ Cannot compare - one or both models failed to generate responses")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save and Load Optimized Model
# MAGIC
# MAGIC Learn how to save your optimized model and load it for future use.

# COMMAND ----------

# Save the optimized model to DBFS for persistence
dbfs_model_path = "/dbfs/tmp/dspy_comment_generator_optimized.json"

try:
    dspy_model.save_optimized_prompts(dbfs_model_path)
    print(f"💾 Model saved to DBFS: {dbfs_model_path}")
    
    # Verify the file exists
    import os
    if os.path.exists(dbfs_model_path):
        file_size = os.path.getsize(dbfs_model_path)
        print(f"✅ File confirmed - Size: {file_size} bytes")
    else:
        print("⚠️ File not found after saving")
        
except Exception as e:
    print(f"❌ Error saving model: {e}")

# COMMAND ----------

# Demonstrate loading the saved model
print("🔄 Loading saved model...")

try:
    # Create a new DSPy model instance
    loaded_model = DSPyCommentGeneratorFactory.create_generator(config)
    
    # Load the optimized prompts
    loaded_model.load_optimized_prompts(dbfs_model_path)
    
    print("✅ Model loaded successfully!")
    
    # Test the loaded model
    print("\n🧪 Testing loaded model...")
    loaded_response = loaded_model.predict_chat_response(test_messages)
    
    print(f"Loaded model table description: {loaded_response.table[:100]}...")
    print(f"Number of column descriptions: {len(loaded_response.column_contents)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Integration with Existing Workflow
# MAGIC
# MAGIC Show how to integrate the optimized DSPy model into your existing dbxmetagen workflow.

# COMMAND ----------

# Example of how to use the optimized model in production
print("🔧 Production Integration Example:")

def create_production_generator(use_optimized_dspy=True, model_path=None):
    """
    Create a production-ready generator with optional DSPy optimization.
    """
    if use_optimized_dspy and model_path and os.path.exists(model_path):
        # Use optimized DSPy model
        generator = DSPyCommentGeneratorFactory.create_generator(config)
        generator.load_optimized_prompts(model_path)
        print("✅ Using optimized DSPy model")
        return generator
    elif use_optimized_dspy:
        # Use non-optimized DSPy model
        generator = DSPyCommentGeneratorFactory.create_generator(config) 
        print("✅ Using standard DSPy model")
        return generator
    else:
        # Use traditional model
        generator = EnhancedMetadataGeneratorFactory.create_generator(config, use_dspy=False)
        print("✅ Using traditional model")
        return generator

# Create production generator
prod_generator = create_production_generator(
    use_optimized_dspy=True, 
    model_path=dbfs_model_path if os.path.exists(dbfs_model_path) else None
)

# Test with production generator
print("\n🚀 Production generator test:")
prod_response = prod_generator.predict_chat_response(test_messages)
print(f"Response generated with {len(prod_response.column_contents)} column descriptions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Next Steps and Best Practices
# MAGIC
# MAGIC ### Key Takeaways:
# MAGIC
# MAGIC 1. **Training Data Quality**: The quality of your training data directly impacts DSPy optimization results
# MAGIC 2. **Iterative Improvement**: Run multiple optimization cycles with different examples to improve performance  
# MAGIC 3. **A/B Testing**: Compare DSPy vs traditional approaches on your specific data
# MAGIC 4. **Model Versioning**: Save different optimized versions and track performance
# MAGIC
# MAGIC ### Recommended Workflow:
# MAGIC
# MAGIC 1. **Start Small**: Begin with 5-10 high-quality training examples
# MAGIC 2. **Gradual Expansion**: Add more examples and re-train as you gather data
# MAGIC 3. **Performance Monitoring**: Track response quality over time
# MAGIC 4. **Domain-Specific Training**: Create separate models for different data domains (finance, HR, etc.)
# MAGIC
# MAGIC ### Production Deployment:
# MAGIC
# MAGIC ```python
# MAGIC # Use environment variables to control DSPy usage
# MAGIC import os
# MAGIC 
# MAGIC generator = EnhancedMetadataGeneratorFactory.create_generator_with_env_config(config)
# MAGIC 
# MAGIC # Set these in your Databricks job/cluster environment:
# MAGIC # DBXMETAGEN_USE_DSPY=true
# MAGIC # DBXMETAGEN_DSPY_PATTERNS=finance.,sales.,marketing.
# MAGIC # DBXMETAGEN_DSPY_FALLBACK=true
# MAGIC ```
# MAGIC
# MAGIC ### Troubleshooting:
# MAGIC
# MAGIC - **API Rate Limits**: Reduce `num_threads` in optimization
# MAGIC - **Memory Issues**: Process training examples in smaller batches  
# MAGIC - **Quality Issues**: Review and improve your training examples
# MAGIC - **Performance**: Use `gpt-4o-mini` for training, `gpt-4` for production

# COMMAND ----------

print("🎉 DSPy Training Quickstart Complete!")
print("\nYou have successfully:")
print("✅ Set up DSPy with dbxmetagen")  
print("✅ Created training data from Databricks tables")
print("✅ Trained and optimized DSPy prompts")
print("✅ Compared traditional vs DSPy performance")
print("✅ Saved and loaded optimized models")
print("✅ Learned production integration patterns")

print(f"\n📁 Your optimized model is saved at: {dbfs_model_path}")
print("🚀 You're ready to integrate DSPy into your metadata generation workflow!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Advanced Features
# MAGIC
# MAGIC ### Batch Processing Multiple Tables
# MAGIC
# MAGIC ```python
# MAGIC # Process multiple tables in batch
# MAGIC tables_to_process = [
# MAGIC     "catalog1.schema1.table1",
# MAGIC     "catalog1.schema1.table2", 
# MAGIC     "catalog2.schema2.table3"
# MAGIC ]
# MAGIC 
# MAGIC for table_name in tables_to_process:
# MAGIC     # Create generator (will use optimized DSPy if available)
# MAGIC     generator = create_production_generator(use_optimized_dspy=True, model_path=dbfs_model_path)
# MAGIC     
# MAGIC     # Process table and generate metadata
# MAGIC     # ... your processing logic here
# MAGIC ```
# MAGIC
# MAGIC ### Custom Evaluation Metrics
# MAGIC
# MAGIC ```python
# MAGIC def evaluate_description_quality(response, ground_truth=None):
# MAGIC     """Custom evaluation for description quality."""
# MAGIC     score = 0
# MAGIC     
# MAGIC     # Check description length (not too short, not too long)
# MAGIC     avg_length = sum(len(desc) for desc in response.column_contents) / len(response.column_contents)
# MAGIC     if 30 <= avg_length <= 200:
# MAGIC         score += 1
# MAGIC     
# MAGIC     # Check for domain-specific terms
# MAGIC     combined_text = " ".join([response.table] + response.column_contents)
# MAGIC     domain_terms = ["identifier", "foreign key", "primary key", "timestamp"]
# MAGIC     if any(term in combined_text.lower() for term in domain_terms):
# MAGIC         score += 1
# MAGIC         
# MAGIC     return score / 2  # Normalize to 0-1
# MAGIC ```
