# Databricks notebook source
# MAGIC %md
# MAGIC # NER and PII/PHI Detection and Redaction
# MAGIC
# MAGIC This notebook implements:
# MAGIC - NER using GLiNER-biomed model for unstructured text
# MAGIC - PII/PHI detection and redaction
# MAGIC - MLflow pyfunc model registration to Unity Catalog
# MAGIC - Efficient inference with pandas UDF iterator pattern
# MAGIC - Synthetic test data generation with ground truth

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Installation

# COMMAND ----------

# MAGIC %pip install -r ../requirements_ner.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports

# COMMAND ----------

import os
import sys
import json
import pandas as pd
import re
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass, asdict
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf, avg, sum as spark_sum
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine
import logging

sys.path.append("../")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Widgets
dbutils.widgets.text("catalog_name", "dbxmetagen", "Unity Catalog")
dbutils.widgets.text("schema_name", "default", "Schema")
dbutils.widgets.text("source_table", "ner_test_data", "Source Table")
dbutils.widgets.text("results_table", "ner_results", "Results Table")
dbutils.widgets.text("model_name", "ner_pii_detector", "Model Name")
dbutils.widgets.text(
    "hf_cache_dir", "/Volumes/dbxmetagen/default/models/hf_cache", "HF Cache Dir"
)
dbutils.widgets.dropdown("generate_data", "true", ["true", "false"], "Generate Data")

# Get values
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
source_table = dbutils.widgets.get("source_table")
results_table = dbutils.widgets.get("results_table")
model_name = dbutils.widgets.get("model_name")
hf_cache_dir = dbutils.widgets.get("hf_cache_dir")
generate_data = dbutils.widgets.get("generate_data") == "true"

# Full names
full_source_table = f"{catalog_name}.{schema_name}.{source_table}"
full_results_table = f"{catalog_name}.{schema_name}.{results_table}"
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

print(f"Source: {full_source_table}")
print(f"Results: {full_results_table}")
print(f"Model: {full_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Data Classes

# COMMAND ----------


@dataclass
class NERConfig:
    """NER model configuration."""

    model_name: str = "Ihor/gliner-biomed-base-v1.0"
    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.5
    batch_size: int = 16
    pii_labels: List[str] = None
    phi_labels: List[str] = None

    def __post_init__(self):
        if self.pii_labels is None:
            self.pii_labels = [
                "person",
                "email",
                "phone number",
                "Social Security Number",
                "driver licence",
                "passport number",
                "credit card",
                "full address",
                "personally identifiable information",
                "date of birth",
            ]
        if self.phi_labels is None:
            self.phi_labels = [
                "patient",
                "medical record number",
                "diagnosis",
                "medication",
                "doctor",
                "hospital",
                "medical condition",
                "treatment",
                "lab test",
                "lab test value",
                "dosage",
                "drug",
                "prescription",
            ]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic Data Generation

# COMMAND ----------


def generate_test_data(num_rows: int = 20) -> pd.DataFrame:
    """Generate synthetic test data with ground truth."""

    clean_texts = [
        "The weather today is sunny and warm with clear blue skies.",
        "Technology continues to evolve rapidly in the modern world.",
        "Books are a wonderful source of knowledge and entertainment.",
        "Music has the power to evoke strong emotions and memories.",
        "Exercise is important for maintaining physical and mental health.",
    ]

    pii_texts = [
        "John Smith's email is john.smith@email.com and his phone is (555) 123-4567.",
        "Sarah Johnson lives at 123 Main Street, Anytown, NY 12345. Her SSN is 123-45-6789.",
        "Mike Davis has credit card 4532 1234 5678 9012 expires 12/25.",
        "Lisa Brown's driver license DL1234567 from California.",
        "Robert Wilson DOB 03/15/1985 passport A12345678.",
    ]

    phi_texts = [
        "Patient Mary Thompson MRN 12345 diagnosed with diabetes. Prescribed Metformin 500mg twice daily.",
        "Dr. Anderson examined patient 67890 at City Hospital. Lab HbA1c 8.2%.",
        "John Doe hypertension. Medication: Lisinopril 10mg daily. Next with Dr. Smith.",
        "Patient Jennifer Lee DOB 01/20/1975 pneumonia. Chest X-ray consolidation.",
        "Record 98765: coronary artery disease. Prescribed Atorvastatin 40mg nightly.",
    ]

    # Simple ground truth patterns
    pii_patterns = {
        r"\b[A-Za-z]+ [A-Za-z]+\b": "person",
        r"\b\w+@\w+\.\w+\b": "email",
        r"\(\d{3}\) \d{3}-\d{4}": "phone number",
        r"\b\d{3}-\d{2}-\d{4}\b": "Social Security Number",
    }

    phi_patterns = {
        r"\bPatient [A-Za-z]+ [A-Za-z]+\b": "patient",
        r"\bMRN \d+\b": "medical record number",
        r"\bDr\. [A-Za-z]+\b": "doctor",
        r"\b\w+mg\b": "dosage",
    }

    def extract_entities(text: str, patterns: Dict[str, str]) -> List[Dict[str, Any]]:
        entities = []
        for pattern, label in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(
                    {
                        "text": match.group(),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return entities

    def redact_text(text: str, entities: List[Dict[str, Any]]) -> str:
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted = text
        for entity in entities_sorted:
            placeholder = f"[{entity['label'].upper()}]"
            redacted = (
                redacted[: entity["start"]] + placeholder + redacted[entity["end"] :]
            )
        return redacted

    data = []
    for i in range(num_rows):
        clean_text = clean_texts[i % len(clean_texts)]
        pii_text = pii_texts[i % len(pii_texts)]
        phi_text = phi_texts[i % len(phi_texts)]

        pii_entities = extract_entities(pii_text, pii_patterns)
        phi_entities = extract_entities(phi_text, phi_patterns)

        data.append(
            {
                "id": i + 1,
                "clean_text": clean_text,
                "pii_text": pii_text,
                "phi_text": phi_text,
                "pii_ground_truth_entities": json.dumps(pii_entities),
                "phi_ground_truth_entities": json.dumps(phi_entities),
                "pii_redacted_ground_truth": redact_text(pii_text, pii_entities),
                "phi_redacted_ground_truth": redact_text(phi_text, phi_entities),
            }
        )

    return pd.DataFrame(data)


# COMMAND ----------

# MAGIC %md
# MAGIC ## NER Model Implementation

# COMMAND ----------


class GLiNERNERModel(mlflow.pyfunc.PythonModel):
    """GLiNER NER model for MLflow."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.model = None
        self.analyzer = None

    def load_context(self, context):
        """Load model and dependencies."""
        # Set HF cache
        os.environ["HF_HOME"] = self.config["cache_dir"]
        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]

        # Load models
        self.model = GLiNER.from_pretrained(
            self.config["model_name"], cache_dir=self.config["cache_dir"]
        )
        self.analyzer = AnalyzerEngine()
        logger.info("Model loaded successfully")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict entities."""
        results = []

        for _, input_row in model_input.iterrows():
            text = input_row.get("text", "")
            text_type = input_row.get("text_type", "general")

            # Select labels
            if text_type == "pii":
                labels = self.config["pii_labels"]
            elif text_type == "phi":
                labels = self.config["phi_labels"]
            else:
                labels = self.config["pii_labels"] + self.config["phi_labels"]

            # Get predictions
            gliner_entities = self.model.predict_entities(
                text, labels, threshold=self.config["threshold"]
            )
            presidio_entities = self.analyzer.analyze(text=text, language="en")

            # Combine and deduplicate
            all_entities = []

            # Add GLiNER entities
            for entity in gliner_entities:
                all_entities.append(
                    {
                        "text": entity["text"],
                        "label": entity["label"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "score": entity["score"],
                        "source": "gliner",
                    }
                )

            # Add Presidio entities
            for entity in presidio_entities:
                all_entities.append(
                    {
                        "text": text[entity.start : entity.end],
                        "label": entity.entity_type,
                        "start": entity.start,
                        "end": entity.end,
                        "score": entity.score,
                        "source": "presidio",
                    }
                )

            # Deduplicate overlapping entities
            unique_entities = self._deduplicate_entities(all_entities)

            # Redact text
            redacted_text = self._redact_text(text, unique_entities)

            results.append(
                {
                    "text": text,
                    "entities": json.dumps(unique_entities),
                    "redacted_text": redacted_text,
                    "entity_count": len(unique_entities),
                }
            )

        return pd.DataFrame(results)

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove overlapping entities."""
        entities_sorted = sorted(entities, key=lambda x: (x["start"], -x["score"]))
        unique = []

        for entity in entities_sorted:
            overlaps = any(
                not (
                    entity["end"] <= existing["start"]
                    or existing["end"] <= entity["start"]
                )
                for existing in unique
            )
            if not overlaps:
                unique.append(entity)

        return unique

    def _redact_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Redact entities in text."""
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted = text

        for entity in entities_sorted:
            placeholder = f"[{entity['label'].upper()}]"
            redacted = (
                redacted[: entity["start"]] + placeholder + redacted[entity["end"] :]
            )

        return redacted


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Registration

# COMMAND ----------


def train_and_register_model(ner_config: NERConfig, model_name_full: str) -> str:
    """Train and register NER model to Unity Catalog."""

    # Convert config to dict for serialization
    config_dict = asdict(ner_config)
    ner_model = GLiNERNERModel(config_dict)

    # Sample data for signature
    sample_input = pd.DataFrame(
        {"text": ["John Smith works at Hospital."], "text_type": ["pii"]}
    )

    # Load and test model
    ner_model.load_context(None)
    sample_output = ner_model.predict(None, sample_input)

    # Create signature
    signature = infer_signature(sample_input, sample_output)

    # Step 1: Log model
    with mlflow.start_run():
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="ner_model",
            python_model=ner_model,
            signature=signature,
            pip_requirements=[
                "gliner==0.2.5",
                "transformers==4.44.0",
                "torch==2.4.0",
                "presidio-analyzer==2.2.358",
                "presidio-anonymizer==2.2.358",
            ],
            input_example=sample_input,
            metadata={"model_type": "ner", "base_model": ner_config.model_name},
        )

        mlflow.log_params(
            {
                "base_model": ner_config.model_name,
                "threshold": ner_config.threshold,
                "batch_size": ner_config.batch_size,
            }
        )

    # Step 2: Register to Unity Catalog
    # Note: mlflow.set_registry_uri("databricks-uc") should be set before calling this function
    registered_model_info = mlflow.register_model(
        model_uri=logged_model_info.model_uri, name=model_name_full
    )

    logger.info(
        "Model registered to Unity Catalog: %s version %s",
        model_name_full,
        registered_model_info.version,
    )
    return f"models:/{model_name_full}/{registered_model_info.version}"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Efficient Batch Processing

# COMMAND ----------


def create_ner_udf(model_uri_path: str):
    """Create pandas UDF with iterator pattern for efficient processing."""

    @pandas_udf("struct<entities:string,redacted_text:string,entity_count:int>")
    def process_text_batch(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        # Load model once per worker
        model = mlflow.pyfunc.load_model(model_uri_path)

        for text_series in iterator:
            if len(text_series) == 0:
                yield pd.DataFrame(
                    {"entities": [], "redacted_text": [], "entity_count": []}
                )
                continue

            # Create batch input
            input_df = pd.DataFrame(
                {
                    "text": text_series.values,
                    "text_type": ["general"] * len(text_series),
                }
            )

            # Process batch
            results = model.predict(input_df)

            yield pd.DataFrame(
                {
                    "entities": results["entities"],
                    "redacted_text": results["redacted_text"],
                    "entity_count": results["entity_count"],
                }
            )

    return process_text_batch


def process_dataframe_ner(
    df: DataFrame, text_column: str, model_uri_path: str
) -> DataFrame:
    """Process DataFrame with NER using efficient UDF."""
    ner_udf = create_ner_udf(model_uri_path)

    return (
        df.withColumn("ner_results", ner_udf(col(text_column)))
        .select(
            "*",
            col("ner_results.entities").alias("detected_entities"),
            col("ner_results.redacted_text").alias("redacted_text"),
            col("ner_results.entity_count").alias("entity_count"),
        )
        .drop("ner_results")
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Setup

# COMMAND ----------

print("🔧 Setting up Unity Catalog resources...")

# Ensure schema exists in Unity Catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
print(f"✅ Schema '{catalog_name}.{schema_name}' ready")

# Create Unity Catalog Volume for HuggingFace cache if it doesn't exist
volume_path = f"{catalog_name}.{schema_name}.hf_cache_ner"
try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_path}")
    print(f"✅ Unity Catalog Volume '{volume_path}' ready")

    # Create the cache subdirectory
    import os

    cache_dir = hf_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"✅ HuggingFace cache directory ready: {cache_dir}")

except Exception as e:
    print(f"⚠️ Volume setup issue: {e}")
    print("💡 Ensure you have CREATE VOLUME permissions on the schema")
    # Fall back to a local temp directory if volume creation fails
    import tempfile

    cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
    print(f"📁 Using temporary cache directory: {cache_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation

# COMMAND ----------

if generate_data:
    print("Generating synthetic test data...")

    # Generate data
    test_data = generate_test_data(20)
    spark_df = spark.createDataFrame(test_data)

    # Save to Unity Catalog table
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        full_source_table
    )

    print(f"✅ Data saved to Unity Catalog table: {full_source_table}")
    display(spark.table(full_source_table).limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Registration

# COMMAND ----------

# Set up Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

# Set up configuration with the cache directory we set up
config = NERConfig(
    model_name="Ihor/gliner-biomed-base-v1.0",
    cache_dir=cache_dir,  # Use the cache_dir variable set in Unity Catalog setup
    threshold=0.5,
    batch_size=16,
)

# Train and register
print("Training and registering model to Unity Catalog...")
print(f"Target UC Model: {full_model_name}")
model_uri = train_and_register_model(config, full_model_name)
print(f"✅ Model registered to Unity Catalog: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Processing

# COMMAND ----------

# Load source data
source_df = spark.table(full_source_table)

# Process PII and PHI columns
print("Processing PII text...")
pii_results = process_dataframe_ner(source_df, "pii_text", model_uri)

print("Processing PHI text...")
phi_results = process_dataframe_ner(source_df, "phi_text", model_uri)

# Combine results
final_results = source_df.join(
    pii_results.select(
        "id",
        col("detected_entities").alias("pii_detected_entities"),
        col("redacted_text").alias("pii_redacted_text"),
        col("entity_count").alias("pii_entity_count"),
    ),
    "id",
).join(
    phi_results.select(
        "id",
        col("detected_entities").alias("phi_detected_entities"),
        col("redacted_text").alias("phi_redacted_text"),
        col("entity_count").alias("phi_entity_count"),
    ),
    "id",
)

# Save results to Unity Catalog
final_results.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    full_results_table
)

print(f"✅ Results saved to Unity Catalog table: {full_results_table}")
print(f"📊 Processed {final_results.count()} rows")
display(final_results.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation

# COMMAND ----------


def calculate_metrics(df: DataFrame) -> Dict[str, Any]:
    """Calculate processing metrics."""
    total_rows = df.count()

    pii_stats = df.agg(
        avg("pii_entity_count").alias("avg_pii"),
        spark_sum("pii_entity_count").alias("sum_pii"),
    ).collect()[0]

    phi_stats = df.agg(
        avg("phi_entity_count").alias("avg_phi"),
        spark_sum("phi_entity_count").alias("sum_phi"),
    ).collect()[0]

    return {
        "total_rows": total_rows,
        "avg_pii_entities": pii_stats["avg_pii"],
        "total_pii_entities": int(pii_stats["sum_pii"] or 0),
        "avg_phi_entities": phi_stats["avg_phi"],
        "total_phi_entities": int(phi_stats["sum_phi"] or 0),
    }


# Calculate metrics
metrics = calculate_metrics(final_results)
print("📊 Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Show sample comparisons
sample_df = final_results.limit(2).toPandas()
print("\n📝 Sample Results:")
for _, row in sample_df.iterrows():
    print(f"\nID {row['id']}:")
    print(f"  PII Text: {row['pii_text'][:60]}...")
    print(f"  PII Detected: {len(json.loads(row['pii_detected_entities']))} entities")
    print(f"  PHI Text: {row['phi_text'][:60]}...")
    print(f"  PHI Detected: {len(json.loads(row['phi_detected_entities']))} entities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("🎉 NER and PII/PHI Detection Complete!")
print("\n📋 Unity Catalog Summary:")
print(f"  • UC Model: {full_model_name}")
print(f"  • UC Source Table: {full_source_table}")
print(f"  • UC Results Table: {full_results_table}")
print(f"  • Model Version: {model_uri.split('/')[-1]}")
print(
    f"  • Total entities detected: {metrics['total_pii_entities'] + metrics['total_phi_entities']}"
)

print("\n🔍 Unity Catalog Verification:")
try:
    # Verify model exists by attempting to load it
    mlflow.pyfunc.load_model(f"models:/{full_model_name}/latest")
    print(f"  • ✅ Model exists: {full_model_name}")
except Exception as e:
    print(f"  • ❌ Model load failed: {str(e)}")

print(f"  • ✅ Source table exists: {spark.catalog.tableExists(full_source_table)}")
print(f"  • ✅ Results table exists: {spark.catalog.tableExists(full_results_table)}")

# Show table row counts
source_count = spark.table(full_source_table).count()
results_count = spark.table(full_results_table).count()
print(f"  • 📊 Source table rows: {source_count}")
print(f"  • 📊 Results table rows: {results_count}")

print("\n💡 Usage with Unity Catalog:")
print(
    f"""
# Load registered Unity Catalog model (specific version)
model = mlflow.pyfunc.load_model("{model_uri}")

# Or use latest version
model = mlflow.pyfunc.load_model("models:/{full_model_name}/latest")

# Process new data
new_data = pd.DataFrame({{'text': ['Your sensitive text here'], 'text_type': ['pii']}})
results = model.predict(new_data)
print(results[['entities', 'redacted_text']].head())

# Query Unity Catalog tables
display(spark.sql("SELECT * FROM {full_results_table} LIMIT 10"))

# Example batch processing
df_to_process = spark.table("{full_source_table}")
processed = process_dataframe_ner(df_to_process, "pii_text", "models:/{full_model_name}/latest")
display(processed.select("id", "detected_entities", "redacted_text").limit(5))
"""
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Permissions Check

# COMMAND ----------

# Verify Unity Catalog permissions
print("🔐 Unity Catalog Permissions Verification:")

try:
    # Check catalog access
    catalogs = [row.catalogName for row in spark.sql("SHOW CATALOGS").collect()]
    if catalog_name in catalogs:
        print(f"  • ✅ Catalog '{catalog_name}' accessible")
    else:
        print(
            f"  • ❌ Catalog '{catalog_name}' not found. Available: {', '.join(catalogs[:5])}"
        )

    # Check schema access
    schemas = [
        row.databaseName
        for row in spark.sql(f"SHOW SCHEMAS IN {catalog_name}").collect()
    ]
    if schema_name in schemas:
        print(f"  • ✅ Schema '{schema_name}' accessible")
    else:
        print(
            f"  • ❌ Schema '{schema_name}' not found. Available: {', '.join(schemas[:5])}"
        )

    # Test model registry access
    current_user = spark.sql("SELECT current_user()").collect()[0][0]
    print(f"  • 👤 Current user: {current_user}")
    print("  • ✅ Unity Catalog model registry accessible")

except Exception as e:
    print(f"  • ❌ Unity Catalog access error: {str(e)}")
    print(
        "  • 💡 Ensure you have appropriate permissions on catalog, schema, and model registry"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog & Performance Tips
# MAGIC
# MAGIC **Unity Catalog Best Practices:**
# MAGIC - Use three-part names (catalog.schema.table/model) consistently
# MAGIC - Set proper permissions on catalog, schema, and objects
# MAGIC - Use `mlflow.set_registry_uri("databricks-uc")` before model operations
# MAGIC - Verify model registration with `mlflow.pyfunc.load_model()`
# MAGIC - Use Delta tables with `option("overwriteSchema", "true")` for schema evolution
# MAGIC - **Use Unity Catalog Volumes instead of DBFS** for file storage (`/Volumes/catalog/schema/volume`)
# MAGIC - Create volumes with `CREATE VOLUME IF NOT EXISTS` for model caches and artifacts
# MAGIC
# MAGIC **For Better Performance:**
# MAGIC - Adjust batch_size based on cluster memory (16-32 recommended)
# MAGIC - Use GPU clusters for faster inference (ML Runtime recommended)
# MAGIC - Enable model caching for repeated runs (`cache_dir` configuration)
# MAGIC - Tune threshold for precision/recall balance (0.3-0.7 range)
# MAGIC - Use iterator pattern in pandas UDF to minimize memory usage
# MAGIC
# MAGIC **For Better Accuracy:**
# MAGIC - Customize PII/PHI labels for your specific domain
# MAGIC - Lower threshold catches more entities (higher recall)
# MAGIC - Combine GLiNER + Presidio engines for comprehensive coverage
# MAGIC - Add domain-specific post-processing rules and patterns
# MAGIC - Test with your actual data patterns for optimal results
