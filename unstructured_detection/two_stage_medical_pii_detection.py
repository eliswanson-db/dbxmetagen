# Databricks notebook source
# MAGIC %md
# MAGIC # Two-Stage Medical PII Detection
# MAGIC
# MAGIC This notebook implements a two-stage approach:
# MAGIC 1. **Stage 1**: BioBERT identifies medical information
# MAGIC 2. **Stage 2**: Configurable PII detection (DistilBERT/GLiNER/Presidio)
# MAGIC
# MAGIC Allows separate redaction of medical content vs PII based on privacy requirements.

# COMMAND ----------

# MAGIC %pip install transformers torch gliner presidio-analyzer presidio-anonymizer mlflow spacy packaging

# COMMAND ----------

import os
import sys
import json
import re
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine
import torch
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.append("../")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# Configuration widgets
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("catalog_name", "dbxmetagen", "Unity Catalog")
dbutils.widgets.text("schema_name", "default", "Schema")
dbutils.widgets.text("source_table", "medical_pii_test_data", "Source Table")
dbutils.widgets.text("results_table", "medical_pii_results", "Results Table")
dbutils.widgets.text("model_name", "medical_pii_detector", "Model Name")
dbutils.widgets.text(
    "hf_cache_dir", "/Volumes/dbxmetagen/default/models/hf_cache", "HF Cache Dir"
)
dbutils.widgets.dropdown("generate_data", "true", ["true", "false"], "Generate Data")
dbutils.widgets.dropdown("train", "false", ["true", "false"], "Train Model")
dbutils.widgets.dropdown(
    "pii_model", "distilbert", ["distilbert", "gliner", "presidio"], "PII Model"
)
dbutils.widgets.dropdown(
    "redaction_mode", "both", ["medical_only", "pii_only", "both"], "Redaction Mode"
)

env = dbutils.widgets.get("environment")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
source_table = dbutils.widgets.get("source_table")
results_table = dbutils.widgets.get("results_table")
model_name = dbutils.widgets.get("model_name")
hf_cache_dir = dbutils.widgets.get("hf_cache_dir")
generate_data = dbutils.widgets.get("generate_data") == "true"
train = dbutils.widgets.get("train") == "true"
pii_model = dbutils.widgets.get("pii_model")
redaction_mode = dbutils.widgets.get("redaction_mode")

full_source_table = f"{catalog_name}.{schema_name}.{source_table}"
full_results_table = f"{catalog_name}.{schema_name}.{results_table}"
cache_dir = hf_cache_dir

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Medical Content Detection (BioBERT)

# COMMAND ----------


@dataclass
class MedicalDetectorConfig:
    """Configuration for medical content detection."""

    model_name: str = "dmis-lab/biobert-v1.1"
    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.7
    batch_size: int = 16


class MedicalContentDetector(mlflow.pyfunc.PythonModel):
    """BioBERT model for detecting medical content."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.ner_pipeline = None

    def load_context(self, context):
        """Load BioBERT for medical content detection."""
        try:
            cache_dir = self.config["cache_dir"]
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            logger.info("Loading BioBERT for medical content detection")

            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"], cache_dir=cache_dir
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.config["model_name"], cache_dir=cache_dir
            )

            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1,
            )

            logger.info("Medical content detector loaded successfully")

        except Exception as e:
            logger.error("Failed to load medical detector: %s", str(e))
            raise RuntimeError(f"Medical detector loading failed: {str(e)}") from e

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Detect medical content in text."""
        results = []

        for _, input_row in model_input.iterrows():
            text = input_row.get("text", "")

            if len(text) > 5000:
                text = text[:5000]

            medical_entities = []
            has_medical_content = False

            try:
                biobert_entities = self.ner_pipeline(text)

                for entity in biobert_entities:
                    medical_label = self._map_to_medical_type(entity["entity_group"])

                    if medical_label and entity["score"] >= self.config["threshold"]:
                        has_medical_content = True
                        medical_entities.append(
                            {
                                "text": entity["word"],
                                "label": medical_label,
                                "start": int(entity["start"]),
                                "end": int(entity["end"]),
                                "score": float(entity["score"]),
                                "source": "biobert_medical",
                            }
                        )

            except Exception as e:
                logger.warning("Medical detection failed: %s", str(e))

            # Create medical-redacted text
            medical_redacted = self._redact_medical_content(text, medical_entities)

            results.append(
                {
                    "text": text,
                    "has_medical_content": has_medical_content,
                    "medical_entities": json.dumps(medical_entities),
                    "medical_redacted_text": medical_redacted,
                    "medical_entity_count": len(medical_entities),
                }
            )

        return pd.DataFrame(results)

    def _map_to_medical_type(self, biobert_label: str) -> str:
        """Map BioBERT labels to medical content types."""
        medical_mapping = {
            "DISEASE": "disease",
            "CHEMICAL": "medication",
            "GENE": "genetic_info",
            "PROTEIN": "protein",
            "CELLTYPE": "cell_type",
            "CELLLINE": "cell_line",
            "DNA": "genetic_info",
            "RNA": "genetic_info",
            "SPECIES": "species",
        }
        return medical_mapping.get(biobert_label.upper())

    def _redact_medical_content(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Redact medical content from text."""
        if not entities:
            return text

        entities = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted_text = text

        for entity in entities:
            redacted_text = (
                redacted_text[: entity["start"]]
                + f"[{entity['label'].upper().replace(' ', '_')}]"
                + redacted_text[entity["end"] :]
            )

        return redacted_text


# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: PII Detection Models

# COMMAND ----------


@dataclass
class PIIDetectorConfig:
    """Configuration for PII detection."""

    model_type: str = "distilbert"  # distilbert, gliner, presidio
    model_name: str = "dslim/distilbert-NER"
    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.8
    batch_size: int = 16


class PIIDetector(mlflow.pyfunc.PythonModel):
    """Configurable PII detector."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.detector = None

    def load_context(self, context):
        """Load the specified PII detection model."""
        try:
            cache_dir = self.config["cache_dir"]
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            model_type = self.config["model_type"]
            logger.info(f"Loading {model_type} PII detector")

            if model_type == "distilbert":
                self._load_distilbert()
            elif model_type == "gliner":
                self._load_gliner()
            elif model_type == "presidio":
                self._load_presidio()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            logger.info(f"{model_type} PII detector loaded successfully")

        except Exception as e:
            logger.error("Failed to load PII detector: %s", str(e))
            raise RuntimeError(f"PII detector loading failed: {str(e)}") from e

    def _load_distilbert(self):
        """Load DistilBERT NER model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], cache_dir=self.config["cache_dir"]
        )
        model = AutoModelForTokenClassification.from_pretrained(
            self.config["model_name"], cache_dir=self.config["cache_dir"]
        )
        self.detector = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

    def _load_gliner(self):
        """Load GLiNER model."""
        self.detector = GLiNER.from_pretrained(
            "Ihor/gliner-biomed-base-v1.0", cache_dir=self.config["cache_dir"]
        )

    def _load_presidio(self):
        """Load Presidio analyzer."""
        self.detector = AnalyzerEngine()

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Detect PII in text."""
        results = []

        for _, input_row in model_input.iterrows():
            text = input_row.get("text", "")

            if len(text) > 5000:
                text = text[:5000]

            pii_entities = []

            try:
                model_type = self.config["model_type"]

                if model_type == "distilbert":
                    pii_entities = self._detect_distilbert_pii(text)
                elif model_type == "gliner":
                    pii_entities = self._detect_gliner_pii(text)
                elif model_type == "presidio":
                    pii_entities = self._detect_presidio_pii(text)

            except Exception as e:
                logger.warning(f"{model_type} PII detection failed: {str(e)}")

            # Create PII-redacted text
            pii_redacted = self._redact_pii_content(text, pii_entities)

            results.append(
                {
                    "text": text,
                    "pii_entities": json.dumps(pii_entities),
                    "pii_redacted_text": pii_redacted,
                    "pii_entity_count": len(pii_entities),
                }
            )

        return pd.DataFrame(results)

    def _detect_distilbert_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using DistilBERT."""
        entities = []
        distilbert_entities = self.detector(text)

        for entity in distilbert_entities:
            pii_label = self._map_distilbert_to_pii(entity["entity_group"])
            if pii_label and entity["score"] >= self.config["threshold"]:
                entities.append(
                    {
                        "text": entity["word"],
                        "label": pii_label,
                        "start": int(entity["start"]),
                        "end": int(entity["end"]),
                        "score": float(entity["score"]),
                        "source": "distilbert_pii",
                    }
                )
        return entities

    def _detect_gliner_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using GLiNER."""
        entities = []
        pii_labels = [
            "person",
            "phone number",
            "email address",
            "social security number",
            "medical record number",
            "date of birth",
            "street address",
            "geographic identifier",
        ]

        gliner_entities = self.detector.predict_entities(
            text, pii_labels, threshold=self.config["threshold"]
        )

        for entity in gliner_entities:
            entities.append(
                {
                    "text": entity["text"],
                    "label": entity["label"],
                    "start": int(entity["start"]),
                    "end": int(entity["end"]),
                    "score": float(entity["score"]),
                    "source": "gliner_pii",
                }
            )
        return entities

    def _detect_presidio_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using Presidio."""
        entities = []
        presidio_entities = self.detector.analyze(text=text, language="en")

        for entity in presidio_entities:
            pii_label = self._map_presidio_to_pii(entity.entity_type)
            if pii_label and entity.score >= self.config["threshold"]:
                entities.append(
                    {
                        "text": text[entity.start : entity.end],
                        "label": pii_label,
                        "start": int(entity.start),
                        "end": int(entity.end),
                        "score": float(entity.score),
                        "source": "presidio_pii",
                    }
                )
        return entities

    def _map_distilbert_to_pii(self, label: str) -> str:
        """Map DistilBERT labels to PII types."""
        mapping = {
            "PER": "person",
            "LOC": "geographic identifier",
            "ORG": "organization",
            "MISC": "unique identifier",
        }
        return mapping.get(label.upper())

    def _map_presidio_to_pii(self, label: str) -> str:
        """Map Presidio labels to PII types."""
        mapping = {
            "PERSON": "person",
            "PHONE_NUMBER": "phone number",
            "EMAIL_ADDRESS": "email address",
            "US_SSN": "social security number",
            "DATE_TIME": "date",
            "LOCATION": "geographic identifier",
        }
        return mapping.get(label.upper())

    def _redact_pii_content(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Redact PII from text."""
        if not entities:
            return text

        entities = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted_text = text

        for entity in entities:
            redacted_text = (
                redacted_text[: entity["start"]]
                + f"[{entity['label'].upper().replace(' ', '_')}]"
                + redacted_text[entity["end"] :]
            )

        return redacted_text


# COMMAND ----------

# MAGIC %md
# MAGIC ## Two-Stage Detection Pipeline

# COMMAND ----------


class TwoStageDetector(mlflow.pyfunc.PythonModel):
    """Combined medical + PII detection pipeline."""

    def __init__(self, medical_config: Dict[str, Any], pii_config: Dict[str, Any]):
        self.medical_detector = MedicalContentDetector(medical_config)
        self.pii_detector = PIIDetector(pii_config)

    def load_context(self, context):
        """Load both detection models."""
        self.medical_detector.load_context(context)
        self.pii_detector.load_context(context)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Run two-stage detection."""
        # Stage 1: Medical content detection
        medical_results = self.medical_detector.predict(context, model_input)

        # Stage 2: PII detection
        pii_results = self.pii_detector.predict(context, model_input)

        # Combine results
        combined_results = []

        for i in range(len(model_input)):
            text = model_input.iloc[i]["text"]

            medical_entities = json.loads(medical_results.iloc[i]["medical_entities"])
            pii_entities = json.loads(pii_results.iloc[i]["pii_entities"])

            # Create different redaction modes
            medical_only_redacted = medical_results.iloc[i]["medical_redacted_text"]
            pii_only_redacted = pii_results.iloc[i]["pii_redacted_text"]
            both_redacted = self._redact_both(text, medical_entities, pii_entities)

            combined_results.append(
                {
                    "text": text,
                    "has_medical_content": medical_results.iloc[i][
                        "has_medical_content"
                    ],
                    "medical_entities": json.dumps(medical_entities),
                    "pii_entities": json.dumps(pii_entities),
                    "medical_only_redacted": medical_only_redacted,
                    "pii_only_redacted": pii_only_redacted,
                    "both_redacted": both_redacted,
                    "medical_entity_count": len(medical_entities),
                    "pii_entity_count": len(pii_entities),
                }
            )

        return pd.DataFrame(combined_results)

    def _redact_both(
        self, text: str, medical_entities: List[Dict], pii_entities: List[Dict]
    ) -> str:
        """Redact both medical and PII content."""
        all_entities = medical_entities + pii_entities

        # Remove overlaps
        all_entities = sorted(all_entities, key=lambda x: x["start"])
        unique_entities = []

        for entity in all_entities:
            overlap = False
            for existing in unique_entities:
                if (
                    entity["start"] < existing["end"]
                    and entity["end"] > existing["start"]
                ):
                    overlap = True
                    break
            if not overlap:
                unique_entities.append(entity)

        # Redact
        entities = sorted(unique_entities, key=lambda x: x["start"], reverse=True)
        redacted_text = text

        for entity in entities:
            redacted_text = (
                redacted_text[: entity["start"]]
                + f"[{entity['label'].upper().replace(' ', '_')}]"
                + redacted_text[entity["end"] :]
            )

        return redacted_text


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Data Generation

# COMMAND ----------


def generate_test_data(num_rows: int = 10) -> pd.DataFrame:
    """Generate test data with medical and PII content."""

    texts = [
        "Patient John Smith was treated for diabetes at City General Hospital. Contact: (555) 123-4567. DOB: 03/15/1975.",
        "Dr. Sarah Johnson prescribed metformin 500mg for glucose control. Patient lives at 123 Oak Street, Chicago, IL.",
        "Blood pressure readings show hypertension. Patient Maria Garcia can be reached at maria@email.com.",
        "Chest X-ray reveals pneumonia. Patient ID: MRN-12345. Emergency contact: (312) 555-9876.",
        "Cardiac enzyme levels are elevated indicating myocardial infarction. Patient address: 456 Elm Drive, Springfield.",
        "Neurological examination shows normal reflexes and cognitive function. No abnormal findings noted.",
        "Laboratory results indicate elevated white blood cell count consistent with bacterial infection.",
        "Physical therapy session focused on range of motion exercises following knee replacement surgery.",
        "Patient Robert Wilson (SSN: 123-45-6789) scheduled for follow-up appointment on April 15, 2024.",
        "Prescription for amoxicillin 875mg twice daily to treat streptococcal pharyngitis.",
    ]

    data = []
    for i in range(num_rows):
        text = texts[i % len(texts)]
        data.append({"text": text, "text_type": "medical"})

    return pd.DataFrame(data)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Registration

# COMMAND ----------


def train_and_register_two_stage_model(
    medical_config: MedicalDetectorConfig,
    pii_config: PIIDetectorConfig,
    model_name_full: str,
) -> str:
    """Train and register the two-stage detection model."""

    medical_config_dict = asdict(medical_config)
    pii_config_dict = asdict(pii_config)

    two_stage_model = TwoStageDetector(medical_config_dict, pii_config_dict)

    sample_input = pd.DataFrame(
        {
            "text": ["Patient John Smith has diabetes. Contact: (555) 123-4567."],
            "text_type": ["medical"],
        }
    )

    two_stage_model.load_context(None)
    sample_output = two_stage_model.predict(None, sample_input)

    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run():
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="two_stage_detector",
            python_model=two_stage_model,
            signature=signature,
            pip_requirements=[
                "numpy>=1.21.5,<2.0",
                "pandas>=1.5.0,<2.1.0",
                "transformers==4.44.0",
                "torch==2.4.0",
                "gliner==0.2.5",
                "presidio-analyzer==2.2.358",
                "packaging>=21.0",
            ],
            input_example=sample_input,
            metadata={
                "model_type": "two_stage_detector",
                "medical_model": medical_config.model_name,
                "pii_model": pii_config.model_type,
                "redaction_modes": ["medical_only", "pii_only", "both"],
            },
        )

        mlflow.log_params(
            {
                "medical_model": medical_config.model_name,
                "pii_model": pii_config.model_type,
                "medical_threshold": medical_config.threshold,
                "pii_threshold": pii_config.threshold,
            }
        )

    registered_model_info = mlflow.register_model(
        model_uri=logged_model_info.model_uri, name=model_name_full
    )

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name_full, alias="champion", version=registered_model_info.version
    )

    logger.info(
        f"Two-stage model registered: {model_name_full} version {registered_model_info.version}"
    )
    return f"models:/{model_name_full}/{registered_model_info.version}"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Execution

# COMMAND ----------

# Set up Unity Catalog
mlflow.set_registry_uri("databricks-uc")

print("Setting up Unity Catalog for two-stage detection")

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
    print(f"Schema '{catalog_name}.{schema_name}' ready")
except Exception as e:
    print(f"Schema setup issue: {e}")

# Generate test data if requested
if generate_data:
    print("Generating test data with medical and PII content")
    test_data = generate_test_data(20)
    spark_df = spark.createDataFrame(test_data)

    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        full_source_table
    )

    print(f"Test data saved to: {full_source_table}")
    display(spark.table(full_source_table))

# COMMAND ----------

# Model configurations
medical_config = MedicalDetectorConfig(
    model_name="dmis-lab/biobert-v1.1",
    cache_dir=cache_dir,
    threshold=0.7,
    batch_size=16,
)

pii_config = PIIDetectorConfig(
    model_type=pii_model,
    model_name=(
        "dslim/distilbert-NER"
        if pii_model == "distilbert"
        else "Ihor/gliner-biomed-base-v1.0"
    ),
    cache_dir=cache_dir,
    threshold=0.8,
    batch_size=16,
)

two_stage_model_name = f"{catalog_name}.{schema_name}.{model_name}_{pii_model}"

print(f"Two-stage detection setup:")
print(f"  Medical detector: {medical_config.model_name}")
print(f"  PII detector: {pii_config.model_type}")
print(f"  Redaction mode: {redaction_mode}")

# Train or load model
if train:
    model_uri = train_and_register_two_stage_model(
        medical_config, pii_config, two_stage_model_name
    )
    print(f"Two-stage model registered: {model_uri}")
else:
    model_uri = f"models:/{two_stage_model_name}@champion"
    print(f"Two-stage model loading from UC: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Processing and Results

# COMMAND ----------


# Create pandas UDF for batch processing
@pandas_udf("string")
def two_stage_udf(text_series: pd.Series) -> pd.Series:
    """UDF for two-stage detection processing."""
    try:
        print(f"Processing batch with two-stage detection: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Two-stage model loaded")

        # Process batch
        input_df = pd.DataFrame(
            {"text": text_series, "text_type": ["medical"] * len(text_series)}
        )
        results = model.predict(input_df)

        # Return appropriate redaction mode
        if redaction_mode == "medical_only":
            return pd.Series(results["medical_only_redacted"].tolist())
        elif redaction_mode == "pii_only":
            return pd.Series(results["pii_only_redacted"].tolist())
        else:  # both
            return pd.Series(results["both_redacted"].tolist())

    except Exception as e:
        error_msg = f"Two-stage processing failed: {str(e)}"
        print(f"{error_msg}")
        return pd.Series([f"[ERROR: {error_msg}]"] * len(text_series))


# Process data
if spark.catalog.tableExists(full_source_table):
    df = spark.table(full_source_table)

    # Apply two-stage detection
    processed_df = df.withColumn(f"two_stage_redacted_text", two_stage_udf(col("text")))

    # Save results
    processed_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        full_results_table
    )

    print(f"Two-stage results saved to: {full_results_table}")

    # Show results
    result_df = spark.table(full_results_table)
    display(result_df.select("text", "two_stage_redacted_text"))

    print(f"\nTwo-stage detection completed:")
    print(f"  Medical detector: {medical_config.model_name}")
    print(f"  PII detector: {pii_config.model_type}")
    print(f"  Redaction mode: {redaction_mode}")
    print(f"  Processed records: {result_df.count()}")

else:
    print(
        f"Source table {full_source_table} not found. Run with generate_data=true first."
    )
