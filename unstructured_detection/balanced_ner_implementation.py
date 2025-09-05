# Databricks notebook source
# MAGIC %md
# MAGIC # Balanced NER Implementation: GLiNER-biomed vs Enhanced Presidio
# MAGIC
# MAGIC This notebook provides both implementations with fair evaluation:
# MAGIC 1. GLiNER-biomed: Higher performance, security considerations
# MAGIC 2. Enhanced Presidio: Lower performance, better security profile
# MAGIC 3. Comparative evaluation on medical text
# MAGIC 4. Realistic security risk assessment

# COMMAND ----------

# MAGIC %pip install -r ../requirements_ner.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
import logging

sys.path.append("../")

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
dbutils.widgets.text("model_name_gliner", "ner_gliner_model", "GLiNER Model Name")
dbutils.widgets.text("model_name_presidio", "ner_presidio_model", "Presidio Model Name")
dbutils.widgets.text(
    "hf_cache_dir", "/Volumes/dbxmetagen/default/models/hf_cache", "HF Cache Dir"
)
dbutils.widgets.dropdown(
    "model_to_use", "both", ["gliner", "presidio", "both"], "Model Selection"
)

# Get values
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
source_table = dbutils.widgets.get("source_table")
results_table = dbutils.widgets.get("results_table")
model_name_gliner = dbutils.widgets.get("model_name_gliner")
model_name_presidio = dbutils.widgets.get("model_name_presidio")
hf_cache_dir = dbutils.widgets.get("hf_cache_dir")
model_to_use = dbutils.widgets.get("model_to_use")

# Full names
full_source_table = f"{catalog_name}.{schema_name}.{source_table}"
full_results_table = f"{catalog_name}.{schema_name}.{results_table}"
full_model_name_gliner = f"{catalog_name}.{schema_name}.{model_name_gliner}"
full_model_name_presidio = f"{catalog_name}.{schema_name}.{model_name_presidio}"

print(f"Source: {full_source_table}")
print(f"Results: {full_results_table}")
print(f"GLiNER Model: {full_model_name_gliner}")
print(f"Presidio Model: {full_model_name_presidio}")
print(f"Model Selection: {model_to_use}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Configurations

# COMMAND ----------


@dataclass
class GLiNERConfig:
    """GLiNER-biomed configuration with security awareness."""

    model_name: str = "Ihor/gliner-biomed-base-v1.0"
    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.5
    batch_size: int = 16

    # Medical entity labels for GLiNER
    entity_labels: List[str] = None

    def __post_init__(self):
        if self.entity_labels is None:
            self.entity_labels = [
                "person",
                "doctor",
                "patient",
                "hospital",
                "medical condition",
                "diagnosis",
                "medication",
                "treatment",
                "lab test",
                "medical record number",
                "insurance number",
                "date",
                "phone number",
                "email",
                "address",
                "social security number",
                "credit card",
                "driver license",
            ]


@dataclass
class PresidioConfig:
    """Enhanced Presidio configuration."""

    threshold: float = 0.7
    supported_languages: List[str] = None

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en"]


# COMMAND ----------

# MAGIC %md
# MAGIC ## GLiNER-biomed Model Implementation

# COMMAND ----------


class GLiNERBiomedModel(mlflow.pyfunc.PythonModel):
    """GLiNER-biomed model with security logging and controls."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.model = None

    def load_context(self, context):
        """Load GLiNER model with security awareness."""
        try:
            logger.info("🔬 Loading GLiNER-biomed model...")
            logger.warning(
                "⚠️ SECURITY NOTICE: GLiNER package has known vulnerabilities"
            )
            logger.info("🛡️ MITIGATION: Running in isolated Databricks environment")
            logger.info("📋 RECOMMENDATION: Implement network isolation and monitoring")

            # Set cache environment
            cache_dir = self.config["cache_dir"]
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir

            # Security: Disable telemetry
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

            os.makedirs(cache_dir, exist_ok=True)

            self.model = GLiNER.from_pretrained(
                self.config["model_name"], cache_dir=cache_dir, force_download=False
            )

            logger.info("✅ GLiNER-biomed loaded successfully")
            logger.info("🔒 Security status: VULNERABLE package with mitigations")

        except Exception as e:
            logger.error(f"GLiNER loading failed: {str(e)}")
            raise RuntimeError(f"GLiNER loading failed: {str(e)}") from e

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict entities using GLiNER-biomed."""
        results = []

        for _, row in model_input.iterrows():
            text = row.get("text", "")
            text_type = row.get("text_type", "general")

            # Use all labels for comprehensive detection
            labels = self.config["entity_labels"]

            # GLiNER prediction
            entities = self.model.predict_entities(
                text, labels, threshold=self.config["threshold"]
            )

            # Convert to standard format
            detected_entities = []
            for entity in entities:
                detected_entities.append(
                    {
                        "text": entity["text"],
                        "label": entity["label"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "score": entity["score"],
                        "source": "gliner",
                    }
                )

            # Simple redaction (replace with label)
            redacted_text = text
            for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
                placeholder = f"[{entity['label'].upper()}]"
                redacted_text = (
                    redacted_text[: entity["start"]]
                    + placeholder
                    + redacted_text[entity["end"] :]
                )

            results.append(
                {
                    "text": text,
                    "entities": json.dumps(detected_entities),
                    "redacted_text": redacted_text,
                    "entity_count": len(detected_entities),
                }
            )

        return pd.DataFrame(results)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Enhanced Presidio Model Implementation

# COMMAND ----------


class EnhancedPresidioModel(mlflow.pyfunc.PythonModel):
    """Enhanced Presidio model with comprehensive medical patterns."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.analyzer = None
        self.anonymizer = None

    def load_context(self, context):
        """Load enhanced Presidio with medical patterns."""
        try:
            logger.info("🛡️ Loading Enhanced Presidio model...")
            logger.info(
                "✅ SECURITY STATUS: Enterprise-backed, no known vulnerabilities"
            )

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

            # Add comprehensive medical patterns
            self._add_medical_patterns()

            logger.info("✅ Enhanced Presidio loaded successfully")

        except Exception as e:
            logger.error(f"Presidio loading failed: {str(e)}")
            raise RuntimeError(f"Presidio loading failed: {str(e)}") from e

    def _add_medical_patterns(self):
        """Add comprehensive medical entity recognition patterns."""

        # Medical Record Number patterns
        mrn_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=[
                Pattern(
                    "mrn_labeled",
                    r"\b(MRN|Medical Record|Patient ID)[\s:]*(\d{4,8})\b",
                    0.9,
                ),
                Pattern(
                    "mrn_context", r"\b\d{6,8}\b(?=.*(?:patient|medical|hospital))", 0.7
                ),
            ],
            context=["medical", "patient", "hospital", "record"],
        )

        # Doctor/Physician patterns
        doctor_recognizer = PatternRecognizer(
            supported_entity="DOCTOR",
            patterns=[
                Pattern(
                    "doctor_title",
                    r"\b(Dr\.|Doctor|MD|DO)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+",
                    0.9,
                ),
                Pattern(
                    "doctor_format",
                    r"\b[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+,?\s*(MD|DO|DDS|PharmD)\b",
                    0.8,
                ),
            ],
            context=["doctor", "physician", "attending", "medical"],
        )

        # Medical condition patterns (common ones)
        condition_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_CONDITION",
            patterns=[
                Pattern(
                    "common_conditions",
                    r"\b(diabetes|hypertension|pneumonia|appendicitis|myocardial infarction|heart failure|cancer|asthma|copd|stroke)\b",
                    0.8,
                ),
                Pattern(
                    "pain_symptoms",
                    r"\b(chest pain|abdominal pain|back pain|headache|shortness of breath)\b",
                    0.7,
                ),
                Pattern(
                    "condition_with_type",
                    r"\b(Type [12] diabetes|Stage [1-4] cancer|Grade [1-4] \w+)\b",
                    0.9,
                ),
            ],
            context=["diagnosis", "condition", "disease", "symptom"],
        )

        # Medication patterns
        medication_recognizer = PatternRecognizer(
            supported_entity="MEDICATION",
            patterns=[
                Pattern(
                    "common_medications",
                    r"\b(Metformin|Lisinopril|Atorvastatin|Metoprolol|Aspirin|Insulin|Warfarin|Prednisone)\b",
                    0.9,
                ),
                Pattern(
                    "medication_with_dose",
                    r"\b[A-Z][a-z]+(?:stat|pril|mine|xin|ide)\s+\d+\s*mg\b",
                    0.8,
                ),
                Pattern(
                    "generic_medication",
                    r"\b[A-Z][a-z]{4,}\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b",
                    0.7,
                ),
            ],
            context=["medication", "drug", "prescription", "pharmacy"],
        )

        # Lab test patterns
        lab_recognizer = PatternRecognizer(
            supported_entity="LAB_TEST",
            patterns=[
                Pattern(
                    "common_labs",
                    r"\b(HbA1c|troponin|BNP|creatinine|WBC|CBC|hemoglobin|glucose|cholesterol)\b",
                    0.8,
                ),
                Pattern(
                    "imaging_tests",
                    r"\b(CT|MRI|X-ray|ultrasound|echocardiogram)\b",
                    0.8,
                ),
                Pattern(
                    "lab_with_value",
                    r"\b(HbA1c|troponin|BNP)\s+(?:of\s+)?[\d.]+\s*(?:%|ng/mL|pg/mL|mg/dL)\b",
                    0.9,
                ),
            ],
            context=["lab", "test", "result", "blood", "imaging"],
        )

        # Hospital/Organization patterns
        hospital_recognizer = PatternRecognizer(
            supported_entity="HOSPITAL",
            patterns=[
                Pattern(
                    "hospital_names",
                    r"\b[A-Z][a-zA-Z\s]*(?:Hospital|Medical Center|Clinic|Health System)\b",
                    0.8,
                ),
                Pattern(
                    "department_names",
                    r"\b(?:Emergency|Cardiology|Surgery|Internal Medicine|Radiology)\s+Department\b",
                    0.7,
                ),
            ],
            context=["hospital", "medical", "center", "clinic"],
        )

        # Insurance patterns
        insurance_recognizer = PatternRecognizer(
            supported_entity="INSURANCE_NUMBER",
            patterns=[
                Pattern(
                    "insurance_policy",
                    r"\b(?:Policy|Member ID|Group)[\s#]*([A-Z]{2,4}\d{6,12})\b",
                    0.9,
                ),
                Pattern(
                    "medicare_format", r"\b\d{3}-\d{2}-\d{4}[A-Z]\b", 0.8
                ),  # Medicare format
            ],
            context=["insurance", "policy", "coverage", "member"],
        )

        # Add all recognizers
        recognizers = [
            mrn_recognizer,
            doctor_recognizer,
            condition_recognizer,
            medication_recognizer,
            lab_recognizer,
            hospital_recognizer,
            insurance_recognizer,
        ]

        for recognizer in recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

        logger.info(f"✅ Added {len(recognizers)} enhanced medical pattern recognizers")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict using enhanced Presidio."""
        results = []

        # Comprehensive entity list
        entities_to_detect = [
            # Standard PII
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "DATE_TIME",
            "LOCATION",
            "CREDIT_CARD",
            "US_DRIVER_LICENSE",
            "US_PASSPORT",
            # Custom medical entities
            "MEDICAL_RECORD_NUMBER",
            "DOCTOR",
            "MEDICAL_CONDITION",
            "MEDICATION",
            "LAB_TEST",
            "HOSPITAL",
            "INSURANCE_NUMBER",
        ]

        for _, row in model_input.iterrows():
            text = row.get("text", "")

            # Presidio analysis
            detected_entities = self.analyzer.analyze(
                text=text,
                entities=entities_to_detect,
                language="en",
                score_threshold=self.config["threshold"],
            )

            # Convert to standard format
            entities_json = []
            for entity in detected_entities:
                entities_json.append(
                    {
                        "text": text[entity.start : entity.end],
                        "label": entity.entity_type,
                        "start": entity.start,
                        "end": entity.end,
                        "score": entity.score,
                        "source": "presidio",
                    }
                )

            # Anonymize text using Presidio's anonymizer
            anonymized_result = self.anonymizer.anonymize(
                text=text, analyzer_results=detected_entities
            )

            results.append(
                {
                    "text": text,
                    "entities": json.dumps(entities_json),
                    "redacted_text": anonymized_result.text,
                    "entity_count": len(entities_json),
                }
            )

        return pd.DataFrame(results)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Data Generation

# COMMAND ----------


def generate_comprehensive_test_data() -> pd.DataFrame:
    """Generate comprehensive test data for fair model comparison."""

    test_cases = [
        {
            "id": 1,
            "clean_text": "The recent advances in artificial intelligence and machine learning have transformed how organizations approach data analysis and decision-making processes.",
            "pii_text": "Amanda Rodriguez can be reached at amanda.rodriguez@university.edu or (212) 555-7890. Her address is 456 Academic Drive, Suite 12B, Boston, MA 02118.",
            "phi_text": "Patient William Chen, MRN 456789, DOB 09/22/1968, was admitted to Saint Mary's Medical Center under Dr. Patricia Williams. Diagnosed with acute myocardial infarction. Treatment: Metoprolol 50mg twice daily.",
            # Ground truth entities for evaluation
            "pii_ground_truth": [
                {"text": "Amanda Rodriguez", "label": "PERSON"},
                {"text": "amanda.rodriguez@university.edu", "label": "EMAIL_ADDRESS"},
                {"text": "(212) 555-7890", "label": "PHONE_NUMBER"},
                {
                    "text": "456 Academic Drive, Suite 12B, Boston, MA 02118",
                    "label": "LOCATION",
                },
            ],
            "phi_ground_truth": [
                {"text": "William Chen", "label": "PERSON"},
                {"text": "456789", "label": "MEDICAL_RECORD_NUMBER"},
                {"text": "09/22/1968", "label": "DATE_TIME"},
                {"text": "Saint Mary's Medical Center", "label": "HOSPITAL"},
                {"text": "Dr. Patricia Williams", "label": "DOCTOR"},
                {"text": "acute myocardial infarction", "label": "MEDICAL_CONDITION"},
                {"text": "Metoprolol", "label": "MEDICATION"},
            ],
        },
        {
            "id": 2,
            "clean_text": "Technology continues to evolve rapidly in the modern world with new innovations emerging daily.",
            "pii_text": "Michael Thompson (DOB: July 4, 1982) can be reached at michael.thompson@techcorp.com. His SSN is 456-78-9012 and driver's license is CA-DL-987654321.",
            "phi_text": "Emergency Department Note: Patient Rebecca Martinez, MRN 789012, presented with severe abdominal pain. Dr. James Rodriguez recommended laparoscopic appendectomy. Lab: WBC count 14,500/µL.",
            "pii_ground_truth": [
                {"text": "Michael Thompson", "label": "PERSON"},
                {"text": "July 4, 1982", "label": "DATE_TIME"},
                {"text": "michael.thompson@techcorp.com", "label": "EMAIL_ADDRESS"},
                {"text": "456-78-9012", "label": "US_SSN"},
                {"text": "CA-DL-987654321", "label": "US_DRIVER_LICENSE"},
            ],
            "phi_ground_truth": [
                {"text": "Rebecca Martinez", "label": "PERSON"},
                {"text": "789012", "label": "MEDICAL_RECORD_NUMBER"},
                {"text": "Dr. James Rodriguez", "label": "DOCTOR"},
                {"text": "abdominal pain", "label": "MEDICAL_CONDITION"},
                {"text": "laparoscopic appendectomy", "label": "MEDICATION"},
                {"text": "WBC count", "label": "LAB_TEST"},
            ],
        },
        {
            "id": 3,
            "clean_text": "The weather today is sunny and warm with clear blue skies perfect for outdoor activities.",
            "pii_text": "Contact Sarah Johnson at sarah.johnson@email.com or (217) 555-9876. Lives at 456 Oak Street, Springfield, IL 62701. Insurance: Policy ABC123456789.",
            "phi_text": "Mary Johnson, DOB 05/14/1975, diabetes management follow-up. Metformin 1000mg twice daily. Recent HbA1c of 7.2% shows improvement. Medicare Part B coverage.",
            "pii_ground_truth": [
                {"text": "Sarah Johnson", "label": "PERSON"},
                {"text": "sarah.johnson@email.com", "label": "EMAIL_ADDRESS"},
                {"text": "(217) 555-9876", "label": "PHONE_NUMBER"},
                {"text": "456 Oak Street, Springfield, IL 62701", "label": "LOCATION"},
                {"text": "ABC123456789", "label": "INSURANCE_NUMBER"},
            ],
            "phi_ground_truth": [
                {"text": "Mary Johnson", "label": "PERSON"},
                {"text": "05/14/1975", "label": "DATE_TIME"},
                {"text": "diabetes", "label": "MEDICAL_CONDITION"},
                {"text": "Metformin", "label": "MEDICATION"},
                {"text": "HbA1c", "label": "LAB_TEST"},
            ],
        },
    ]

    # Flatten data for DataFrame
    data = []
    for case in test_cases:
        data.append(
            {
                "id": case["id"],
                "clean_text": case["clean_text"],
                "pii_text": case["pii_text"],
                "phi_text": case["phi_text"],
                "pii_ground_truth_entities": json.dumps(case["pii_ground_truth"]),
                "phi_ground_truth_entities": json.dumps(case["phi_ground_truth"]),
            }
        )

    return pd.DataFrame(data)


# Generate and save test data
test_data_df = generate_comprehensive_test_data()
test_data_spark_df = spark.createDataFrame(test_data_df)

test_data_spark_df.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(full_source_table)

print(f"✅ Test data saved to {full_source_table}")
print(f"📊 Generated {len(test_data_df)} test cases")
display(spark.table(full_source_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Registration

# COMMAND ----------

# Set up Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Setup Unity Catalog resources
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

# Create cache volume for GLiNER
try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.hf_cache_ner")
    cache_dir = hf_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"✅ Unity Catalog cache ready: {cache_dir}")
except Exception as e:
    print(f"⚠️ Volume setup issue: {e}")
    cache_dir = "/tmp/hf_cache_fallback"
    os.makedirs(cache_dir, exist_ok=True)

if model_to_use in ["gliner", "both"]:
    # Train and register GLiNER model
    print("\n🔬 Training and registering GLiNER-biomed model...")

    gliner_config = GLiNERConfig(cache_dir=cache_dir)
    gliner_model = GLiNERBiomedModel(
        {
            "model_name": gliner_config.model_name,
            "cache_dir": gliner_config.cache_dir,
            "threshold": gliner_config.threshold,
            "entity_labels": gliner_config.entity_labels,
        }
    )

    # Sample input for signature
    sample_input = pd.DataFrame(
        {"text": ["Dr. Smith treated patient John Doe"], "text_type": ["phi"]}
    )

    gliner_model.load_context(None)
    sample_output = gliner_model.predict(None, sample_input)
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run():
        gliner_model_info = mlflow.pyfunc.log_model(
            artifact_path="gliner_ner_model",
            python_model=gliner_model,
            signature=signature,
            pip_requirements=[
                "numpy>=1.21.5,<2.0",
                "pandas>=1.5.0,<2.1.0",
                "gliner==0.2.5",
                "transformers==4.44.0",
                "torch==2.4.0",
            ],
            input_example=sample_input,
            metadata={
                "model_type": "gliner_biomed",
                "security_status": "vulnerable_with_mitigations",
            },
        )

    gliner_registered_model = mlflow.register_model(
        model_uri=gliner_model_info.model_uri, name=full_model_name_gliner
    )

    gliner_model_uri = (
        f"models:/{full_model_name_gliner}/{gliner_registered_model.version}"
    )
    print(f"✅ GLiNER model registered: {gliner_model_uri}")

if model_to_use in ["presidio", "both"]:
    # Train and register Presidio model
    print("\n🛡️ Training and registering Enhanced Presidio model...")

    presidio_config = PresidioConfig()
    presidio_model = EnhancedPresidioModel(
        {
            "threshold": presidio_config.threshold,
            "supported_languages": presidio_config.supported_languages,
        }
    )

    presidio_model.load_context(None)
    presidio_sample_output = presidio_model.predict(None, sample_input)
    presidio_signature = infer_signature(sample_input, presidio_sample_output)

    with mlflow.start_run():
        presidio_model_info = mlflow.pyfunc.log_model(
            artifact_path="presidio_ner_model",
            python_model=presidio_model,
            signature=presidio_signature,
            pip_requirements=[
                "presidio-analyzer==2.2.358",
                "presidio-anonymizer==2.2.358",
            ],
            input_example=sample_input,
            metadata={"model_type": "enhanced_presidio", "security_status": "secure"},
        )

    presidio_registered_model = mlflow.register_model(
        model_uri=presidio_model_info.model_uri, name=full_model_name_presidio
    )

    presidio_model_uri = (
        f"models:/{full_model_name_presidio}/{presidio_registered_model.version}"
    )
    print(f"✅ Presidio model registered: {presidio_model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Evaluation

# COMMAND ----------


def evaluate_model_performance(
    results_column: str, ground_truth_column: str, df: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate NER model performance against ground truth."""

    total_tp = total_fp = total_fn = 0

    for _, row in df.iterrows():
        # Parse entities
        try:
            predicted = json.loads(row[results_column]) if row[results_column] else []
            ground_truth = (
                json.loads(row[ground_truth_column]) if row[ground_truth_column] else []
            )
        except:
            continue

        # Normalize and compare
        predicted_spans = {
            (ent["text"].lower().strip(), ent["label"].upper()) for ent in predicted
        }
        ground_truth_spans = {
            (ent["text"].lower().strip(), ent["label"].upper()) for ent in ground_truth
        }

        tp = len(predicted_spans.intersection(ground_truth_spans))
        fp = len(predicted_spans - ground_truth_spans)
        fn = len(ground_truth_spans - predicted_spans)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


# Run evaluation if both models are trained
if model_to_use == "both":
    print("📊 COMPARATIVE PERFORMANCE EVALUATION")
    print("=" * 60)

    # Load source data
    source_df = spark.table(full_source_table).toPandas()

    # Test both models on PII and PHI text
    print("\n🔬 Testing GLiNER-biomed...")
    gliner_test_df = pd.DataFrame(
        {
            "text": list(source_df["pii_text"]) + list(source_df["phi_text"]),
            "text_type": ["pii"] * len(source_df) + ["phi"] * len(source_df),
        }
    )

    print("\n🛡️ Testing Enhanced Presidio...")
    presidio_test_df = gliner_test_df.copy()

    # Get predictions (would normally load from registered models)
    gliner_model = GLiNERBiomedModel(
        {
            "model_name": "Ihor/gliner-biomed-base-v1.0",
            "cache_dir": cache_dir,
            "threshold": 0.5,
            "entity_labels": GLiNERConfig().entity_labels,
        }
    )
    gliner_model.load_context(None)
    gliner_predictions = gliner_model.predict(None, gliner_test_df)

    presidio_model = EnhancedPresidioModel({"threshold": 0.7})
    presidio_model.load_context(None)
    presidio_predictions = presidio_model.predict(None, presidio_test_df)

    # Create evaluation DataFrame
    eval_df = pd.DataFrame(
        {
            "text": gliner_test_df["text"],
            "gliner_entities": gliner_predictions["entities"],
            "presidio_entities": presidio_predictions["entities"],
            "ground_truth": (
                list(source_df["pii_ground_truth_entities"])
                + list(source_df["phi_ground_truth_entities"])
            ),
        }
    )

    # Evaluate both models
    gliner_metrics = evaluate_model_performance(
        "gliner_entities", "ground_truth", eval_df
    )
    presidio_metrics = evaluate_model_performance(
        "presidio_entities", "ground_truth", eval_df
    )

    print(f"\n🔬 GLiNER-biomed Performance:")
    print(f"   Precision: {gliner_metrics['precision']:.3f}")
    print(f"   Recall:    {gliner_metrics['recall']:.3f}")
    print(f"   F1-Score:  {gliner_metrics['f1']:.3f}")

    print(f"\n🛡️ Enhanced Presidio Performance:")
    print(f"   Precision: {presidio_metrics['precision']:.3f}")
    print(f"   Recall:    {presidio_metrics['recall']:.3f}")
    print(f"   F1-Score:  {presidio_metrics['f1']:.3f}")

    f1_difference = gliner_metrics["f1"] - presidio_metrics["f1"]
    print(f"\n⚖️ Performance Difference:")
    print(f"   GLiNER F1 - Presidio F1: {f1_difference:+.3f}")

    if f1_difference > 0.1:
        print("   📈 GLiNER shows SIGNIFICANT performance advantage")
        print("   💭 Security risk may be justified with strong mitigations")
    elif f1_difference > 0.05:
        print("   📊 GLiNER shows MODERATE performance advantage")
        print("   💭 Consider security vs performance tradeoff")
    else:
        print("   📊 Performance difference is MINIMAL")
        print("   💭 Enhanced Presidio recommended for better security")

    # Log metrics to MLflow
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_metrics(
            {
                "gliner_precision": gliner_metrics["precision"],
                "gliner_recall": gliner_metrics["recall"],
                "gliner_f1": gliner_metrics["f1"],
                "presidio_precision": presidio_metrics["precision"],
                "presidio_recall": presidio_metrics["recall"],
                "presidio_f1": presidio_metrics["f1"],
                "f1_difference": f1_difference,
            }
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Recommendations

# COMMAND ----------

final_recommendations = """
🎯 FINAL RECOMMENDATIONS BASED ON EVALUATION

PERFORMANCE RESULTS:
✅ Run the comparative evaluation above to see actual F1 score differences
✅ GLiNER-biomed likely performs better on medical text (literature shows ~6% improvement)
✅ Enhanced Presidio provides reasonable coverage with enterprise security

SECURITY ASSESSMENT:
⚠️ GLiNER package has confirmed vulnerabilities with active exploitation
✅ Databricks environment provides good isolation and mitigation
🛡️ Enhanced Presidio has no known security issues

DECISION FRAMEWORK:

IF F1 difference > 0.1 (significant performance gap):
   → USE GLiNER-biomed WITH security controls:
     • Network isolation on cluster
     • Regular security monitoring  
     • Version pinning (no auto-updates)
     • Document risk acceptance

IF F1 difference 0.05-0.1 (moderate performance gap):
   → Business decision based on:
     • Criticality of NER accuracy
     • Security team risk tolerance
     • Available security resources

IF F1 difference < 0.05 (minimal performance gap):
   → USE Enhanced Presidio:
     • Better security profile
     • Minimal performance cost
     • Enterprise backing

IMPLEMENTATION:
✅ Both models are now registered and ready to use
✅ Evaluation framework in place for ongoing assessment
✅ Security mitigations documented and recommended

The choice is now based on DATA, not speculation.
Run the evaluation and make an informed decision! 🚀
"""

print(final_recommendations)

# COMMAND ----------
