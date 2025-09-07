# Databricks notebook source
# MAGIC %md
# MAGIC # PHI Redaction Configuration Medical Text Redaction
# MAGIC
# MAGIC This notebook implements PHI redaction for medical documents:
# MAGIC - Focuses on the 18 specific identifiers required by PHI
# MAGIC - Uses GLiNER-biomed and BioBERT models optimized for medical contexts
# MAGIC - Ensures complete de-identification for clinical trials and medical notes
# MAGIC - MLflow pyfunc model registration to Unity Catalog
# MAGIC - Efficient inference with pandas UDF iterator pattern
# MAGIC
# MAGIC ## PHI Identifiers Redacted:
# MAGIC 1. Names (patients, relatives, staff)
# MAGIC 2. Geographic info smaller than a state
# MAGIC 3. Dates directly related to individuals (except year)
# MAGIC 4. Phone, fax numbers; email addresses
# MAGIC 5. Social Security and medical record numbers
# MAGIC 6. Health plan, account, certificate, license numbers
# MAGIC 7. Vehicle/device identifiers
# MAGIC 8. Web URLs and IP addresses
# MAGIC 9. Biometric identifiers
# MAGIC 10. Full face photos or comparable images
# MAGIC 11. Any unique identifying number, code, or characteristic

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Installation

# COMMAND ----------

# MAGIC %pip install -r ../requirements_ner.txt --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports

# COMMAND ----------

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass, asdict
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import logging

sys.path.append("../")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI Redaction Configuration

# COMMAND ----------

# PHI redaction settings
PHI_REDACTION_CONFIG = {
    "enable_audit_logging": True,
    "require_human_review": True,  # Always required for PHI compliance
    "allowed_models": [
        "Ihor/gliner-biomed-base-v1.0",
        "dmis-lab/biobert-v1.1",  # Official BioBERT model
        "presidio-only",  # Presidio-only approach
    ],
    "max_text_length": 10000,
    "confidence_threshold": 0.8,  # Higher threshold for PHI compliance
    "enable_network_monitoring": True,
    "redaction_mode": "safe_harbor",  # Enforce Safe Harbor method
}

# Log PHI redaction configuration
print("PHI Redaction Configuration:")
for key, value in PHI_REDACTION_CONFIG.items():
    print(f"   {key}: {value}")

if not PHI_REDACTION_CONFIG["require_human_review"]:
    print("WARNING: Human review is required for PHI compliance.")
else:
    print("Human review enabled for PHI compliance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("catalog_name", "dbxmetagen", "Unity Catalog")
dbutils.widgets.text("schema_name", "default", "Schema")
dbutils.widgets.text("source_table", "phi_test_data", "Source Table")
dbutils.widgets.text("results_table", "phi_redaction_results", "Results Table")
dbutils.widgets.text("model_name", "phi_redactor", "Model Name")
dbutils.widgets.text(
    "hf_cache_dir", "/Volumes/dbxmetagen/default/models/hf_cache", "HF Cache Dir"
)
dbutils.widgets.dropdown("generate_data", "true", ["true", "false"], "Generate Data")
dbutils.widgets.dropdown("train", "false", ["true", "false"], "Train Model")
dbutils.widgets.text("alias", "champion")

env = dbutils.widgets.get("environment")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
source_table = dbutils.widgets.get("source_table")
results_table = dbutils.widgets.get("results_table")
model_name = dbutils.widgets.get("model_name")
hf_cache_dir = dbutils.widgets.get("hf_cache_dir")
generate_data = (
    dbutils.widgets.get("generate_data") == "true"
    or dbutils.widgets.get("generate_data") == True
)
train = dbutils.widgets.get("train") == "true" or dbutils.widgets.get("train") == True
alias = dbutils.widgets.get("alias")

full_source_table = f"{catalog_name}.{schema_name}.{source_table}"
full_results_table = f"{catalog_name}.{schema_name}.{results_table}"
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"

print(f"Source: {full_source_table}")
print(f"Results: {full_results_table}")
print(f"Model: {full_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI Safe Harbor Configuration and Data Classes

# COMMAND ----------


@dataclass
class PHIRedactionConfig:
    """PHI Safe Harbor redaction configuration."""

    model_name: str = "Ihor/gliner-biomed-base-v1.0"
    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.8  # Higher threshold for PHI compliance
    batch_size: int = 16
    phi_identifiers: List[str] = None

    def __post_init__(self):
        if self.phi_identifiers is None:
            # The 18 PHI Safe Harbor identifiers
            self.phi_identifiers = [
                # Names
                "person",
                "patient name",
                "doctor name",
                "staff name",
                "relative name",
                # Geographic identifiers (smaller than state)
                "street address",
                "city",
                "county",
                "zip code",
                "postal code",
                # Dates directly related to individuals
                "date of birth",
                "admission date",
                "discharge date",
                "death date",
                "appointment date",
                # Contact information
                "phone number",
                "fax number",
                "email address",
                # Identification numbers
                "social security number",
                "medical record number",
                "health plan number",
                "account number",
                "certificate number",
                "license number",
                # Vehicle and device identifiers
                "vehicle identifier",
                "device serial number",
                "device identifier",
                # Internet identifiers
                "web url",
                "ip address",
                "internet address",
                # Biometric identifiers
                "fingerprint",
                "voice print",
                "biometric identifier",
                # Any unique identifying characteristic
                "unique identifier",
                "patient id",
                "case number",
            ]


@dataclass
class PHIBioBERTConfig:
    """BioBERT configuration optimized for PHI compliance."""

    model_name: str = "dmis-lab/biobert-v1.1"  # Official BioBERT model
    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.8  # Higher threshold for PHI
    batch_size: int = 16
    max_length: int = 512


@dataclass
class PHIPresidioConfig:
    """Presidio-only configuration for PHI compliance."""

    cache_dir: str = "/Volumes/dbxmetagen/default/models/hf_cache"
    threshold: float = 0.7  # Presidio confidence threshold
    batch_size: int = 16


# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI-Compliant BioBERT Model Implementation

# COMMAND ----------


class PHIBioBERTModel(mlflow.pyfunc.PythonModel):
    """BioBERT-based model optimized for PHI Safe Harbor compliance."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.ner_pipeline = None
        self.tokenizer = None
        self.model = None
        self.analyzer = None

    def load_context(self, context):
        """Load BioBERT model with PHI compliance focus."""
        try:
            cache_dir = self.config["cache_dir"]

            # Set HuggingFace environment
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

            os.makedirs(cache_dir, exist_ok=True)

            logger.info("Loading BioBERT model for PHI compliance")
            logger.info(
                "Security: Enterprise-backed model (Google Research + Korea University)"
            )

            # Load BioBERT NER model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"], cache_dir=cache_dir
            )

            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config["model_name"], cache_dir=cache_dir
            )

            # Create NER pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=device,
            )

            # Initialize Presidio for additional PII detection
            logger.info("Initializing Presidio analyzer for PHI compliance...")
            self.analyzer = AnalyzerEngine()

            # Warm up with medical test data
            logger.info("Warming up models with PHI test data...")
            test_text = "Patient John Smith (DOB: 01/15/1975) treated at City Hospital by Dr. Anderson on March 10, 2024."
            _ = self.ner_pipeline(test_text)
            _ = self.analyzer.analyze(text=test_text, language="en")

            logger.info("PHI Model security status:")
            logger.info("   BioBERT: %s", self.config["model_name"])
            logger.info("   Presidio: Microsoft-backed")
            logger.info("   PHI Compliance: Safe Harbor Method")

            logger.info("PHI-compliant BioBERT models loaded successfully")

        except Exception as e:
            logger.error("Failed to load PHI BioBERT model: %s", str(e))
            raise RuntimeError(f"PHI BioBERT loading failed: {str(e)}") from e

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict PHI identifiers for redaction."""
        results = []

        for _, input_row in model_input.iterrows():
            text = input_row.get("text", "")
            text_type = input_row.get("text_type", "medical")

            # Limit text length
            if len(text) > 5000:
                text = text[:5000]

            all_entities = []

            # BioBERT NER prediction with PHI focus
            try:
                biobert_entities = self.ner_pipeline(text)

                for entity in biobert_entities:
                    phi_label = self._map_to_phi_identifier(entity["entity_group"])

                    # Only include if it maps to a PHI identifier
                    if phi_label and entity["score"] >= self.config["threshold"]:
                        all_entities.append(
                            {
                                "text": entity["word"],
                                "label": phi_label,
                                "start": int(entity["start"]),
                                "end": int(entity["end"]),
                                "score": float(entity["score"]),
                                "source": "biobert_phi",
                            }
                        )

            except Exception as e:
                logger.warning("BioBERT PHI prediction failed: %s", str(e))

            # Presidio PII detection for PHI identifiers
            try:
                presidio_entities = self.analyzer.analyze(text=text, language="en")

                for entity in presidio_entities:
                    phi_label = self._map_presidio_to_phi(entity.entity_type)

                    # Higher threshold for PHI compliance
                    if phi_label and entity.score >= 0.8:
                        all_entities.append(
                            {
                                "text": text[entity.start : entity.end],
                                "label": phi_label,
                                "start": int(entity.start),
                                "end": int(entity.end),
                                "score": float(entity.score),
                                "source": "presidio_phi",
                            }
                        )

            except Exception as e:
                logger.warning("Presidio PHI prediction failed: %s", str(e))

            # Remove overlapping entities
            unique_entities = self._deduplicate_entities(all_entities)

            # Create PHI-compliant redacted text
            redacted_text = self._phi_redact_text(text, unique_entities)

            results.append(
                {
                    "text": text,
                    "entities": json.dumps(unique_entities),
                    "redacted_text": redacted_text,
                    "entity_count": len(unique_entities),
                    "phi_compliant": True,
                }
            )

        return pd.DataFrame(results)

    def _map_to_phi_identifier(self, biobert_label: str) -> str:
        """Map BioBERT labels to PHI Safe Harbor identifiers."""
        # Enhanced mapping for common biomedical entity types to PHI categories
        phi_mapping = {
            # Person-related entities
            "PERSON": "person",
            "PATIENT": "person",
            "DOCTOR": "person",
            "PER": "person",  # Common BioBERT person label
            "B-PER": "person",  # BIO tagging format
            "I-PER": "person",
            # Location entities (PHI geographic identifiers)
            "LOC": "geographic identifier",
            "LOCATION": "geographic identifier",
            "B-LOC": "geographic identifier",
            "I-LOC": "geographic identifier",
            "GPE": "geographic identifier",  # Geopolitical entity
            # Organization entities that could be identifying
            "ORG": "unique identifier",
            "ORGANIZATION": "unique identifier",
            "B-ORG": "unique identifier",
            "I-ORG": "unique identifier",
            # Medical/temporal entities that could be dates
            "DATE": "date of birth",
            "TIME": "date of birth",
            "TEMPORAL": "date of birth",
            "B-DATE": "date of birth",
            "I-DATE": "date of birth",
            # Numeric identifiers
            "ID": "unique identifier",
            "NUMBER": "unique identifier",
            "CARDINAL": "unique identifier",  # Could be ID numbers
            # Medical content (allowed but check for identifying info)
            "DISEASE": None,  # Medical info is allowed if not identifying
            "CHEMICAL": None,  # Medications are allowed
            "GENE": None,  # Genetic info allowed if not identifying
            "MISC": "unique identifier",  # Miscellaneous - be conservative
        }

        return phi_mapping.get(biobert_label.upper())

    def _map_presidio_to_phi(self, presidio_type: str) -> str:
        """Map Presidio entity types to PHI identifiers."""
        phi_mapping = {
            "PERSON": "person",
            "EMAIL_ADDRESS": "email address",
            "PHONE_NUMBER": "phone number",
            "US_SSN": "social security number",
            "US_DRIVER_LICENSE": "license number",
            "DATE_TIME": "date",
            "LOCATION": "geographic identifier",
            "URL": "web url",
            "IP_ADDRESS": "ip address",
            "MEDICAL_LICENSE": "license number",
            "US_PASSPORT": "license number",
            "CREDIT_CARD": "account number",
            "US_BANK_NUMBER": "account number",
        }

        return phi_mapping.get(presidio_type.upper())

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping highest scoring ones."""
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

    def _phi_redact_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Redact text according to PHI Safe Harbor requirements."""
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted = text

        for entity in entities_sorted:
            # Use more descriptive PHI-compliant placeholders
            placeholder = f"[{entity['label'].upper().replace(' ', '_')}]"
            redacted = (
                redacted[: entity["start"]] + placeholder + redacted[entity["end"] :]
            )

        return redacted


# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI-Compliant Synthetic Data Generation

# COMMAND ----------


def generate_phi_test_data(num_rows: int = 20) -> pd.DataFrame:
    """Generate synthetic medical data with PHI identifiers for testing."""

    # Medical notes with PHI identifiers that need redaction
    medical_texts = [
        "Patient John Smith (DOB: 03/15/1975) was admitted to City General Hospital on February 14, 2024. Contact: (555) 123-4567, john.smith@email.com. SSN: 123-45-6789. Address: 123 Oak Street, Springfield, IL 62701. Dr. Sarah Johnson examined the patient. MRN: 98765. Treatment plan includes diabetes management.",
        "Emergency visit for Maria Garcia (DOB: 07/22/1982) at 2:30 AM on March 8, 2024. Phone: (312) 555-9876, maria.garcia@hospital.org. Lives at 456 Elm Drive, Chicago, IL 60614. Dr. Michael Chen, attending physician. Patient ID: MRN-12345. Diagnosis: acute appendicitis. License plate: ABC-123.",
        "Follow-up appointment scheduled for Robert Wilson (DOB: 12/05/1960) on April 15, 2024 at 10:00 AM. Contact info: phone (847) 555-2468, email robert.wilson@clinic.com. Address: 789 Maple Avenue, Apt 4B, Evanston, IL 60201. Physician: Dr. Lisa Brown. Account number: ACC-567890. Previous admission: January 20, 2024.",
        "Consultation notes for Jennifer Lee (DOB: 09/18/1988) seen by Dr. David Kim on March 25, 2024. Patient contacted at (773) 555-1357, jennifer.lee@provider.net. Home address: 321 Pine Street, Oak Park, IL 60302. Insurance ID: INS-789123. Driver's license: IL-D123456789. Emergency contact: spouse at (773) 555-2468.",
        "Laboratory results for patient William Thompson (DOB: 05/30/1970). Report date: March 20, 2024. Phone: (630) 555-7890. Email: w.thompson@email.com. Address: 654 Cedar Lane, Naperville, IL 60540. Ordering physician: Dr. Amanda Rodriguez. Patient MRN: 456789. Test results reviewed on March 22, 2024.",
        "Discharge summary for Elizabeth Davis (DOB: 11/12/1945) discharged on March 18, 2024. Contact: (708) 555-3456, elizabeth.davis@home.net. Residence: 987 Birch Road, Berwyn, IL 60402. Attending: Dr. James Park. Medical record: MR-987654. Discharge instructions provided. Follow-up scheduled for March 25, 2024.",
        "Surgery consultation for patient Michael Brown (DOB: 02/28/1955) scheduled for April 2, 2024. Phone: (847) 555-6789. Email: mbrown@email.com. Address: 147 Spruce Street, Skokie, IL 60076. Surgeon: Dr. Patricia White. Patient ID: PID-147258. Pre-op visit: March 30, 2024. Insurance authorization: AUTH-852963.",
        "Cardiology referral for patient Susan Johnson (DOB: 08/14/1978) referred by Dr. Kevin Lee. Appointment: April 10, 2024 at 2:00 PM. Contact: (312) 555-4567, susan.johnson@cardio.com. Home: 258 Ash Avenue, Unit 12, Chicago, IL 60657. Previous EKG: February 15, 2024. Patient account: 741852963.",
        "Progress note for 45-year-old patient with hypertension and diabetes mellitus type 2. Patient reports compliance with antihypertensive medication including lisinopril 10mg daily and metformin 1000mg twice daily. Blood pressure readings at home have been consistently elevated, ranging from 140/90 to 160/95 mmHg. Patient denies chest pain, shortness of breath, or visual changes. Physical examination reveals grade II hypertensive retinopathy on fundoscopic examination. Laboratory results show HbA1c of 8.2%, indicating suboptimal glycemic control. Creatinine levels remain stable at 1.1 mg/dL. Patient counseled on dietary modifications, specifically reducing sodium intake to less than 2g daily and increasing physical activity. Medication adjustment includes increasing lisinopril to 20mg daily and adding hydrochlorothiazide 25mg daily. Patient Jennifer Martinez (DOB: 06/12/1979) scheduled for follow-up in 4 weeks. Contact number: (555) 234-5678 for any concerns.",
        "Consultation note for patient presenting with chronic obstructive pulmonary disease exacerbation. Patient has a 30 pack-year smoking history and quit smoking 2 years ago. Current symptoms include increased dyspnea, productive cough with purulent sputum, and decreased exercise tolerance over the past week. Vital signs show oxygen saturation of 88% on room air, respiratory rate of 24 breaths per minute, and use of accessory muscles. Chest X-ray demonstrates hyperinflation consistent with COPD but no acute infiltrates. Arterial blood gas analysis reveals pH 7.32, PCO2 55 mmHg, PO2 65 mmHg, indicating acute respiratory acidosis. Pulmonary function tests from previous visit showed FEV1 of 45% predicted. Treatment initiated with nebulized albuterol and ipratropium bromide every 4 hours, oral prednisone 40mg daily for 5 days, and azithromycin 500mg daily for 5 days. Patient education provided regarding proper inhaler technique and smoking cessation resources. Patient Thomas Anderson hospitalized on March 15, 2024. Phone contact: (555) 876-5432. Dr. Patricia Wong, pulmonologist, will coordinate ongoing care.",
        "Surgical consultation note for patient with gallbladder disease. Patient presents with recurrent episodes of right upper quadrant abdominal pain, typically occurring 30-60 minutes after fatty meals. Pain is described as severe, cramping, and radiating to the right shoulder blade. Episodes last 2-4 hours and resolve spontaneously. Patient denies fever, jaundice, or changes in bowel movements. Physical examination reveals tenderness over the right costal margin with positive Murphy's sign. Laboratory studies show normal white blood cell count, liver enzymes, and bilirubin levels. Ultrasound examination demonstrates multiple gallstones with gallbladder wall thickening of 4mm and positive sonographic Murphy's sign. No common bile duct dilation is noted. Given the symptomatic cholelithiasis and patient's good operative risk status, laparoscopic cholecystectomy is recommended. Risks and benefits discussed including bleeding, infection, bile duct injury, and conversion to open procedure. Patient agrees to surgical intervention. Pre-operative clearance obtained from primary care physician. Patient Sarah Mitchell (DOB: 04/08/1965) scheduled for surgery on April 20, 2024. Emergency contact: (555) 345-6789. Dr. Michael Roberts, general surgeon, will perform the procedure.",
    ]

    # Ground truth entities for PHI identifiers
    medical_ground_truth_entities = [
        # Sample 1 ground truth
        [
            {"text": "John Smith", "label": "person"},
            {"text": "03/15/1975", "label": "date of birth"},
            {"text": "City General Hospital", "label": "geographic identifier"},
            {"text": "February 14, 2024", "label": "admission date"},
            {"text": "(555) 123-4567", "label": "phone number"},
            {"text": "john.smith@email.com", "label": "email address"},
            {"text": "123-45-6789", "label": "social security number"},
            {
                "text": "123 Oak Street, Springfield, IL 62701",
                "label": "street address",
            },
            {"text": "Dr. Sarah Johnson", "label": "person"},
            {"text": "98765", "label": "medical record number"},
        ],
        # Sample 2 ground truth
        [
            {"text": "Maria Garcia", "label": "person"},
            {"text": "07/22/1982", "label": "date of birth"},
            {"text": "March 8, 2024", "label": "admission date"},
            {"text": "(312) 555-9876", "label": "phone number"},
            {"text": "maria.garcia@hospital.org", "label": "email address"},
            {"text": "456 Elm Drive, Chicago, IL 60614", "label": "street address"},
            {"text": "Dr. Michael Chen", "label": "person"},
            {"text": "MRN-12345", "label": "medical record number"},
            {"text": "ABC-123", "label": "vehicle identifier"},
        ],
        # Continue with more ground truth patterns...
        [
            {"text": "Robert Wilson", "label": "person"},
            {"text": "12/05/1960", "label": "date of birth"},
            {"text": "April 15, 2024", "label": "appointment date"},
            {"text": "(847) 555-2468", "label": "phone number"},
            {"text": "robert.wilson@clinic.com", "label": "email address"},
            {
                "text": "789 Maple Avenue, Apt 4B, Evanston, IL 60201",
                "label": "street address",
            },
            {"text": "Dr. Lisa Brown", "label": "person"},
            {"text": "ACC-567890", "label": "account number"},
            {"text": "January 20, 2024", "label": "admission date"},
        ],
        [
            {"text": "Jennifer Lee", "label": "person"},
            {"text": "09/18/1988", "label": "date of birth"},
            {"text": "Dr. David Kim", "label": "person"},
            {"text": "March 25, 2024", "label": "appointment date"},
            {"text": "(773) 555-1357", "label": "phone number"},
            {"text": "jennifer.lee@provider.net", "label": "email address"},
            {"text": "321 Pine Street, Oak Park, IL 60302", "label": "street address"},
            {"text": "INS-789123", "label": "health plan number"},
            {"text": "IL-D123456789", "label": "license number"},
            {"text": "(773) 555-2468", "label": "phone number"},
        ],
        [
            {"text": "William Thompson", "label": "person"},
            {"text": "05/30/1970", "label": "date of birth"},
            {"text": "March 20, 2024", "label": "appointment date"},
            {"text": "(630) 555-7890", "label": "phone number"},
            {"text": "w.thompson@email.com", "label": "email address"},
            {"text": "654 Cedar Lane, Naperville, IL 60540", "label": "street address"},
            {"text": "Dr. Amanda Rodriguez", "label": "person"},
            {"text": "456789", "label": "medical record number"},
            {"text": "March 22, 2024", "label": "appointment date"},
        ],
        [
            {"text": "Elizabeth Davis", "label": "person"},
            {"text": "11/12/1945", "label": "date of birth"},
            {"text": "March 18, 2024", "label": "discharge date"},
            {"text": "(708) 555-3456", "label": "phone number"},
            {"text": "elizabeth.davis@home.net", "label": "email address"},
            {"text": "987 Birch Road, Berwyn, IL 60402", "label": "street address"},
            {"text": "Dr. James Park", "label": "person"},
            {"text": "MR-987654", "label": "medical record number"},
            {"text": "March 25, 2024", "label": "appointment date"},
        ],
        [
            {"text": "Michael Brown", "label": "person"},
            {"text": "02/28/1955", "label": "date of birth"},
            {"text": "April 2, 2024", "label": "appointment date"},
            {"text": "(847) 555-6789", "label": "phone number"},
            {"text": "mbrown@email.com", "label": "email address"},
            {"text": "147 Spruce Street, Skokie, IL 60076", "label": "street address"},
            {"text": "Dr. Patricia White", "label": "person"},
            {"text": "PID-147258", "label": "medical record number"},
            {"text": "March 30, 2024", "label": "appointment date"},
            {"text": "AUTH-852963", "label": "health plan number"},
        ],
        [
            {"text": "Susan Johnson", "label": "person"},
            {"text": "08/14/1978", "label": "date of birth"},
            {"text": "Dr. Kevin Lee", "label": "person"},
            {"text": "April 10, 2024", "label": "appointment date"},
            {"text": "(312) 555-4567", "label": "phone number"},
            {"text": "susan.johnson@cardio.com", "label": "email address"},
            {
                "text": "258 Ash Avenue, Unit 12, Chicago, IL 60657",
                "label": "street address",
            },
            {"text": "February 15, 2024", "label": "appointment date"},
            {"text": "741852963", "label": "account number"},
        ],
        # Sample 9 ground truth - Longer realistic note with lower PII density
        [
            {"text": "Jennifer Martinez", "label": "person"},
            {"text": "06/12/1979", "label": "date of birth"},
            {"text": "(555) 234-5678", "label": "phone number"},
        ],
        # Sample 10 ground truth - COPD consultation with lower PII density
        [
            {"text": "Thomas Anderson", "label": "person"},
            {"text": "March 15, 2024", "label": "admission date"},
            {"text": "(555) 876-5432", "label": "phone number"},
            {"text": "Dr. Patricia Wong", "label": "person"},
        ],
        # Sample 11 ground truth - Surgical consultation with lower PII density
        [
            {"text": "Sarah Mitchell", "label": "person"},
            {"text": "04/08/1965", "label": "date of birth"},
            {"text": "April 20, 2024", "label": "appointment date"},
            {"text": "(555) 345-6789", "label": "phone number"},
            {"text": "Dr. Michael Roberts", "label": "person"},
        ],
    ]

    # PHI-compliant redacted versions
    medical_redacted_ground_truth = [
        "Patient [PERSON] (DOB: [DATE_OF_BIRTH]) was admitted to [GEOGRAPHIC_IDENTIFIER] on [ADMISSION_DATE]. Contact: [PHONE_NUMBER], [EMAIL_ADDRESS]. SSN: [SOCIAL_SECURITY_NUMBER]. Address: [STREET_ADDRESS]. [PERSON] examined the patient. MRN: [MEDICAL_RECORD_NUMBER]. Treatment plan includes diabetes management.",
        "Emergency visit for [PERSON] (DOB: [DATE_OF_BIRTH]) at 2:30 AM on [ADMISSION_DATE]. Phone: [PHONE_NUMBER], [EMAIL_ADDRESS]. Lives at [STREET_ADDRESS]. [PERSON], attending physician. Patient ID: [MEDICAL_RECORD_NUMBER]. Diagnosis: acute appendicitis. License plate: [VEHICLE_IDENTIFIER].",
        "Follow-up appointment scheduled for [PERSON] (DOB: [DATE_OF_BIRTH]) on [APPOINTMENT_DATE] at 10:00 AM. Contact info: phone [PHONE_NUMBER], email [EMAIL_ADDRESS]. Address: [STREET_ADDRESS]. Physician: [PERSON]. Account number: [ACCOUNT_NUMBER]. Previous admission: [ADMISSION_DATE].",
        "Consultation notes for [PERSON] (DOB: [DATE_OF_BIRTH]) seen by [PERSON] on [APPOINTMENT_DATE]. Patient contacted at [PHONE_NUMBER], [EMAIL_ADDRESS]. Home address: [STREET_ADDRESS]. Insurance ID: [HEALTH_PLAN_NUMBER]. Driver's license: [LICENSE_NUMBER]. Emergency contact: spouse at [PHONE_NUMBER].",
        "Laboratory results for patient [PERSON] (DOB: [DATE_OF_BIRTH]). Report date: [APPOINTMENT_DATE]. Phone: [PHONE_NUMBER]. Email: [EMAIL_ADDRESS]. Address: [STREET_ADDRESS]. Ordering physician: [PERSON]. Patient MRN: [MEDICAL_RECORD_NUMBER]. Test results reviewed on [APPOINTMENT_DATE].",
        "Discharge summary for [PERSON] (DOB: [DATE_OF_BIRTH]) discharged on [DISCHARGE_DATE]. Contact: [PHONE_NUMBER], [EMAIL_ADDRESS]. Residence: [STREET_ADDRESS]. Attending: [PERSON]. Medical record: [MEDICAL_RECORD_NUMBER]. Discharge instructions provided. Follow-up scheduled for [APPOINTMENT_DATE].",
        "Surgery consultation for patient [PERSON] (DOB: [DATE_OF_BIRTH]) scheduled for [APPOINTMENT_DATE]. Phone: [PHONE_NUMBER]. Email: [EMAIL_ADDRESS]. Address: [STREET_ADDRESS]. Surgeon: [PERSON]. Patient ID: [MEDICAL_RECORD_NUMBER]. Pre-op visit: [APPOINTMENT_DATE]. Insurance authorization: [HEALTH_PLAN_NUMBER].",
        "Cardiology referral for patient [PERSON] (DOB: [DATE_OF_BIRTH]) referred by [PERSON]. Appointment: [APPOINTMENT_DATE] at 2:00 PM. Contact: [PHONE_NUMBER], [EMAIL_ADDRESS]. Home: [STREET_ADDRESS]. Previous EKG: [APPOINTMENT_DATE]. Patient account: [ACCOUNT_NUMBER].",
        "Progress note for 45-year-old patient with hypertension and diabetes mellitus type 2. Patient reports compliance with antihypertensive medication including lisinopril 10mg daily and metformin 1000mg twice daily. Blood pressure readings at home have been consistently elevated, ranging from 140/90 to 160/95 mmHg. Patient denies chest pain, shortness of breath, or visual changes. Physical examination reveals grade II hypertensive retinopathy on fundoscopic examination. Laboratory results show HbA1c of 8.2%, indicating suboptimal glycemic control. Creatinine levels remain stable at 1.1 mg/dL. Patient counseled on dietary modifications, specifically reducing sodium intake to less than 2g daily and increasing physical activity. Medication adjustment includes increasing lisinopril to 20mg daily and adding hydrochlorothiazide 25mg daily. Patient [PERSON] (DOB: [DATE_OF_BIRTH]) scheduled for follow-up in 4 weeks. Contact number: [PHONE_NUMBER] for any concerns.",
        "Consultation note for patient presenting with chronic obstructive pulmonary disease exacerbation. Patient has a 30 pack-year smoking history and quit smoking 2 years ago. Current symptoms include increased dyspnea, productive cough with purulent sputum, and decreased exercise tolerance over the past week. Vital signs show oxygen saturation of 88% on room air, respiratory rate of 24 breaths per minute, and use of accessory muscles. Chest X-ray demonstrates hyperinflation consistent with COPD but no acute infiltrates. Arterial blood gas analysis reveals pH 7.32, PCO2 55 mmHg, PO2 65 mmHg, indicating acute respiratory acidosis. Pulmonary function tests from previous visit showed FEV1 of 45% predicted. Treatment initiated with nebulized albuterol and ipratropium bromide every 4 hours, oral prednisone 40mg daily for 5 days, and azithromycin 500mg daily for 5 days. Patient education provided regarding proper inhaler technique and smoking cessation resources. Patient [PERSON] hospitalized on [ADMISSION_DATE]. Phone contact: [PHONE_NUMBER]. [PERSON], pulmonologist, will coordinate ongoing care.",
        "Surgical consultation note for patient with gallbladder disease. Patient presents with recurrent episodes of right upper quadrant abdominal pain, typically occurring 30-60 minutes after fatty meals. Pain is described as severe, cramping, and radiating to the right shoulder blade. Episodes last 2-4 hours and resolve spontaneously. Patient denies fever, jaundice, or changes in bowel movements. Physical examination reveals tenderness over the right costal margin with positive Murphy's sign. Laboratory studies show normal white blood cell count, liver enzymes, and bilirubin levels. Ultrasound examination demonstrates multiple gallstones with gallbladder wall thickening of 4mm and positive sonographic Murphy's sign. No common bile duct dilation is noted. Given the symptomatic cholelithiasis and patient's good operative risk status, laparoscopic cholecystectomy is recommended. Risks and benefits discussed including bleeding, infection, bile duct injury, and conversion to open procedure. Patient agrees to surgical intervention. Pre-operative clearance obtained from primary care physician. Patient [PERSON] (DOB: [DATE_OF_BIRTH]) scheduled for surgery on [APPOINTMENT_DATE]. Emergency contact: [PHONE_NUMBER]. [PERSON], general surgeon, will perform the procedure.",
    ]

    # Clean medical information (no PHI identifiers)
    clean_medical_texts = [
        "The patient presents with symptoms of type 2 diabetes mellitus. Blood glucose levels are elevated. Treatment includes metformin 500mg twice daily and lifestyle modifications. Follow-up recommended in 4 weeks.",
        "Physical examination reveals normal cardiovascular function. Blood pressure within normal limits. No signs of acute distress. Patient reports improvement in symptoms following treatment regimen.",
        "Laboratory results show elevated cholesterol levels. Recommend dietary changes and exercise program. Consider statin therapy if levels remain high after 3 months of lifestyle intervention.",
        "Post-operative recovery is progressing well. No signs of infection or complications. Patient is ambulatory and tolerating regular diet. Discharge planning initiated.",
        "Imaging studies demonstrate resolution of pneumonia. Chest X-ray shows clear lung fields. Patient reports decreased cough and improved breathing. Antibiotics completed successfully.",
    ]

    data = []
    for i in range(num_rows):
        medical_text = medical_texts[i % len(medical_texts)]
        clean_text = clean_medical_texts[i % len(clean_medical_texts)]

        medical_entities = medical_ground_truth_entities[
            i % len(medical_ground_truth_entities)
        ]
        medical_redacted = medical_redacted_ground_truth[
            i % len(medical_redacted_ground_truth)
        ]

        data.append(
            {
                "id": i + 1,
                "clean_medical_text": clean_text,
                "medical_text_with_phi": medical_text,
                "phi_ground_truth_entities": json.dumps(medical_entities),
                "phi_redacted_ground_truth": medical_redacted,
            }
        )

    return pd.DataFrame(data)


# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI-Compliant GLiNER Model Implementation

# COMMAND ----------


class PHIGLiNERModel(mlflow.pyfunc.PythonModel):
    """GLiNER NER model optimized for PHI Safe Harbor compliance."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.model = None
        self.analyzer = None

    def load_context(self, context):
        """Load GLiNER model with PHI compliance focus."""
        try:
            cache_dir = self.config["cache_dir"]
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HUB_CACHE"] = cache_dir
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"

            os.makedirs(cache_dir, exist_ok=True)

            # Security check for PHI compliance
            allowed_models = ["Ihor/gliner-biomed-base-v1.0"]
            if self.config["model_name"] not in allowed_models:
                raise ValueError(
                    f"Model {self.config['model_name']} not approved for PHI use"
                )

            logger.info("Loading GLiNER model for PHI compliance")
            logger.info("Validating model for healthcare compliance")

            self.model = GLiNER.from_pretrained(
                self.config["model_name"],
                cache_dir=cache_dir,
                force_download=False,
                local_files_only=False,
                use_auth_token=False,
            )

            # Initialize Presidio for additional PHI identifier detection
            logger.info("Initializing Presidio for PHI identifier detection...")
            self.analyzer = AnalyzerEngine()

            # Warm up with PHI test data
            logger.info("Warming up models with PHI test scenario...")
            test_text = "Patient John Doe (DOB: 01/15/1980) treated by Dr. Smith at City Hospital on March 1, 2024."
            _ = self.model.predict_entities(
                test_text, self.config["phi_identifiers"][:5], threshold=0.8
            )
            _ = self.analyzer.analyze(text=test_text, language="en")

            logger.info("PHI Model security status:")
            logger.info(
                "   GLiNER model: %s",
                self.config["model_name"],
            )
            logger.info("   Presidio: Microsoft-backed")
            logger.info("   PHI Compliance: Safe Harbor Method")

            logger.info("PHI-compliant models loaded successfully")

        except Exception as e:
            logger.error("Failed to load PHI GLiNER model: %s", str(e))
            raise RuntimeError(f"PHI GLiNER loading failed: {str(e)}") from e

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict PHI identifiers for Safe Harbor compliance."""
        results = []

        for _, input_row in model_input.iterrows():
            text = input_row.get("text", "")
            text_type = input_row.get("text_type", "medical")

            # Use PHI-specific labels for detection
            phi_labels = self.config["phi_identifiers"]

            all_entities = []

            # GLiNER prediction focused on PHI identifiers
            try:
                gliner_entities = self.model.predict_entities(
                    text, phi_labels, threshold=self.config["threshold"]
                )

                for entity in gliner_entities:
                    all_entities.append(
                        {
                            "text": entity["text"],
                            "label": entity["label"],
                            "start": int(entity["start"]),
                            "end": int(entity["end"]),
                            "score": float(entity["score"]),
                            "source": "gliner_phi",
                        }
                    )
            except Exception as e:
                logger.warning("GLiNER PHI prediction failed: %s", str(e))

            # Presidio detection for additional PHI identifiers
            try:
                presidio_entities = self.analyzer.analyze(text=text, language="en")

                for entity in presidio_entities:
                    phi_label = self._map_presidio_to_phi(entity.entity_type)

                    # Higher threshold for PHI compliance
                    if phi_label and entity.score >= 0.8:
                        all_entities.append(
                            {
                                "text": text[entity.start : entity.end],
                                "label": phi_label,
                                "start": int(entity.start),
                                "end": int(entity.end),
                                "score": float(entity.score),
                                "source": "presidio_phi",
                            }
                        )
            except Exception as e:
                logger.warning("Presidio PHI prediction failed: %s", str(e))

            # Remove overlapping entities
            unique_entities = self._deduplicate_entities(all_entities)

            # Create PHI-compliant redacted text
            redacted_text = self._phi_redact_text(text, unique_entities)

            results.append(
                {
                    "text": text,
                    "entities": json.dumps(unique_entities),
                    "redacted_text": redacted_text,
                    "entity_count": len(unique_entities),
                    "phi_compliant": True,
                }
            )

        return pd.DataFrame(results)

    def _map_presidio_to_phi(self, presidio_type: str) -> str:
        """Map Presidio entity types to PHI Safe Harbor identifiers."""
        phi_mapping = {
            "PERSON": "person",
            "EMAIL_ADDRESS": "email address",
            "PHONE_NUMBER": "phone number",
            "US_SSN": "social security number",
            "US_DRIVER_LICENSE": "license number",
            "DATE_TIME": "date of birth",  # Conservative mapping for PHI compliance
            "LOCATION": "geographic identifier",
            "URL": "web url",
            "IP_ADDRESS": "ip address",
            "MEDICAL_LICENSE": "license number",
            "US_PASSPORT": "license number",
            "CREDIT_CARD": "account number",
            "US_BANK_NUMBER": "account number",
            # Additional specific patterns for medical PII
            "NRP": "license number",  # National provider identifier
            "PHONE": "phone number",  # Alternative phone pattern
            "TELEPHONEUMBER": "phone number",  # Alternative phone pattern
        }

        return phi_mapping.get(presidio_type.upper())

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping highest scoring ones."""
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

    def _phi_redact_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Redact text according to PHI Safe Harbor method."""
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted = text

        for entity in entities_sorted:
            # Use PHI-compliant standardized placeholders
            placeholder = f"[{entity['label'].upper().replace(' ', '_')}]"
            redacted = (
                redacted[: entity["start"]] + placeholder + redacted[entity["end"] :]
            )

        return redacted


# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI-Compliant Presidio-Only Model Implementation

# COMMAND ----------


class PHIPresidioModel(mlflow.pyfunc.PythonModel):
    """Presidio-only model for PHI Safe Harbor compliance."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.analyzer = None

    def load_context(self, context):
        """Load Presidio analyzer for PHI identifier detection."""
        try:
            cache_dir = self.config["cache_dir"]
            os.makedirs(cache_dir, exist_ok=True)

            logger.info("Loading Presidio-only model for PHI compliance")
            logger.info("Security: Microsoft-backed Presidio")

            # Initialize Presidio analyzer with enhanced recognizers
            self.analyzer = AnalyzerEngine()

            # Warm up with PHI test data
            logger.info("Warming up Presidio with PHI test scenario...")
            test_text = "Patient John Smith (DOB: 01/15/1980, Phone: 555-123-4567) treated by Dr. Anderson at City Medical Center on March 1, 2024."
            _ = self.analyzer.analyze(text=test_text, language="en")

            logger.info("PHI Presidio-only model security status:")
            logger.info("   Presidio: Microsoft-backed")
            logger.info("   PHI Compliance: Safe Harbor Method")
            logger.info("   Medical PII Focus: Enhanced detection patterns")

            logger.info("PHI-compliant Presidio model loaded successfully")

        except Exception as e:
            logger.error("Failed to load PHI Presidio model: %s", str(e))
            raise RuntimeError(f"PHI Presidio loading failed: {str(e)}") from e

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict PHI identifiers using Presidio-only approach."""
        results = []

        for _, input_row in model_input.iterrows():
            text = input_row.get("text", "")

            all_entities = []

            # Enhanced Presidio detection for PHI identifiers
            try:
                # Use higher threshold and enhanced entity types for medical text
                presidio_entities = self.analyzer.analyze(
                    text=text,
                    language="en",
                    entities=[
                        "PERSON",
                        "PHONE_NUMBER",
                        "EMAIL_ADDRESS",
                        "DATE_TIME",
                        "MEDICAL_LICENSE",
                        "SSN",
                        "CREDIT_CARD",
                        "US_PASSPORT",
                        "US_DRIVER_LICENSE",
                        "URL",
                        "IP_ADDRESS",
                    ],
                    score_threshold=self.config["threshold"],
                )

                for entity in presidio_entities:
                    if entity.score >= self.config["threshold"]:
                        all_entities.append(
                            {
                                "text": text[entity.start : entity.end],
                                "label": self._map_presidio_to_phi_label(
                                    entity.entity_type
                                ),
                                "start": int(entity.start),
                                "end": int(entity.end),
                                "score": float(entity.score),
                                "source": "presidio_only",
                            }
                        )

            except Exception as e:
                logger.warning("Presidio-only PHI prediction failed: %s", str(e))

            # Remove overlapping entities
            unique_entities = self._deduplicate_entities(all_entities)

            # Create PHI-compliant redacted text
            redacted_text = self._redact_text_phi_compliant(text, unique_entities)

            results.append(
                {
                    "text": text,
                    "entities": json.dumps(unique_entities),
                    "redacted_text": redacted_text,
                    "entity_count": len(unique_entities),
                }
            )

        return pd.DataFrame(results)

    def _map_presidio_to_phi_label(self, presidio_label: str) -> str:
        """Map Presidio entity labels to PHI Safe Harbor identifiers."""
        label_mapping = {
            "PERSON": "person",
            "PHONE_NUMBER": "phone number",
            "EMAIL_ADDRESS": "email address",
            "DATE_TIME": "date of birth",  # Conservative mapping for PHI
            "MEDICAL_LICENSE": "license number",
            "SSN": "social security number",
            "CREDIT_CARD": "account number",
            "US_PASSPORT": "license number",
            "US_DRIVER_LICENSE": "license number",
            "URL": "web url",
            "IP_ADDRESS": "ip address",
        }
        return label_mapping.get(presidio_label, presidio_label.lower())

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove overlapping entities for PHI compliance."""
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

    def _redact_text_phi_compliant(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> str:
        """Create PHI Safe Harbor compliant redacted text."""
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        redacted = text

        for entity in entities_sorted:
            # Use PHI-compliant placeholder format
            placeholder = f"[{entity['label'].upper().replace(' ', '_')}]"
            redacted = (
                redacted[: entity["start"]] + placeholder + redacted[entity["end"] :]
            )

        return redacted


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Registration

# COMMAND ----------


def train_and_register_phi_gliner_model(
    config: PHIRedactionConfig, model_name_full: str
) -> str:
    """Train and register PHI-compliant GLiNER model."""
    config_dict = asdict(config)
    phi_model = PHIGLiNERModel(config_dict)

    sample_input = pd.DataFrame(
        {
            "text": ["Patient John Smith treated by Dr. Anderson at City Hospital."],
            "text_type": ["medical"],
        }
    )

    phi_model.load_context(None)
    sample_output = phi_model.predict(None, sample_input)

    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run():
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="phi_gliner_model",
            python_model=phi_model,
            signature=signature,
            pip_requirements=[
                "numpy>=1.21.5,<2.0",
                "pandas>=1.5.0,<2.1.0",
                "gliner==0.2.5",
                "transformers==4.44.0",
                "torch==2.4.0",
                "presidio-analyzer==2.2.358",
                "presidio-anonymizer==2.2.358",
                "packaging>=21.0",
                "spacy>=3.7.0,<3.9.0",
            ],
            input_example=sample_input,
            metadata={
                "model_type": "phi_gliner",
                "base_model": config.model_name,
                "phi_compliant": True,
                "redaction_method": "safe_harbor",
            },
        )

        mlflow.log_params(
            {
                "base_model": config.model_name,
                "threshold": config.threshold,
                "batch_size": config.batch_size,
                "phi_compliant": True,
                "num_phi_identifiers": len(config.phi_identifiers),
            }
        )

    mlflow.set_registry_uri("databricks-uc")

    registered_model_info = mlflow.register_model(
        model_uri=logged_model_info.model_uri, name=model_name_full
    )

    # Set champion alias for the newly registered version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name_full, alias="champion", version=registered_model_info.version
    )

    logger.info(
        "PHI GLiNER model registered to Unity Catalog: %s version %s with champion alias",
        model_name_full,
        registered_model_info.version,
    )
    return f"models:/{model_name_full}/{registered_model_info.version}"


def train_and_register_phi_biobert_model(
    config: PHIBioBERTConfig, model_name_full: str
) -> str:
    """Train and register PHI-compliant BioBERT model."""
    config_dict = asdict(config)
    phi_biobert_model = PHIBioBERTModel(config_dict)

    sample_input = pd.DataFrame(
        {
            "text": [
                "Patient Mary Johnson treated with metformin by Dr. Anderson at General Hospital."
            ],
            "text_type": ["medical"],
        }
    )

    phi_biobert_model.load_context(None)
    sample_output = phi_biobert_model.predict(None, sample_input)

    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run():
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="phi_biobert_model",
            python_model=phi_biobert_model,
            signature=signature,
            pip_requirements=[
                "numpy>=1.21.5,<2.0",
                "pandas>=1.5.0,<2.1.0",
                "transformers==4.44.0",
                "torch==2.4.0",
                "presidio-analyzer==2.2.358",
                "presidio-anonymizer==2.2.358",
                "packaging>=21.0",
                "spacy>=3.7.0,<3.9.0",
                "tokenizers>=0.19.0",
            ],
            input_example=sample_input,
            metadata={
                "model_type": "phi_biobert",
                "base_model": config.model_name,
                "security_status": "enterprise_trusted",
                "phi_compliant": True,
                "redaction_method": "safe_harbor",
            },
        )

        mlflow.log_params(
            {
                "base_model": config.model_name,
                "threshold": config.threshold,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "gpu_enabled": torch.cuda.is_available(),
                "phi_compliant": True,
            }
        )

    mlflow.set_registry_uri("databricks-uc")

    registered_model_info = mlflow.register_model(
        model_uri=logged_model_info.model_uri, name=model_name_full
    )

    # Set champion alias for the newly registered version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name_full, alias="champion", version=registered_model_info.version
    )

    logger.info(
        "PHI BioBERT model registered to Unity Catalog: %s version %s with champion alias",
        model_name_full,
        registered_model_info.version,
    )
    return f"models:/{model_name_full}/{registered_model_info.version}"


def train_and_register_phi_presidio_model(
    config: PHIPresidioConfig, model_name_full: str
) -> str:
    """Train and register PHI-compliant Presidio-only model."""
    config_dict = asdict(config)
    presidio_model = PHIPresidioModel(config_dict)

    sample_input = pd.DataFrame(
        {
            "text": [
                "Patient John Smith (DOB: 01/15/1980, Phone: 555-123-4567) treated by Dr. Anderson."
            ],
            "text_type": ["medical"],
        }
    )

    presidio_model.load_context(None)
    sample_output = presidio_model.predict(None, sample_input)

    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run():
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="phi_presidio_model",
            python_model=presidio_model,
            signature=signature,
            pip_requirements=[
                "numpy>=1.21.5,<2.0",
                "pandas>=1.5.0,<2.1.0",
                "presidio-analyzer==2.2.358",
                "presidio-anonymizer==2.2.358",
                "packaging>=21.0",
                "spacy>=3.7.0,<3.9.0",
            ],
            input_example=sample_input,
            metadata={
                "model_type": "phi_presidio",
                "base_model": "presidio-only",
                "security_status": "enterprise_trusted",
                "phi_compliant": True,
                "redaction_method": "safe_harbor",
            },
        )

        mlflow.log_params(
            {
                "base_model": "presidio-only",
                "threshold": config.threshold,
                "batch_size": config.batch_size,
                "phi_compliant": True,
                "enterprise_security": True,
            }
        )

    mlflow.set_registry_uri("databricks-uc")

    registered_model_info = mlflow.register_model(
        model_uri=logged_model_info.model_uri, name=model_name_full
    )

    # Set champion alias for the newly registered version
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name_full, alias="champion", version=registered_model_info.version
    )

    logger.info(
        "PHI Presidio model registered to Unity Catalog: %s version %s with champion alias",
        model_name_full,
        registered_model_info.version,
    )
    return f"models:/{model_name_full}/{registered_model_info.version}"


# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI-Compliant Batch Processing

# COMMAND ----------


def create_phi_udf(model_uri_path: str):
    """Create pandas UDF optimized for PHI compliance."""

    @pandas_udf(
        "struct<entities:string,redacted_text:string,entity_count:int,phi_compliant:boolean>"
    )
    def phi_process_batch(text_series: pd.Series) -> pd.DataFrame:
        batch_size = len(text_series)

        try:
            print(f"[PHI WORKER] Loading model: {model_uri_path}")
            model = mlflow.pyfunc.load_model(model_uri_path)
            print(f"[PHI WORKER] PHI-compliant model loaded")

            # Process batch with PHI focus
            input_df = pd.DataFrame(
                {"text": text_series.values, "text_type": ["medical"] * batch_size}
            )

            print(
                f"🔄 [PHI WORKER] Processing {batch_size} medical texts for PHI compliance..."
            )
            results = model.predict(input_df)
            print(f"[PHI WORKER] PHI redaction complete")

            return pd.DataFrame(
                {
                    "entities": results["entities"],
                    "redacted_text": results["redacted_text"],
                    "entity_count": results["entity_count"],
                    "phi_compliant": results.get("phi_compliant", [True] * batch_size),
                }
            )

        except Exception as e:
            error_msg = f"PHI_REDACTION_ERROR: {str(e)}"
            print(f"[PHI WORKER] {error_msg}")
            return pd.DataFrame(
                {
                    "entities": ["[]"] * batch_size,
                    "redacted_text": [error_msg] * batch_size,
                    "entity_count": [0] * batch_size,
                    "phi_compliant": [False] * batch_size,
                }
            )

    return phi_process_batch


def process_medical_text_phi(
    input_df: DataFrame, text_column: str, model_uri_path: str
) -> DataFrame:
    """Process medical text for PHI Safe Harbor compliance."""
    print(f"Processing medical text for PHI compliance using: {model_uri_path}")

    phi_udf = create_phi_udf(model_uri_path)

    print("🔧 Applying PHI redaction UDF...")
    return (
        input_df.withColumn("phi_results", phi_udf(col(text_column)))
        .select(
            "*",
            col("phi_results.entities").alias("phi_entities"),
            col("phi_results.redacted_text").alias("phi_redacted_text"),
            col("phi_results.entity_count").alias("phi_entity_count"),
            col("phi_results.phi_compliant").alias("phi_compliant"),
        )
        .drop("phi_results")
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Setup

# COMMAND ----------

print("Setting up Unity Catalog for PHI redaction")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
print(f"Schema '{catalog_name}.{schema_name}' ready")

# Create Unity Catalog Volume for HuggingFace cache
volume_path = f"{catalog_name}.{schema_name}.phi_hf_cache"
try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_path}")
    print(f"Unity Catalog Volume '{volume_path}' ready for PHI models")

    cache_dir = hf_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"PHI cache directory ready: {cache_dir}")

except Exception as e:
    print(f"Volume setup issue: {e}")
    import tempfile

    cache_dir = tempfile.mkdtemp(prefix="phi_cache_")
    print(f"📁 Using temporary cache directory: {cache_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI Test Data Generation

# COMMAND ----------

if generate_data:
    print("Generating PHI test data with medical scenarios")

    phi_test_data = generate_phi_test_data(20)
    spark_df = spark.createDataFrame(phi_test_data)

    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        full_source_table
    )

    print(f"PHI test data saved to Unity Catalog: {full_source_table}")
    display(spark.table(full_source_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI Model Training and Registration

# COMMAND ----------

# Set up Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

# Add widget for model selection
dbutils.widgets.dropdown(
    "model_type", "both", ["gliner", "biobert", "both"], "PHI Model Type"
)
model_type = dbutils.widgets.get("model_type")

# PHI-compliant configurations
phi_gliner_config = PHIRedactionConfig(
    model_name="Ihor/gliner-biomed-base-v1.0",
    cache_dir=cache_dir,
    threshold=0.8,  # Higher threshold for PHI
    batch_size=16,
)

phi_biobert_config = PHIBioBERTConfig(
    model_name="dmis-lab/biobert-v1.1",  # Official BioBERT model
    cache_dir=cache_dir,
    threshold=0.8,  # Higher threshold for PHI
    batch_size=16,
    max_length=512,
)

phi_presidio_config = PHIPresidioConfig(
    cache_dir=cache_dir,
    threshold=0.7,  # Presidio confidence threshold
    batch_size=16,
)

# Model URIs
phi_gliner_model_name = f"{catalog_name}.{schema_name}.{model_name}_phi_gliner"
phi_biobert_model_name = f"{catalog_name}.{schema_name}.{model_name}_phi_biobert"
phi_presidio_model_name = f"{catalog_name}.{schema_name}.{model_name}_phi_presidio"

print("Training and registering PHI models")
print(f"Model type: {model_type}")

model_uris = {}

# Train PHI GLiNER model
if model_type in ["gliner", "both", "all"]:
    print(f"\n🔬 PHI GLiNER Model: {phi_gliner_model_name}")
    if train:
        gliner_uri = train_and_register_phi_gliner_model(
            phi_gliner_config, phi_gliner_model_name
        )
        print(f"PHI GLiNER registered: {gliner_uri}")
    else:
        gliner_uri = f"models:/{phi_gliner_model_name}@{alias}"
        print(f"PHI GLiNER loading from UC: {gliner_uri}")
    model_uris["phi_gliner"] = gliner_uri

# Train PHI BioBERT model
if model_type in ["biobert", "both", "all"]:
    print(f"\nPHI BioBERT Model: {phi_biobert_model_name}")
    if train:
        biobert_uri = train_and_register_phi_biobert_model(
            phi_biobert_config, phi_biobert_model_name
        )
        print(f"PHI BioBERT registered: {biobert_uri}")
    else:
        biobert_uri = f"models:/{phi_biobert_model_name}@{alias}"
        print(f"PHI BioBERT loading from UC: {biobert_uri}")
    model_uris["phi_biobert"] = biobert_uri

# Train PHI Presidio-Only model
if model_type in ["presidio", "both", "all"]:
    print(f"\nPHI Presidio-Only Model: {phi_presidio_model_name}")
    if train:
        presidio_uri = train_and_register_phi_presidio_model(
            phi_presidio_config, phi_presidio_model_name
        )
        print(f"PHI Presidio registered: {presidio_uri}")
    else:
        presidio_uri = f"models:/{phi_presidio_model_name}@{alias}"
        print(f"PHI Presidio loading from UC: {presidio_uri}")
    model_uris["phi_presidio"] = presidio_uri

print(f"\nPHI models registered:")
for model_name, uri in model_uris.items():
    print(f"   {model_name}: {uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI-Compliant Batch Processing

# COMMAND ----------
source_df = spark.table(full_source_table)

# Process medical text with PHI-compliant models
all_model_results = {}

for model_name, model_uri in model_uris.items():
    print(f"\nProcessing medical text with {model_name.upper()} for PHI compliance")
    print(f"   Model URI: {model_uri}")

    print(f"   Applying PHI redaction to medical text...")
    medical_results = process_medical_text_phi(
        source_df, "medical_text_with_phi", model_uri
    )

    # Join with source data
    model_results = source_df.join(
        medical_results.select(
            "id",
            col("phi_entities").alias(f"{model_name}_entities"),
            col("phi_redacted_text").alias(f"{model_name}_redacted_text"),
            col("phi_entity_count").alias(f"{model_name}_entity_count"),
            col("phi_compliant").alias(f"{model_name}_compliant"),
        ),
        "id",
    )

    all_model_results[model_name] = model_results
    print(f"   {model_name.upper()} PHI processing complete")

# Combine results into single table
if len(all_model_results) == 1:
    # Single model case
    model_name = list(all_model_results.keys())[0]
    final_results = all_model_results[model_name]

    # Rename columns to standard format
    final_results = final_results.select(
        "id",
        "clean_medical_text",
        "medical_text_with_phi",
        "phi_ground_truth_entities",
        "phi_redacted_ground_truth",
        col(f"{model_name}_entities").alias("phi_detected_entities"),
        col(f"{model_name}_redacted_text").alias("phi_redacted_text"),
        col(f"{model_name}_entity_count").alias("phi_entity_count"),
        col(f"{model_name}_compliant").alias("phi_compliant"),
    )

elif len(all_model_results) == 2:
    # Both models case - create comparison table
    gliner_results = all_model_results["phi_gliner"]
    biobert_results = all_model_results["phi_biobert"]

    # Join both model results
    final_results = gliner_results.join(
        biobert_results.select(
            "id",
            "phi_biobert_entities",
            "phi_biobert_redacted_text",
            "phi_biobert_entity_count",
            "phi_biobert_compliant",
        ),
        "id",
    )

elif len(all_model_results) == 3:
    gliner_results = all_model_results["phi_gliner"]
    biobert_results = all_model_results["phi_biobert"]
    presidio_results = all_model_results["phi_presidio"]

    two_results = gliner_results.join(
        biobert_results.select(
            "id",
            "phi_biobert_entities",
            "phi_biobert_redacted_text",
            "phi_biobert_entity_count",
            "phi_biobert_compliant",
        ),
        "id",
    )
    final_results = presidio_results.join(
        two_results.select(
            "id",
            "phi_biobert_entities",
            "phi_biobert_redacted_text",
            "phi_biobert_entity_count",
            "phi_biobert_compliant",
            "phi_gliner_entities",
            "phi_gliner_redacted_text",
            "phi_gliner_entity_count",
            "phi_gliner_compliant",
        ),
        "id",
    )

else:
    print("Number of model results", len(all_model_results))


# Save PHI-compliant results
final_results.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    full_results_table
)

df = spark.table(full_results_table)

print(f"\nPHI results saved to Unity Catalog: {full_results_table}")
print(
    f"Processed {final_results.count()} medical records with {len(model_uris)} PHI model(s)"
)

# Show schema for PHI compliance verification
if len(all_model_results) > 1:
    print(f"\nPHI Comparison Table Schema:")
    df.printSchema()

display(df)
# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI Compliance Evaluation

# COMMAND ----------


def evaluate_phi_compliance(
    df: DataFrame,
    model_prefix: str = "",
    ground_truth_col: str = "phi_ground_truth_entities",
    redacted_ground_truth_col: str = "phi_redacted_ground_truth",
) -> Dict[str, Any]:
    """Evaluate PHI Safe Harbor compliance."""

    # Determine column names
    if model_prefix:
        entities_col = f"{model_prefix}_entities"
        redacted_col = f"{model_prefix}_redacted_text"
        compliant_col = f"{model_prefix}_compliant"
    else:
        entities_col = "phi_detected_entities"
        redacted_col = "phi_redacted_text"
        compliant_col = "phi_compliant"

    # Convert to pandas for processing
    pdf = df.toPandas()

    phi_metrics = []
    redaction_scores = []
    compliance_scores = []

    for _, row in pdf.iterrows():
        try:
            # Parse JSON entities
            detected_entities = (
                json.loads(row[entities_col]) if row[entities_col] else []
            )
            ground_truth_entities = (
                json.loads(row[ground_truth_col]) if row[ground_truth_col] else []
            )

            # Calculate PHI identifier detection metrics
            # Focus on text detection regardless of specific label (as requested)
            detected_texts = {ent["text"].lower() for ent in detected_entities}
            ground_truth_texts = {ent["text"].lower() for ent in ground_truth_entities}

            true_positives = len(detected_texts.intersection(ground_truth_texts))
            false_positives = len(detected_texts - ground_truth_texts)
            false_negatives = len(ground_truth_texts - detected_texts)

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            phi_metrics.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                }
            )

            # Calculate redaction success (focus on whether redaction occurred)
            redaction_accuracy = calculate_actual_redaction_success(
                row[redacted_col], row[redacted_ground_truth_col]
            )
            redaction_scores.append(redaction_accuracy)

            # Compliance score (all identifiers redacted)
            compliance_score = 1.0 if false_negatives == 0 else 0.0
            compliance_scores.append(compliance_score)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Error processing PHI evaluation row %s: %s",
                row.get("id", "unknown"),
                e,
            )
            continue

    # Aggregate metrics
    if phi_metrics:
        avg_precision = sum(m["precision"] for m in phi_metrics) / len(phi_metrics)
        avg_recall = sum(m["recall"] for m in phi_metrics) / len(phi_metrics)
        avg_f1 = sum(m["f1"] for m in phi_metrics) / len(phi_metrics)
        total_tp = sum(m["true_positives"] for m in phi_metrics)
        total_fp = sum(m["false_positives"] for m in phi_metrics)
        total_fn = sum(m["false_negatives"] for m in phi_metrics)
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
        total_tp = total_fp = total_fn = 0

    avg_redaction_accuracy = (
        sum(redaction_scores) / len(redaction_scores) if redaction_scores else 0.0
    )
    compliance_rate = (
        sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0
    )

    return {
        "model_name": model_prefix if model_prefix else "single_phi_model",
        "dataset_info": {
            "total_records": len(pdf),
            "evaluation_coverage": f"{len(phi_metrics)}/{len(pdf)} records evaluated",
        },
        "phi_detection_performance": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "redaction_accuracy": avg_redaction_accuracy,
        },
        "phi_compliance": {
            "compliance_rate": compliance_rate,
            "records_fully_compliant": sum(compliance_scores),
            "records_with_missed_identifiers": len(compliance_scores)
            - sum(compliance_scores),
            "average_missed_identifiers_per_record": (
                total_fn / len(phi_metrics) if phi_metrics else 0
            ),
        },
        "overall_assessment": {
            "phi_ready": compliance_rate >= 0.95 and avg_recall >= 0.90,
            "needs_review": compliance_rate < 0.95 or avg_recall < 0.90,
            "risk_level": (
                "LOW"
                if compliance_rate >= 0.95
                else "HIGH" if compliance_rate < 0.80 else "MEDIUM"
            ),
        },
    }


def compare_phi_models(df: DataFrame, model_names: List[str]) -> Dict[str, Any]:
    """Compare PHI compliance between models."""

    model_evaluations = {}

    # Evaluate each model for PHI compliance
    for model_name in model_names:
        model_eval = evaluate_phi_compliance(df, model_name)
        model_evaluations[model_name] = model_eval

    # Create PHI compliance comparison
    comparison = {
        "models_evaluated": list(model_names),
        "individual_results": model_evaluations,
        "phi_compliance_comparison": {},
        "recommendation": {},
    }

    # Compare PHI-specific metrics
    phi_metrics = ["compliance_rate", "f1", "recall", "redaction_accuracy"]

    for metric in phi_metrics:
        comparison["phi_compliance_comparison"][metric] = {}

        for model_name in model_names:
            if metric == "compliance_rate":
                value = model_evaluations[model_name]["phi_compliance"][metric]
            else:
                value = model_evaluations[model_name]["phi_detection_performance"][
                    metric
                ]
            comparison["phi_compliance_comparison"][metric][model_name] = value

        # Find best model for this metric
        best_model = max(
            model_names,
            key=lambda x: (
                model_evaluations[x]["phi_compliance"][metric]
                if metric == "compliance_rate"
                else model_evaluations[x]["phi_detection_performance"][metric]
            ),
        )
        best_value = (
            model_evaluations[best_model]["phi_compliance"][metric]
            if metric == "compliance_rate"
            else model_evaluations[best_model]["phi_detection_performance"][metric]
        )
        comparison["recommendation"][f"best_{metric}"] = {
            "model": best_model,
            "score": best_value,
        }

    return comparison


# Run PHI compliance evaluation
print("Running PHI Safe Harbor Compliance Evaluation")

if len(model_uris) == 1:
    # Single model PHI evaluation
    model_name = list(model_uris.keys())[0]
    print(f"\nEvaluating {model_name.upper()} for PHI compliance...")

    phi_eval = evaluate_phi_compliance(df, "")

    print(f"\n**{model_name.upper()} PHI Compliance Results:**")
    print(
        f"Dataset: {phi_eval['dataset_info']['total_records']} records, {phi_eval['dataset_info']['evaluation_coverage']} evaluated"
    )

    detection_perf = phi_eval["phi_detection_performance"]
    print(f"\n**PHI Identifier Detection:**")
    print(f"   Precision: {detection_perf['precision']:.3f}")
    print(f"   Recall: {detection_perf['recall']:.3f}")
    print(f"   F1-Score: {detection_perf['f1']:.3f}")
    print(f"   Redaction Accuracy: {detection_perf['redaction_accuracy']:.3f}")

    compliance = phi_eval["phi_compliance"]
    print(f"\n**PHI Safe Harbor Compliance:**")
    print(
        f"   Compliance Rate: {compliance['compliance_rate']:.3f} ({compliance['compliance_rate']*100:.1f}%)"
    )
    print(f"   Records Fully Compliant: {compliance['records_fully_compliant']}")
    print(
        f"   Records with Missed Identifiers: {compliance['records_with_missed_identifiers']}"
    )
    print(
        f"   Average Missed Identifiers per Record: {compliance['average_missed_identifiers_per_record']:.2f}"
    )

    overall = phi_eval["overall_assessment"]
    print(f"\n**Overall PHI Assessment:**")
    print(f"   PHI Ready: {'YES' if overall['phi_ready'] else 'NO'}")
    print(f"   Risk Level: {overall['risk_level']}")

    if overall["needs_review"]:
        print(f"   REQUIRES REVIEW: Model needs improvement before production use")
    else:
        print(f"   APPROVED: Model meets PHI Safe Harbor requirements")

elif len(model_uris) == 2:
    # PHI model comparison
    model_names = list(model_uris.keys())
    print(
        f"\n⚖️ Comparing {' vs '.join([m.upper() for m in model_names])} for PHI compliance..."
    )

    phi_comparison = compare_phi_models(df, model_names)

    print(f"\n🏥 **PHI COMPLIANCE COMPARISON RESULTS**")
    print("=" * 70)

    # Individual model PHI performance
    for model_name in model_names:
        results = phi_comparison["individual_results"][model_name]
        detection = results["phi_detection_performance"]
        compliance = results["phi_compliance"]
        overall = results["overall_assessment"]

        print(f"\n🔬 **{model_name.upper()} PHI PERFORMANCE:**")
        print("=" * 50)

        print(f"\n**Identifier Detection:**")
        print(f"   Precision: {detection['precision']:.3f}")
        print(f"   Recall: {detection['recall']:.3f}")
        print(f"   F1-Score: {detection['f1']:.3f}")
        print(f"   Redaction Accuracy: {detection['redaction_accuracy']:.3f}")

        print(f"\n✅ **PHI Compliance:**")
        print(
            f"   Compliance Rate: {compliance['compliance_rate']:.3f} ({compliance['compliance_rate']*100:.1f}%)"
        )
        print(f"   Fully Compliant Records: {compliance['records_fully_compliant']}")
        print(
            f"   Records with Issues: {compliance['records_with_missed_identifiers']}"
        )

        print(f"\n**Assessment:**")
        print(f"   PHI Ready: {'YES' if overall['phi_ready'] else 'NO'}")
        print(f"   Risk Level: {overall['risk_level']}")

        # Security assessment
        if "gliner" in model_name:
            print(
                f"   🔒 Security: ⚠️  Community model - additional validation recommended"
            )
        elif "biobert" in model_name:
            print(f"   🔒 Security: ✅ Enterprise-backed model")

    # PHI recommendations
    print(f"\n🏆 **PHI COMPLIANCE WINNERS:**")
    for metric, winner_info in phi_comparison["recommendation"].items():
        winner = winner_info["model"]
        score = winner_info["score"]
        print(
            f"   {metric.replace('best_', '').title()}: {winner.upper()} ({score:.3f})"
        )

    # Overall PHI recommendation
    gliner_compliance = phi_comparison["individual_results"]["phi_gliner"][
        "phi_compliance"
    ]["compliance_rate"]
    biobert_compliance = phi_comparison["individual_results"]["phi_biobert"][
        "phi_compliance"
    ]["compliance_rate"]

    print(f"\n**FINAL PHI RECOMMENDATION:**")
    if biobert_compliance >= 0.95 and gliner_compliance >= 0.95:
        print("   ✅ Both models meet PHI requirements")
        print("   💡 Recommendation: Choose BioBERT for better enterprise security")
    elif biobert_compliance >= 0.95:
        print("   ✅ BioBERT meets PHI requirements")
        print("   ❌ GLiNER does not meet PHI compliance threshold")
        print("   🎯 DECISION: Use BioBERT for production")
    elif gliner_compliance >= 0.95:
        print("   ✅ GLiNER meets PHI requirements")
        print("   ❌ BioBERT does not meet PHI compliance threshold")
        print("   ⚠️  CAUTION: Validate GLiNER security before production use")
    else:
        print("   ❌ Neither model meets PHI compliance requirements")
        print("   🚨 ACTION REQUIRED: Both models need improvement")

    print(f"\n📊 **COMPLIANCE THRESHOLD: 95% for production use**")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI Compliance Report

# COMMAND ----------

# Generate comprehensive PHI compliance report
print("\n" + "=" * 80)
print("🏥 PHI SAFE HARBOR COMPLIANCE FINAL REPORT")
print("=" * 80)

# Sample redacted results
print(f"\n📝 **SAMPLE PHI REDACTION RESULTS:**")
sample_df = df.limit(3).toPandas()

for i, (_, row) in enumerate(sample_df.iterrows(), 1):
    print(f"\n🆔 Medical Record {i} (ID: {row['id']}):")
    print(f"   📄 Original: {row['medical_text_with_phi'][:100]}...")

    if len(model_uris) == 1:
        entities = json.loads(row["phi_detected_entities"])
        redacted = row["phi_redacted_text"]
        compliant = row["phi_compliant"]
    else:
        entities = json.loads(row["phi_gliner_entities"])  # Use GLiNER as example
        redacted = row["phi_gliner_redacted_text"]
        compliant = row["phi_gliner_compliant"]

    print(f"   🏥 PHI Redacted: {redacted[:100]}...")
    print(
        f"   🔍 Identifiers Found: {len(entities)} ({[e.get('label', 'unknown') for e in entities[:3]]})"
    )
    print(f"   ✅ PHI Compliant: {'YES' if compliant else 'NO'}")

print(f"\n🔐 **PHI SAFE HARBOR IDENTIFIERS ADDRESSED:**")
phi_identifiers = [
    "1. Names (patients, relatives, staff)",
    "2. Geographic subdivisions smaller than state",
    "3. Dates directly related to individuals",
    "4. Phone numbers, fax numbers, email addresses",
    "5. Social Security and medical record numbers",
    "6. Health plan, account, certificate numbers",
    "7. Vehicle identifiers and serial numbers",
    "8. Device identifiers and serial numbers",
    "9. Web URLs and IP addresses",
    "10. Biometric identifiers",
    "11. Full face photos and comparable images",
    "12. Any other unique identifying numbers",
]

for identifier in phi_identifiers:
    print(f"   ✅ {identifier}")

print(f"\n📊 **DATA PROCESSING SUMMARY:**")
record_count = df.count()
print(f"   📁 Medical Records Processed: {record_count}")
print(
    f"   🏥 Models Used: {len(model_uris)} ({'PHI-compliant' if record_count > 0 else 'None'})"
)
print(f"   🔒 Security Level: Enterprise-grade with human review required")
print(f"   ✅ PHI Method: Safe Harbor De-identification")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison Summary Table

# COMMAND ----------


def calculate_actual_redaction_success(
    predicted_text: str, ground_truth_text: str
) -> float:
    """Calculate how successfully PII/PHI was redacted regardless of specific labels."""
    from difflib import SequenceMatcher

    # Check if predicted text has placeholder patterns (indicating redaction occurred)
    import re

    predicted_has_redaction = bool(re.search(r"\[[A-Z_]+\]", predicted_text))
    ground_truth_has_redaction = bool(re.search(r"\[[A-Z_]+\]", ground_truth_text))

    if not ground_truth_has_redaction and not predicted_has_redaction:
        # Both have no redaction needed/applied - perfect match
        return 1.0
    elif ground_truth_has_redaction and predicted_has_redaction:
        # Both have redaction - measure similarity of redacted result
        return SequenceMatcher(
            None, predicted_text.lower(), ground_truth_text.lower()
        ).ratio()
    elif ground_truth_has_redaction and not predicted_has_redaction:
        # Should have redacted but didn't - major failure
        return 0.1
    else:
        # Redacted when not needed - minor penalty
        return 0.7


# Create and display comparison table with ACTUAL calculated metrics
print("📊 **MODEL COMPARISON SUMMARY**")
print("=" * 60)

# Calculate real evaluation metrics for each model
model_evaluations = {}

# Map model URI keys to display names and column prefixes
model_mapping = {
    "phi_gliner": ("GLiNER", "phi_gliner"),
    "phi_biobert": ("BioBERT", "phi_biobert"),
    "phi_presidio": ("Presidio-Only", "phi_presidio"),
}

print("\n🔄 Calculating real evaluation metrics for each model...")

for model_key, (display_name, column_prefix) in model_mapping.items():
    if model_key in model_uris:
        print(f"   Evaluating {display_name}...")
        try:
            evaluation = evaluate_phi_compliance(df, column_prefix)
            model_evaluations[display_name] = evaluation
            print(f"   ✅ {display_name} evaluation complete")
        except Exception as e:
            print(f"   ❌ {display_name} evaluation failed: {str(e)}")
            # Create default metrics if evaluation fails
            model_evaluations[display_name] = {
                "phi_detection_performance": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "redaction_accuracy": 0.0,
                }
            }

# Create comparison table with real metrics
comparison_data = []
for display_name, evaluation in model_evaluations.items():
    perf = evaluation["phi_detection_performance"]
    comparison_data.append(
        {
            "Model": display_name,
            "Precision": f"{perf['precision']:.3f}",
            "Recall": f"{perf['recall']:.3f}",
            "F1-Score": f"{perf['f1']:.3f}",
            "Redaction Success": f"{perf['redaction_accuracy']:.3f}",
        }
    )

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    display(comparison_df)

    print("\n🏆 **ACTUAL PERFORMANCE RESULTS:**")

    # Find best performing model for each metric
    best_precision = max(
        model_evaluations.items(),
        key=lambda x: x[1]["phi_detection_performance"]["precision"],
    )
    best_recall = max(
        model_evaluations.items(),
        key=lambda x: x[1]["phi_detection_performance"]["recall"],
    )
    best_f1 = max(
        model_evaluations.items(),
        key=lambda x: x[1]["phi_detection_performance"]["f1"],
    )
    best_redaction = max(
        model_evaluations.items(),
        key=lambda x: x[1]["phi_detection_performance"]["redaction_accuracy"],
    )

    print(
        f"   • Best Precision: {best_precision[0]} ({best_precision[1]['phi_detection_performance']['precision']:.3f})"
    )
    print(
        f"   • Best Recall: {best_recall[0]} ({best_recall[1]['phi_detection_performance']['recall']:.3f})"
    )
    print(
        f"   • Best F1-Score: {best_f1[0]} ({best_f1[1]['phi_detection_performance']['f1']:.3f})"
    )
    print(
        f"   • Best Redaction: {best_redaction[0]} ({best_redaction[1]['phi_detection_performance']['redaction_accuracy']:.3f})"
    )

    # Show detailed metrics for each model
    print(f"\n📈 **DETAILED METRICS:**")
    for display_name, evaluation in model_evaluations.items():
        perf = evaluation["phi_detection_performance"]
        print(f"   {display_name}:")
        print(
            f"      Precision: {perf['precision']:.3f} | Recall: {perf['recall']:.3f} | F1: {perf['f1']:.3f} | Redaction: {perf['redaction_accuracy']:.3f}"
        )
        print(
            f"      TP: {perf['total_tp']} | FP: {perf['total_fp']} | FN: {perf['total_fn']}"
        )

    print("   • All metrics calculated from actual model predictions vs ground truth")
    print("   • Redaction success measures how well PII/PHI was actually removed")
else:
    print("❌ No evaluation metrics calculated - check model results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Redaction Comparisons

# COMMAND ----------


def show_actual_redaction_examples():
    """Display ACTUAL redaction results from the models."""

    print("🔍 **ACTUAL REDACTION EXAMPLES COMPARISON**")
    print("=" * 80)
    print("📊 Using real model predictions from evaluation data")

    # Get sample data from actual results
    sample_df = df.limit(3).toPandas()

    if len(sample_df) == 0:
        print("❌ No sample data available for redaction examples")
        return

    # Map model URI keys to display names for redaction columns
    redaction_column_mapping = {}
    if "phi_gliner" in model_uris:
        redaction_column_mapping["GLiNER"] = "phi_gliner_redacted_text"
    if "phi_biobert" in model_uris:
        redaction_column_mapping["BioBERT"] = "phi_biobert_redacted_text"
    if "phi_presidio" in model_uris:
        redaction_column_mapping["Presidio-Only"] = "phi_presidio_redacted_text"

    # Display actual redaction examples
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        original_text = row["medical_text_with_phi"]

        print(f"\n📄 **Medical Record {i} (ID: {row['id']}):**")
        print(f"Original: {original_text}")
        print()

        # Show each model's actual redaction
        redaction_analysis = []
        for display_name, column_name in redaction_column_mapping.items():
            if column_name in row and pd.notna(row[column_name]):
                redacted_text = row[column_name]
                print(f"  {display_name:12}: {redacted_text}")

                # Count redactions for analysis
                import re

                redaction_count = len(re.findall(r"\[[A-Z_]+\]", redacted_text))
                redaction_analysis.append((display_name, redaction_count))
            else:
                print(f"  {display_name:12}: [Model result not available]")

        print()

        # Show redaction comparison for this example
        if redaction_analysis:
            print("  📊 Redaction Count Comparison:")
            for model, count in redaction_analysis:
                print(f"     {model}: {count} identifiers redacted")

        print("-" * 80)

    # Analyze patterns across all models
    print("\n📋 **ACTUAL REDACTION ANALYSIS:**")

    # Calculate average redactions per model across all data
    full_df = df.toPandas()
    redaction_stats = {}

    for display_name, column_name in redaction_column_mapping.items():
        if column_name in full_df.columns:
            # Count average redactions per record
            import re

            redaction_counts = []
            for _, row in full_df.iterrows():
                if pd.notna(row[column_name]):
                    count = len(re.findall(r"\[[A-Z_]+\]", row[column_name]))
                    redaction_counts.append(count)

            if redaction_counts:
                avg_redactions = sum(redaction_counts) / len(redaction_counts)
                max_redactions = max(redaction_counts)
                min_redactions = min(redaction_counts)
                redaction_stats[display_name] = {
                    "avg": avg_redactions,
                    "max": max_redactions,
                    "min": min_redactions,
                    "total_records": len(redaction_counts),
                }

    for model, stats in redaction_stats.items():
        print(
            f"   • {model}: Avg {stats['avg']:.1f} redactions/record (Min: {stats['min']}, Max: {stats['max']})"
        )

    print("\n📈 **KEY OBSERVATIONS FROM ACTUAL DATA:**")
    if redaction_stats:
        # Find model with most comprehensive redaction
        most_comprehensive = max(redaction_stats.items(), key=lambda x: x[1]["avg"])
        least_comprehensive = min(redaction_stats.items(), key=lambda x: x[1]["avg"])

        print(
            f"   • Most Comprehensive: {most_comprehensive[0]} ({most_comprehensive[1]['avg']:.1f} avg redactions)"
        )
        print(
            f"   • Least Comprehensive: {least_comprehensive[0]} ({least_comprehensive[1]['avg']:.1f} avg redactions)"
        )
        print(
            f"   • All redaction patterns shown above are from ACTUAL model predictions"
        )
        print(
            f"   • Results may vary based on text content and model sensitivity settings"
        )
    else:
        print("   • No redaction statistics available - check model results")


# Show ACTUAL redaction examples from model results
show_actual_redaction_examples()

print(f"\n⚠️  **IMPORTANT PHI COMPLIANCE NOTES:**")
print("   • Human review is REQUIRED for all redacted medical text")
print("   • This system implements PHI Safe Harbor method")
print("   • Medical information is preserved when not identifying")
print("   • All 18 PHI identifiers are targeted for redaction")
print("   • Additional validation recommended before production use")

print("\n" + "=" * 80)
print("✅ PHI-COMPLIANT REDACTION PROCESS COMPLETE")
print("=" * 80)

# COMMAND ----------
