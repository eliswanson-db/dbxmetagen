import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult, Pattern
import spacy
from spacy.cli import download
from datetime import datetime

from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.parsing import sanitize_email


def get_analyzer_engine(add_pci: bool = True, add_phi: bool = True) -> AnalyzerEngine:
    """
    Initialize Presidio AnalyzerEngine, optionally adding PCI/PHI recognizers.
    """
    analyzer = AnalyzerEngine()
    if add_pci:
        # TODO: Need to build a library of specific recognizers for PCI.
        pci_patterns = [
            Pattern(name="credit_card_pattern", regex=r"\b(?:\d[ -]*?){13,16}\b", score=0.8)
        ]
        pci_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD",
            patterns=pci_patterns,
            context=["credit", "card", "visa", "mastercard", "amex"]
        )
        analyzer.registry.add_recognizer(pci_recognizer)
    if add_phi:
        # TODO: Need to build a library of recognizers for medical information and PHI.
        mrn_pattern = Pattern(
            name="mrn_pattern",
            regex=r"\bMRN[:\s]*\d{6,10}\b",
            score=0.8
        )
        phi_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=[mrn_pattern],
            context=["mrn", "medical", "record"]
        )
        analyzer.registry.add_recognizer(phi_recognizer)
    return analyzer

def analyze_column(
    analyzer: AnalyzerEngine,
    column_data: List[Any],
    entities: Optional[List[str]] = None,
    language: str = "en"
) -> List[List[RecognizerResult]]:
    """
    Analyze each cell in a column for PII/PHI/PCI entities.
    """
    results = []
    for cell in column_data:
        try:
            text = str(cell) if cell is not None else ""
            analysis = analyzer.analyze(text=text, language=language, entities=entities)
            results.append(analysis)
        except Exception as e:
            logging.error(f"Error analyzing cell '{cell}': {e}")
            results.append([])
    return results

def classify_column(
    analyzer: AnalyzerEngine,
    column_name: str,
    column_data: List[Any]
) -> Tuple[str, List[str]]:
    """
    Classify a column as PII, PHI, PCI, or Non-sensitive.
    Returns the detected type and a list of detected entities.
    """
    entity_map = {
        "PII": [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS", "DATE_TIME", "NRP",
            "LOCATION", "IP_ADDRESS", "URL", "CREDIT_CARD", "IBAN_CODE", "CRYPTO"
        ],
        "US": [
            "US_SSN", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_PASSPORT", "US_ITIN"
        ],
        "International": [
            "UK_NHS", "UK_NINO", "IT_FISCAL_CODE", "IT_DRIVER_LICENSE", "IT_VAT_CODE",
            "IT_PASSPORT", "IT_IDENTITY_CARD", "ES_NIF", "ES_NIE", "PL_PESEL", "SG_NRIC_FIN",
            "SG_UEN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE", "IN_PAN", "IN_AADHAAR",
            "IN_VEHICLE_REGISTRATION", "IN_VOTER_ID", "IN_PASSPORT", "FI_PERSONAL_ID"
        ],
        "PHI": [
            "MEDICAL_LICENSE", "MEDICAL_RECORD_NUMBER", "HEALTH_INSURANCE_NUMBER"
        ],
        "PCI": [
            "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE"
        ]
    }
    detected_types = set()
    detected_entities = set()
    results = analyze_column(analyzer, column_data)
    for cell_results in results:
        for res in cell_results:
            for typ, ents in entity_map.items():
                if res.entity_type in ents:
                    detected_types.add(typ)
                    detected_entities.add(res.entity_type)
    if not detected_types:
        return "Non-sensitive", []
    return ", ".join(sorted(detected_types)), sorted(detected_entities)

def process_table(
    config: MetadataConfig,
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Process the input data, classify each column, and save results.
    """
    current_date = datetime.now().strftime("%Y%m%d")
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"/Volumes/{config.catalog_name}/{config.schema_name}/generated_metadata/{sanitize_email(config.current_user)}/{current_date}/presidio_logs"

    os.makedirs(output_dir, exist_ok=True)
    analyzer = get_analyzer_engine()
    column_contents = data.get("column_contents", {})
    columns = column_contents.get("columns", [])
    data_rows = column_contents.get("data", [])
    results = []
    for idx, col in enumerate(columns):
        column_data = [row[idx] for row in data_rows]
        col_type, entities = classify_column(analyzer, col, column_data)
        results.append({
            "column": col,
            "classification": col_type,
            "entities": entities
        })
        logging.info(f"Column '{col}' classified as '{col_type}' with entities: {entities}")
    output_path = os.path.join(output_dir, f"presidio_column_classification_results_{current_timestamp}.txt")
    try:
        with open(output_path, "w") as f:
            for res in results:
                f.write(f"{res['column']}: {res['classification']} ({', '.join(res['entities'])})\n")
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
    return results

def ensure_spacy_model(model_name: str = "en_core_web_lg"):
    """
    Ensure the specified spaCy model is installed and loaded. By default using the designated english core.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        download(model_name)
        return spacy.load(model_name)

def detect_pi(config, input_data: Dict[str, Any]) -> str:
    """
    Main function to process input data for PII/PHI/PCI detection.
    """
    #setup_logging()
    logging.info("Starting PII/PHI/PCI detection process.")
    results = process_table(config, input_data)
    return json.dumps({"deterministic_results": results})
    logging.info("Detection process completed.")