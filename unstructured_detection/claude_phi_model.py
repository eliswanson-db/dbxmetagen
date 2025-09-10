"""
Claude PHI detection model using ChatDatabricks for simplified authentication.
This file contains the PHI detection model implementation using Claude Sonnet
via Databricks Foundation Models API with automatic authentication.
"""

import json
import re
import pandas as pd
import logging
from typing import Dict, Any, List
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model

logger = logging.getLogger(__name__)


class PHIClaudeModel(PythonModel):
    """Claude Sonnet model for PHI detection via Databricks Foundation Models."""

    def __init__(self):
        self.chat_model = None

    def load_context(self, context):
        """Load model artifacts and configuration."""
        _ = context  # Unused but required by MLflow interface
        # Lazy import - only import when actually needed (avoids import-time dependency issues)
        try:
            from databricks_langchain import ChatDatabricks
        except ImportError:
            from langchain_community.chat_models import ChatDatabricks

        # Initialize ChatDatabricks - handles authentication automatically
        self.chat_model = ChatDatabricks(
            endpoint="databricks-claude-3-7-sonnet", max_tokens=2000, temperature=0.1
        )
        logger.info(
            "Claude PHI model loaded with ChatDatabricks - authentication handled automatically"
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Predict PHI identifiers using Claude Sonnet."""
        _ = context  # Unused but required by MLflow interface
        print("[CLAUDE] Starting prediction process...")
        results = []

        # Use default config - can be overridden by passing config during training
        threshold = 0.8
        max_tokens = 4000

        print(f"[CLAUDE] Config - threshold: {threshold}, max_tokens: {max_tokens}")

        phi_prompt = """You are a PHI detection system. Analyze the following medical text and identify all personally identifiable information (PHI) that must be redacted for HIPAA compliance.

Return a JSON list of entities with this format:
[{{"text": "found_text", "label": "phi_category", "start": start_pos, "end": end_pos, "score": 0.9}}]

PHI categories to detect:
- person (names)
- phone number
- email address
- social security number
- medical record number
- date of birth
- street address
- geographic identifier

Text to analyze: {text}

Return only the JSON list, no additional text. It needs to be a list of dictionaries, with the exact keys referenced."""

        print(f"[CLAUDE] Processing {len(model_input)} input rows...")

        for row_idx, (_, input_row) in enumerate(model_input.iterrows()):
            print(f"[CLAUDE] Processing row {row_idx + 1}/{len(model_input)}")

            text = input_row.get("text", "")

            if len(text) > max_tokens:
                text = text[:max_tokens]

            print(f"[CLAUDE DEBUG] Input text length: {len(text)}")
            print(f"[CLAUDE DEBUG] Input text sample: {text[:100]}...")

            all_entities = []

            # Use ChatDatabricks - handles authentication automatically!
            try:
                response = self.chat_model.invoke(phi_prompt.format(text=text))
                response_text = response.content
                print(f"[CLAUDE DEBUG] Raw Claude response: {response_text}")
                print(f"[CLAUDE DEBUG] Response length: {len(response_text)}")
            except (ConnectionError, TimeoutError, ValueError) as e:
                print(f"[CLAUDE DEBUG] ❌ ChatDatabricks call failed: {e}")
                response_text = "[]"  # Empty response on failure

            # Parse JSON response - let it fail if malformed
            try:
                entities_data = json.loads(response_text)
                print(
                    f"[CLAUDE DEBUG] ✅ JSON parsing successful: {type(entities_data)}"
                )
                print(f"[CLAUDE DEBUG] Entities data: {entities_data}")
            except json.JSONDecodeError as e:
                print(f"[CLAUDE DEBUG] ❌ Direct JSON parsing failed: {e}")
                print("[CLAUDE DEBUG] Attempting regex extraction...")

                # Fallback to regex extraction
                json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if json_match:
                    entities_data = json.loads(json_match.group(0))
                    print("[CLAUDE DEBUG] ✅ Regex extraction successful")
                else:
                    print("[CLAUDE DEBUG] ❌ No JSON array found in response")
                    entities_data = []

            # Process entities with validation
            print("[CLAUDE DEBUG] Processing entities...")
            if isinstance(entities_data, list):
                print(f"[CLAUDE DEBUG] Found {len(entities_data)} potential entities")
                for i, entity_data in enumerate(entities_data):
                    print(f"[CLAUDE DEBUG] Entity {i}: {entity_data}")

                    if (
                        isinstance(entity_data, dict)
                        and "text" in entity_data
                        and "label" in entity_data
                        and entity_data.get("score", 0) >= threshold
                    ):
                        print(f"[CLAUDE DEBUG] ✅ Entity {i} passed validation")

                        # Ensure start/end positions exist and are valid
                        if "start" not in entity_data or "end" not in entity_data:
                            print(
                                f"[CLAUDE DEBUG] Finding missing positions for: {entity_data['text']}"
                            )
                            # Find positions in text if missing
                            entity_text = entity_data["text"]
                            start_pos = text.lower().find(entity_text.lower())
                            if start_pos != -1:
                                entity_data["start"] = start_pos
                                entity_data["end"] = start_pos + len(entity_text)
                                print(
                                    f"[CLAUDE DEBUG] Found positions: start={start_pos}, end={start_pos + len(entity_text)}"
                                )
                            else:
                                print(
                                    "[CLAUDE DEBUG] ❌ Could not find entity text in input"
                                )
                                continue  # Skip if we can't find the text

                        all_entities.append(entity_data)
                    else:
                        print(f"[CLAUDE DEBUG] ❌ Entity {i} failed validation")
            else:
                print(
                    f"[CLAUDE DEBUG] ❌ Entities data is not a list: {type(entities_data)}"
                )

            print(f"[CLAUDE DEBUG] Final entities count: {len(all_entities)}")

            redacted_text = self._phi_redact_text(text, all_entities)
            print(f"[CLAUDE DEBUG] Redacted text: {redacted_text}")

            result_entry = {
                "text": text,
                "entities": json.dumps(all_entities),
                "redacted_text": redacted_text,
                "entity_count": len(all_entities),
                "phi_compliant": True,
            }

            results.append(result_entry)
            print(f"[CLAUDE DEBUG] Result entry for row {row_idx}: {result_entry}")

        print(f"[CLAUDE DEBUG] Final results count: {len(results)}")
        final_df = pd.DataFrame(results)
        print(f"[CLAUDE DEBUG] Final DataFrame shape: {final_df.shape}")
        print(f"[CLAUDE DEBUG] Final DataFrame columns: {final_df.columns.tolist()}")

        return final_df

    def _phi_redact_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Create PHI-compliant redacted text."""
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


# Specify which definition in this script represents the model instance
set_model(PHIClaudeModel())
