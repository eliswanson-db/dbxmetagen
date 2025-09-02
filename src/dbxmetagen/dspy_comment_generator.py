"""
DSPy-powered CommentGeneratorModel that inherits from CommentGenerator.

This module provides a DSPy-optimized version of the comment generation functionality
while maintaining full compatibility with the existing CommentGenerator interface.
It's designed to be used alongside the existing system without impacting it.
"""

import json
from typing import Dict, Any

from src.dbxmetagen.metadata_generator import CommentGenerator, CommentResponse
from src.dbxmetagen.config import MetadataConfig

# Try to import optional dependencies
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None


class CommentSignature(dspy.Signature):
    """Generate table and column comments for database metadata."""

    # Input fields
    table_name: str = dspy.InputField(desc="Full table name (catalog.schema.table)")
    table_data: str = dspy.InputField(
        desc="JSON representation of table data with columns and sample rows"
    )
    column_metadata: str = dspy.InputField(
        desc="JSON metadata for columns including data types, null counts, etc."
    )
    abbreviations: str = dspy.InputField(
        desc="JSON containing abbreviations and their expansions"
    )

    # Output field
    metadata_response: str = dspy.OutputField(
        desc="JSON dictionary with 'table' (table description), 'columns' (list of column names), and 'column_contents' (list of column descriptions)"
    )


class CommentNoDataSignature(dspy.Signature):
    """Generate table and column comments for database metadata without including actual data values in the output."""

    # Input fields
    table_name: str = dspy.InputField(desc="Full table name (catalog.schema.table)")
    table_data: str = dspy.InputField(
        desc="JSON representation of table structure with columns and sample rows"
    )
    column_metadata: str = dspy.InputField(
        desc="JSON metadata for columns including data types, null counts, etc."
    )
    abbreviations: str = dspy.InputField(
        desc="JSON containing abbreviations and their expansions"
    )

    # Output field
    metadata_response: str = dspy.OutputField(
        desc="JSON dictionary with 'table' (table description), 'columns' (list of column names), and 'column_contents' (list of column descriptions) - no actual data values should be included"
    )


class DSPyCommentModule(dspy.Module):
    """DSPy module for generating table and column comments."""

    def __init__(self, allow_data_in_comments=True):
        super().__init__()
        self.allow_data_in_comments = allow_data_in_comments

        if allow_data_in_comments:
            self.generate_comments = dspy.ChainOfThought(CommentSignature)
        else:
            self.generate_comments = dspy.ChainOfThought(CommentNoDataSignature)

    def forward(self, table_name, table_data, column_metadata, abbreviations):
        """Generate metadata comments using DSPy."""

        # Convert inputs to appropriate format
        table_data_str = (
            json.dumps(table_data) if isinstance(table_data, dict) else str(table_data)
        )
        column_metadata_str = (
            json.dumps(column_metadata)
            if isinstance(column_metadata, dict)
            else str(column_metadata)
        )
        abbreviations_str = (
            json.dumps(abbreviations)
            if isinstance(abbreviations, dict)
            else str(abbreviations)
        )

        # Generate response using DSPy
        result = self.generate_comments(
            table_name=table_name,
            table_data=table_data_str,
            column_metadata=column_metadata_str,
            abbreviations=abbreviations_str,
        )

        return result


class DSPyCommentGeneratorModel(
    CommentGenerator, mlflow.pyfunc.PythonModel if MLFLOW_AVAILABLE else object
):
    """
    DSPy-powered CommentGeneratorModel that inherits from CommentGenerator.

    This class combines DSPy's prompt optimization capabilities with the existing
    CommentGenerator interface. It maintains full compatibility while providing
    enhanced prompt engineering through DSPy's optimization framework.
    """

    def __init__(self, config: MetadataConfig = None):
        """Initialize the DSPy Comment Generator Model."""
        self.config = config
        self.dspy_module = None
        self._initialized = False
        self.chat_client = None
        self.chat_response = None

    def load_context(self, context):
        """MLflow pyfunc load_context method."""
        # Initialize DSPy if not already done
        if not self._initialized:
            self._initialize_dspy()
        # Context parameter is required by MLflow interface but not used

    def _initialize_dspy(self):
        """Initialize DSPy configuration and modules."""
        if self._initialized:
            return

        # Configure DSPy to use OpenAI
        if hasattr(self.config, "model") and hasattr(self.config, "max_tokens"):
            dspy.settings.configure(
                lm=dspy.OpenAI(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=getattr(self.config, "temperature", 0.7),
                )
            )
        else:
            # Default configuration
            dspy.settings.configure(lm=dspy.OpenAI())

        # Initialize DSPy module
        allow_data = getattr(self.config, "allow_data_in_comments", True)
        self.dspy_module = DSPyCommentModule(allow_data_in_comments=allow_data)
        self._initialized = True

    def from_context(self, config):
        """Override from_context to initialize DSPy."""
        # Call parent method to maintain compatibility
        super().from_context(config)

        # Initialize DSPy
        self._initialize_dspy()

    def predict(self, model_input, params=None):
        """
        MLflow pyfunc predict method.

        Args:
            model_input: Input data containing table information
            params: Optional parameters

        Returns:
            Chat completion response
        """
        if not self._initialized:
            self._initialize_dspy()

        # Handle config serialization
        if isinstance(self.config, dict):
            config_dict = self.config
        else:
            config_dict = self.config.__dict__

        # Extract input data
        if isinstance(model_input, list) and len(model_input) > 0:
            # Handle chat-style input
            content = self._extract_content_from_messages(model_input)
        else:
            content = model_input

        # Generate response using DSPy
        try:
            # Parse the content to extract components
            parsed_content = self._parse_input_content(content)

            # Use DSPy module to generate response
            result = self.dspy_module(
                table_name=parsed_content.get("table_name", ""),
                table_data=parsed_content.get("table_data", {}),
                column_metadata=parsed_content.get("column_metadata", {}),
                abbreviations=parsed_content.get("abbreviations", {}),
            )

            # Convert DSPy output to expected format
            response_content = result.metadata_response

            # Create mock ChatCompletion response for compatibility
            chat_response = self._create_chat_completion_response(response_content)

            return chat_response

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"DSPy generation failed, falling back to parent method: {e}")
            # Fallback to parent implementation
            return self._fallback_predict(model_input, params or {})

    def predict_chat_response(self, prompt_content):
        """
        Override predict_chat_response to use DSPy.

        Args:
            prompt_content: Prompt messages

        Returns:
            Structured chat response
        """
        if not self._initialized:
            self._initialize_dspy()

        try:
            # Extract content from prompt
            content = self._extract_content_from_messages(prompt_content)
            parsed_content = self._parse_input_content(content)

            # Use DSPy module
            result = self.dspy_module(
                table_name=parsed_content.get("table_name", ""),
                table_data=parsed_content.get("table_data", {}),
                column_metadata=parsed_content.get("column_metadata", {}),
                abbreviations=parsed_content.get("abbreviations", {}),
            )

            # Parse and validate the response
            response_dict = json.loads(result.metadata_response)

            # Create structured response
            comment_response = CommentResponse(
                table=response_dict.get("table", ""),
                columns=response_dict.get("columns", []),
                column_contents=response_dict.get("column_contents", []),
            )

            return comment_response

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"DSPy structured response failed, falling back: {e}")
            # Fallback to parent method
            return super().predict_chat_response(prompt_content)

    def _extract_content_from_messages(self, messages):
        """Extract content from chat messages."""
        if isinstance(messages, list):
            for message in reversed(messages):  # Start from last message
                if isinstance(message, dict) and "content" in message:
                    if "Content is here -" in message["content"]:
                        return message["content"]
            # If no specific content found, return the last message
            if messages and isinstance(messages[-1], dict):
                return messages[-1].get("content", "")
        return str(messages)

    def _parse_input_content(self, content):
        """Parse input content to extract table information."""
        result = {
            "table_name": "",
            "table_data": {},
            "column_metadata": {},
            "abbreviations": {},
        }

        try:
            # Look for content patterns from existing prompts
            if "Content is here -" in content:
                # Extract JSON content
                content_start = content.find("Content is here -") + len(
                    "Content is here -"
                )
                content_end = content.find("and abbreviations")
                if content_end == -1:
                    content_end = len(content)

                json_str = content[content_start:content_end].strip()

                # Try to parse the JSON
                import re

                json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    result["table_name"] = data.get("table_name", "")
                    result["table_data"] = data.get("column_contents", {})
                    result["column_metadata"] = data.get("column_metadata", {})

                # Extract abbreviations
                if "abbreviations" in content:
                    abbr_start = content.find("abbreviations") + len(
                        "abbreviations are here -"
                    )
                    abbr_content = content[abbr_start:].strip()
                    abbr_match = re.search(r"\{.*\}", abbr_content)
                    if abbr_match:
                        result["abbreviations"] = json.loads(abbr_match.group())

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing input content: {e}")
            # Return default structure

        return result

    def _create_chat_completion_response(self, content):
        """Create a mock ChatCompletion response for compatibility."""

        # This creates a simplified response that matches expected interface
        class MockChoice:
            def __init__(self, content):
                self.message = type(
                    "obj", (object,), {"content": content, "role": "assistant"}
                )

        class MockChatCompletion:
            def __init__(self, content, model="gpt-4"):
                self.choices = [MockChoice(content)]
                self.model = model
                self.id = "dspy-generated"

        model_name = (
            getattr(self.config, "model", "gpt-4")
            if hasattr(self, "config")
            else "gpt-4"
        )
        return MockChatCompletion(content, model_name)

    def _fallback_predict(self, model_input, params):
        """Fallback to parent predict method."""

        # Convert dict back to MetadataConfig-like object for chat client
        class TempConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

        config_dict = (
            self.config if isinstance(self.config, dict) else self.config.__dict__
        )
        temp_config = TempConfig(config_dict)

        if not hasattr(self, "chat_client"):
            from src.dbxmetagen.chat_client import ChatClientFactory

            self.chat_client = ChatClientFactory.create_client(temp_config)

        chat_response = self.chat_client.create_completion(
            messages=model_input,
            model=config_dict["model"],
            max_tokens=config_dict["max_tokens"],
            temperature=config_dict["temperature"],
        )
        return chat_response

    def optimize_prompts(self, train_examples, validation_examples=None, num_threads=1):
        """
        Optimize DSPy prompts using training examples.

        Args:
            train_examples: List of training examples with input/output pairs
            validation_examples: Optional validation examples
            num_threads: Number of threads for optimization

        Returns:
            Optimized module
        """
        if not self._initialized:
            self._initialize_dspy()

        # Create training set
        trainset = []
        for example in train_examples:
            trainset.append(
                dspy.Example(
                    table_name=example["table_name"],
                    table_data=example["table_data"],
                    column_metadata=example["column_metadata"],
                    abbreviations=example.get("abbreviations", {}),
                    metadata_response=example["expected_response"],
                ).with_inputs(
                    "table_name", "table_data", "column_metadata", "abbreviations"
                )
            )

        # Set up validation set if provided
        valset = None
        if validation_examples:
            valset = []
            for example in validation_examples:
                valset.append(
                    dspy.Example(
                        table_name=example["table_name"],
                        table_data=example["table_data"],
                        column_metadata=example["column_metadata"],
                        abbreviations=example.get("abbreviations", {}),
                        metadata_response=example["expected_response"],
                    ).with_inputs(
                        "table_name", "table_data", "column_metadata", "abbreviations"
                    )
                )

        # Configure optimizer
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available for optimization")

        from dspy.teleprompt import BootstrapFewShot  # noqa: E402

        def validate_response(example, pred, trace=None):
            """Validation function for optimization."""
            # trace parameter is required by DSPy interface but not used
            try:
                # Parse prediction and expected response
                pred_json = json.loads(pred.metadata_response)
                # expected_response = json.loads(example.metadata_response)

                # Check if required keys are present
                has_table = "table" in pred_json
                has_columns = "columns" in pred_json
                has_column_contents = "column_contents" in pred_json

                # Basic validation
                return has_table and has_columns and has_column_contents
            except (ValueError, KeyError, json.JSONDecodeError, AttributeError):
                return False

        # Optimize
        teleprompter = BootstrapFewShot(
            metric=validate_response,
            max_bootstrapped_demos=4,
            max_labeled_demos=8,
            num_threads=num_threads,
        )

        optimized_module = teleprompter.compile(
            self.dspy_module, trainset=trainset, valset=valset
        )

        # Replace the current module with optimized version
        self.dspy_module = optimized_module

        return optimized_module

    def save_optimized_prompts(self, filepath):
        """Save optimized prompts to file."""
        if self.dspy_module:
            self.dspy_module.save(filepath)

    def load_optimized_prompts(self, filepath):
        """Load optimized prompts from file."""
        if not self._initialized:
            self._initialize_dspy()

        # Load the optimized module
        self.dspy_module.load(filepath)


class DSPyCommentGeneratorFactory:
    """Factory for creating DSPy Comment Generator instances."""

    @staticmethod
    def create_generator(config) -> DSPyCommentGeneratorModel:
        """Create a DSPy-powered comment generator."""
        generator = DSPyCommentGeneratorModel(config)
        generator.from_context(config)
        return generator

    @staticmethod
    def create_training_example(
        table_name, table_data, column_metadata, expected_response, abbreviations=None
    ):
        """Helper to create training examples for optimization."""
        return {
            "table_name": table_name,
            "table_data": table_data,
            "column_metadata": column_metadata,
            "abbreviations": abbreviations or {},
            "expected_response": expected_response,
        }
