"""
Test framework for DSPy CommentGenerator integration.

This module provides comprehensive tests for the DSPy-powered comment generator,
including unit tests, integration tests, and comparison tests with the traditional generator.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Try to import DSPy components, skip tests if not available
try:
    from src.dbxmetagen.dspy_comment_generator import (
        DSPyCommentGeneratorModel,
        DSPyCommentGeneratorFactory,
        DSPyCommentModule,
        CommentSignature,
        CommentNoDataSignature,
    )
    from src.dbxmetagen.dspy_prompts import (
        DSPyPromptExtractor,
        DSPyPromptInitializer,
        initialize_dspy_from_existing_prompts,
    )
    from src.dbxmetagen.enhanced_factory import (
        EnhancedMetadataGeneratorFactory,
        DSPyMigrationHelper,
    )

    DSPY_AVAILABLE = True
except ImportError as e:
    DSPY_AVAILABLE = False
    skip_reason = f"DSPy not available: {e}"

from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import CommentResponse


@pytest.fixture
def sample_config():
    """Create a sample MetadataConfig for testing."""
    return MetadataConfig(
        mode="comment",
        model="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.7,
        allow_data_in_comments=True,
        max_prompt_length=50000,
        limit_prompt_based_on_cell_len=True,
        word_limit_per_cell=100,
        add_metadata=True,
        include_datatype_from_metadata=True,
        include_possible_data_fields_in_metadata=False,
        include_existing_table_comment=True,
        acro_content={"MRR": "Monthly Recurring Revenue"},
    )


@pytest.fixture
def sample_table_data():
    """Create sample table data for testing."""
    return {
        "table_name": "test.schema.sample_table",
        "column_contents": {
            "index": [0, 1],
            "columns": ["id", "name", "email"],
            "data": [
                ["1", "John Doe", "john@test.com"],
                ["2", "Jane Smith", "jane@test.com"],
            ],
        },
        "column_metadata": {
            "id": {
                "col_name": "id",
                "data_type": "int",
                "num_nulls": "0",
                "distinct_count": "100",
            },
            "name": {
                "col_name": "name",
                "data_type": "string",
                "num_nulls": "0",
                "distinct_count": "95",
            },
            "email": {
                "col_name": "email",
                "data_type": "string",
                "num_nulls": "0",
                "distinct_count": "100",
            },
        },
    }


@pytest.fixture
def sample_chat_messages(sample_table_data):
    """Create sample chat messages for testing."""
    content = json.dumps(sample_table_data)
    return [
        {
            "role": "user",
            "content": f"Content is here - {content} and abbreviations are here - {{}}",
        }
    ]


@pytest.mark.skipif(not DSPY_AVAILABLE, reason=skip_reason)
class TestDSPyCommentGeneratorModel:
    """Test suite for DSPy CommentGeneratorModel."""

    def test_model_initialization(self, sample_config):
        """Test model can be initialized properly."""
        model = DSPyCommentGeneratorModel(sample_config)
        assert model.config == sample_config
        assert not model._initialized

    def test_from_context_initialization(self, sample_config):
        """Test from_context method initializes DSPy properly."""
        with patch("dspy.settings.configure") as mock_configure, patch(
            "dspy.OpenAI"
        ) as mock_openai:

            model = DSPyCommentGeneratorModel()
            model.from_context(sample_config)

            assert model.config == sample_config
            assert model._initialized
            mock_configure.assert_called_once()

    def test_predict_method_basic(self, sample_config, sample_chat_messages):
        """Test predict method with basic functionality."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"), patch.object(
            DSPyCommentGeneratorModel, "_initialize_dspy"
        ), patch.object(
            DSPyCommentGeneratorModel, "_fallback_predict"
        ) as mock_fallback:

            # Mock fallback response
            mock_response = Mock()
            mock_fallback.return_value = mock_response

            model = DSPyCommentGeneratorModel(sample_config)
            model._initialized = True

            result = model.predict(sample_chat_messages)
            assert result == mock_response

    def test_predict_chat_response_structured(
        self, sample_config, sample_chat_messages
    ):
        """Test predict_chat_response returns structured response."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"), patch.object(
            DSPyCommentGeneratorModel, "_initialize_dspy"
        ):

            model = DSPyCommentGeneratorModel(sample_config)
            model._initialized = True

            # Mock DSPy module
            mock_module = Mock()
            mock_module.return_value = Mock(
                metadata_response='{"table": "Test table", "columns": ["id", "name"], "column_contents": ["ID column", "Name column"]}'
            )
            model.dspy_module = mock_module

            result = model.predict_chat_response(sample_chat_messages)

            assert isinstance(result, CommentResponse)
            assert result.table == "Test table"
            assert result.columns == ["id", "name"]
            assert result.column_contents == ["ID column", "Name column"]

    def test_content_extraction(self, sample_config):
        """Test content extraction from messages."""
        model = DSPyCommentGeneratorModel(sample_config)

        messages = [
            {"role": "system", "content": "System message"},
            {
                "role": "user",
                "content": 'Content is here - {"test": "data"} and abbreviations are here - {}',
            },
        ]

        content = model._extract_content_from_messages(messages)
        assert "Content is here" in content
        assert "test" in content

    def test_input_parsing(self, sample_config):
        """Test input content parsing."""
        model = DSPyCommentGeneratorModel(sample_config)

        content = 'Content is here - {"table_name": "test.table", "column_contents": {"columns": ["id"]}} and abbreviations are here - {"ID": "identifier"}'

        parsed = model._parse_input_content(content)

        assert parsed["table_name"] == "test.table"
        assert "columns" in parsed["table_data"]
        assert parsed["abbreviations"]["ID"] == "identifier"

    def test_fallback_behavior(self, sample_config, sample_chat_messages):
        """Test fallback to parent method on DSPy failure."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"), patch.object(
            DSPyCommentGeneratorModel, "_initialize_dspy"
        ):

            model = DSPyCommentGeneratorModel(sample_config)
            model._initialized = True

            # Mock DSPy module to raise exception
            mock_module = Mock()
            mock_module.side_effect = Exception("DSPy error")
            model.dspy_module = mock_module

            # Mock fallback method
            with patch.object(model, "_fallback_predict") as mock_fallback:
                mock_response = Mock()
                mock_fallback.return_value = mock_response

                result = model.predict(sample_chat_messages)
                assert result == mock_response
                mock_fallback.assert_called_once()


@pytest.mark.skipif(not DSPY_AVAILABLE, reason=skip_reason)
class TestDSPyPromptExtraction:
    """Test suite for DSPy prompt extraction and initialization."""

    def test_prompt_extractor_comment_instructions(self):
        """Test extraction of comment prompt instructions."""
        extractor = DSPyPromptExtractor()
        instructions = extractor.extract_comment_prompt_instructions()

        assert "AI assistant helping to generate metadata" in instructions
        assert "Databricks" in instructions
        assert "dictionary" in instructions

    def test_prompt_extractor_no_data_instructions(self):
        """Test extraction of no-data prompt instructions."""
        extractor = DSPyPromptExtractor()
        instructions = extractor.extract_comment_no_data_prompt_instructions()

        assert "sensitive" in instructions
        assert "do not include any data" in instructions
        assert "dictionary" in instructions

    def test_comment_examples(self):
        """Test extraction of comment examples."""
        extractor = DSPyPromptExtractor()
        examples = extractor.get_comment_examples()

        assert len(examples) > 0
        assert "input" in examples[0]
        assert "output" in examples[0]
        assert "table_name" in examples[0]["input"]

    def test_prompt_initializer(self, sample_config):
        """Test DSPy prompt initializer."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            initializer = DSPyPromptInitializer(allow_data_in_comments=True)
            instructions, examples = initializer.initialize_dspy_with_existing_prompts()

            assert instructions
            assert examples
            assert len(examples) > 0

    def test_initialize_from_existing_prompts(self, sample_config):
        """Test convenience function for initialization."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            instructions, examples, signature = initialize_dspy_from_existing_prompts(
                sample_config, allow_data_in_comments=True
            )

            assert instructions
            assert examples
            assert signature


@pytest.mark.skipif(not DSPY_AVAILABLE, reason=skip_reason)
class TestEnhancedFactory:
    """Test suite for enhanced metadata generator factory."""

    def test_create_traditional_generator(self, sample_config):
        """Test creation of traditional generator."""
        generator = EnhancedMetadataGeneratorFactory.create_generator(
            sample_config, use_dspy=False
        )

        # Should be traditional generator
        assert generator.__class__.__name__ == "CommentGenerator"

    def test_create_dspy_generator_when_available(self, sample_config):
        """Test creation of DSPy generator when available."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            generator = EnhancedMetadataGeneratorFactory.create_generator(
                sample_config, use_dspy=True
            )

            assert generator.__class__.__name__ == "DSPyCommentGeneratorModel"

    def test_fallback_on_dspy_error(self, sample_config):
        """Test fallback to traditional generator on DSPy error."""
        with patch(
            "src.dbxmetagen.enhanced_factory.DSPyCommentGeneratorFactory.create_generator",
            side_effect=Exception("DSPy error"),
        ):

            generator = EnhancedMetadataGeneratorFactory.create_generator(
                sample_config, use_dspy=True, fallback_on_error=True
            )

            # Should fallback to traditional
            assert generator.__class__.__name__ == "CommentGenerator"

    def test_selective_dspy_usage(self, sample_config):
        """Test selective DSPy usage based on patterns."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            # Should use DSPy for finance tables
            generator = (
                EnhancedMetadataGeneratorFactory.create_generator_with_selective_dspy(
                    sample_config,
                    table_name="finance.customer.data",
                    dspy_patterns=["finance.", "sales."],
                )
            )
            assert generator.__class__.__name__ == "DSPyCommentGeneratorModel"

            # Should use traditional for other tables
            generator = (
                EnhancedMetadataGeneratorFactory.create_generator_with_selective_dspy(
                    sample_config,
                    table_name="hr.employee.data",
                    dspy_patterns=["finance.", "sales."],
                )
            )
            assert generator.__class__.__name__ == "CommentGenerator"

    def test_environment_config(self, sample_config):
        """Test environment-based configuration."""
        with patch.dict(
            os.environ,
            {"DBXMETAGEN_USE_DSPY": "true", "DBXMETAGEN_DSPY_FALLBACK": "true"},
        ), patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            generator = (
                EnhancedMetadataGeneratorFactory.create_generator_with_env_config(
                    sample_config
                )
            )
            assert generator.__class__.__name__ == "DSPyCommentGeneratorModel"


@pytest.mark.skipif(not DSPY_AVAILABLE, reason=skip_reason)
class TestDSPyMigrationHelper:
    """Test suite for DSPy migration helper."""

    def test_create_comparison_generators(self, sample_config):
        """Test creation of comparison generators."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            traditional, dspy = DSPyMigrationHelper.create_comparison_generators(
                sample_config
            )

            assert traditional.__class__.__name__ == "CommentGenerator"
            assert dspy.__class__.__name__ == "DSPyCommentGeneratorModel"

    def test_response_comparison(self, sample_config):
        """Test response comparison functionality."""
        # Create mock responses
        trad_response = CommentResponse(
            table="Traditional table description",
            columns=["id", "name"],
            column_contents=["ID column", "Name column"],
        )

        dspy_response = CommentResponse(
            table="DSPy table description",
            columns=["id", "name"],
            column_contents=["Identifier column", "Full name column"],
        )

        comparison = DSPyMigrationHelper.compare_responses(trad_response, dspy_response)

        assert "traditional" in comparison
        assert "dspy" in comparison
        assert "metrics" in comparison
        assert "length" in comparison["metrics"]
        assert "structure" in comparison["metrics"]

    def test_completeness_check(self):
        """Test completeness checking functionality."""
        complete_response = CommentResponse(
            table="Complete description",
            columns=["id", "name"],
            column_contents=["ID desc", "Name desc"],
        )

        completeness = DSPyMigrationHelper._check_completeness(complete_response)

        assert completeness["has_table_desc"] == True
        assert completeness["has_columns"] == True
        assert completeness["has_column_contents"] == True
        assert completeness["column_count_match"] == True

    def test_migration_test_runner(self, sample_config, sample_chat_messages):
        """Test migration test runner."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"), patch.object(
            DSPyMigrationHelper, "create_comparison_generators"
        ) as mock_create:

            # Mock generators
            mock_traditional = Mock()
            mock_dspy = Mock()
            mock_traditional.predict_chat_response.return_value = CommentResponse(
                table="Traditional", columns=["id"], column_contents=["Traditional ID"]
            )
            mock_dspy.predict_chat_response.return_value = CommentResponse(
                table="DSPy", columns=["id"], column_contents=["DSPy ID"]
            )

            mock_create.return_value = (mock_traditional, mock_dspy)

            results = DSPyMigrationHelper.run_migration_test(
                sample_config, [sample_chat_messages[0]], save_results=False
            )

            assert results["test_count"] == 1
            assert len(results["results"]) == 1
            assert "summary" in results


@pytest.mark.skipif(not DSPY_AVAILABLE, reason=skip_reason)
class TestOptimization:
    """Test suite for DSPy optimization functionality."""

    def test_training_example_creation(self):
        """Test creation of training examples."""
        example = DSPyCommentGeneratorFactory.create_training_example(
            table_name="test.table",
            table_data={"columns": ["id"]},
            column_metadata={"id": {"data_type": "int"}},
            expected_response='{"table": "Test", "columns": ["id"], "column_contents": ["ID"]}',
            abbreviations={"ID": "identifier"},
        )

        assert example["table_name"] == "test.table"
        assert example["abbreviations"]["ID"] == "identifier"
        assert "expected_response" in example

    def test_optimize_prompts_interface(self, sample_config):
        """Test optimize_prompts method interface (without actual optimization)."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"), patch(
            "dspy.teleprompt.BootstrapFewShot"
        ) as mock_bootstrap:

            # Mock optimization
            mock_optimizer = Mock()
            mock_optimized_module = Mock()
            mock_optimizer.compile.return_value = mock_optimized_module
            mock_bootstrap.return_value = mock_optimizer

            model = DSPyCommentGeneratorFactory.create_generator(sample_config)

            training_examples = [
                DSPyCommentGeneratorFactory.create_training_example(
                    table_name="test",
                    table_data={},
                    column_metadata={},
                    expected_response='{"table": "test", "columns": [], "column_contents": []}',
                )
            ]

            result = model.optimize_prompts(training_examples)
            assert result == mock_optimized_module

    def test_save_load_optimized_prompts(self, sample_config):
        """Test saving and loading optimized prompts."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            model = DSPyCommentGeneratorFactory.create_generator(sample_config)

            # Mock DSPy module save/load methods
            model.dspy_module.save = Mock()
            model.dspy_module.load = Mock()

            # Test save
            model.save_optimized_prompts("test_prompts.json")
            model.dspy_module.save.assert_called_once_with("test_prompts.json")

            # Test load
            model.load_optimized_prompts("test_prompts.json")
            model.dspy_module.load.assert_called_once_with("test_prompts.json")


class TestIntegrationWithoutDSPy:
    """Test integration scenarios when DSPy is not available."""

    def test_enhanced_factory_without_dspy(self, sample_config):
        """Test enhanced factory falls back gracefully without DSPy."""
        generator = EnhancedMetadataGeneratorFactory.create_generator(
            sample_config, use_dspy=False
        )

        assert generator.__class__.__name__ == "CommentGenerator"

    def test_import_error_handling(self, sample_config):
        """Test graceful handling of DSPy import errors."""
        with patch(
            "src.dbxmetagen.enhanced_factory.DSPyCommentGeneratorFactory.create_generator",
            side_effect=ImportError("DSPy not available"),
        ):

            # Should raise ImportError when fallback is disabled
            with pytest.raises(ImportError):
                EnhancedMetadataGeneratorFactory.create_generator(
                    sample_config, use_dspy=True, fallback_on_error=False
                )

            # Should fallback when enabled
            generator = EnhancedMetadataGeneratorFactory.create_generator(
                sample_config, use_dspy=True, fallback_on_error=True
            )
            assert generator.__class__.__name__ == "CommentGenerator"


@pytest.mark.skipif(not DSPY_AVAILABLE, reason=skip_reason)
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, sample_config, sample_table_data):
        """Test complete workflow from input to output."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"), patch.object(
            DSPyCommentModule, "forward"
        ) as mock_forward:

            # Mock DSPy response
            mock_result = Mock()
            mock_result.metadata_response = json.dumps(
                {
                    "table": "Sample table for testing purposes",
                    "columns": ["id", "name", "email"],
                    "column_contents": [
                        "Unique identifier for records",
                        "Full name of the person",
                        "Email address for communication",
                    ],
                }
            )
            mock_forward.return_value = mock_result

            # Create generator
            generator = DSPyCommentGeneratorFactory.create_generator(sample_config)

            # Create input
            content = json.dumps(sample_table_data)
            messages = [
                {
                    "role": "user",
                    "content": f"Content is here - {content} and abbreviations are here - {{}}",
                }
            ]

            # Generate response
            response = generator.predict_chat_response(messages)

            assert isinstance(response, CommentResponse)
            assert response.table == "Sample table for testing purposes"
            assert len(response.columns) == 3
            assert len(response.column_contents) == 3

    def test_error_recovery(self, sample_config, sample_chat_messages):
        """Test error recovery and fallback mechanisms."""
        with patch("dspy.settings.configure"), patch("dspy.OpenAI"):

            generator = DSPyCommentGeneratorFactory.create_generator(sample_config)

            # Mock DSPy to fail
            generator.dspy_module = Mock(side_effect=Exception("DSPy error"))

            # Mock fallback chat client
            with patch.object(generator, "chat_client") as mock_client:
                mock_completion = Mock()
                mock_client.create_completion.return_value = mock_completion

                result = generator.predict(sample_chat_messages)
                assert result == mock_completion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
