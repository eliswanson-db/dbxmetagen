"""
Enhanced MetadataGeneratorFactory that supports DSPy integration.

This module provides a factory that can create either traditional or DSPy-powered
generators without impacting the existing system. It serves as the integration path
for gradually adopting DSPy optimization.
"""

import json
import os
from typing import Optional, List, Dict, Any
from src.dbxmetagen.metadata_generator import (
    MetadataGeneratorFactory,
    MetadataGenerator,
)
from src.dbxmetagen.config import MetadataConfig


class EnhancedMetadataGeneratorFactory:
    """
    Enhanced factory that can create both traditional and DSPy-powered generators.

    This factory extends the existing MetadataGeneratorFactory to optionally use
    DSPy optimization while maintaining full backwards compatibility.
    """

    @staticmethod
    def create_generator(
        config: MetadataConfig,
        use_dspy: bool = False,
        dspy_config: Optional[Dict[str, Any]] = None,
        fallback_on_error: bool = True,
    ) -> MetadataGenerator:
        """
        Create a metadata generator with optional DSPy support.

        Args:
            config: MetadataConfig object with generation settings
            use_dspy: Whether to use DSPy-powered generator
            dspy_config: Optional DSPy-specific configuration
            fallback_on_error: Whether to fallback to traditional generator on DSPy errors

        Returns:
            MetadataGenerator instance (traditional or DSPy-powered)
        """
        if use_dspy and config.mode == "comment":
            try:
                return EnhancedMetadataGeneratorFactory._create_dspy_generator(
                    config, dspy_config or {}
                )
            except Exception as e:
                if fallback_on_error:
                    print(f"DSPy generator creation failed, using traditional: {e}")
                    return MetadataGeneratorFactory.create_generator(config)
                else:
                    raise
        else:
            # Use existing factory logic
            return MetadataGeneratorFactory.create_generator(config)

    @staticmethod
    def _create_dspy_generator(config: MetadataConfig, dspy_config: Dict[str, Any]):
        """Create DSPy-powered generator."""
        try:
            from src.dbxmetagen.dspy_comment_generator import (
                DSPyCommentGeneratorFactory,
            )

            # Apply DSPy-specific configuration
            if dspy_config:
                for key, value in dspy_config.items():
                    setattr(config, key, value)

            return DSPyCommentGeneratorFactory.create_generator(config)
        except ImportError as e:
            raise ImportError(
                f"DSPy integration not available. Install with: pip install dspy-ai>=2.4.0. Error: {e}"
            ) from e

    @staticmethod
    def create_generator_with_selective_dspy(
        config: MetadataConfig,
        table_name: str = "",
        dspy_patterns: Optional[List[str]] = None,
        dspy_config: Optional[Dict[str, Any]] = None,
    ) -> MetadataGenerator:
        """
        Create generator with selective DSPy usage based on table patterns.

        This allows for gradual migration by using DSPy only for specific
        tables, schemas, or catalogs.

        Args:
            config: MetadataConfig object
            table_name: Full table name (catalog.schema.table)
            dspy_patterns: List of patterns to match for DSPy usage
            dspy_config: Optional DSPy configuration

        Returns:
            MetadataGenerator instance
        """
        use_dspy = False

        if dspy_patterns and table_name:
            use_dspy = any(pattern in table_name for pattern in dspy_patterns)

        return EnhancedMetadataGeneratorFactory.create_generator(
            config=config,
            use_dspy=use_dspy,
            dspy_config=dspy_config,
            fallback_on_error=True,
        )

    @staticmethod
    def create_generator_with_env_config(config: MetadataConfig) -> MetadataGenerator:
        """
        Create generator based on environment configuration.

        This reads environment variables to determine DSPy usage:
        - DBXMETAGEN_USE_DSPY: Enable/disable DSPy
        - DBXMETAGEN_DSPY_PATTERNS: Comma-separated patterns for selective usage
        - DBXMETAGEN_DSPY_FALLBACK: Enable/disable fallback to traditional

        Args:
            config: MetadataConfig object

        Returns:
            MetadataGenerator instance
        """

        # Check environment variables
        use_dspy = os.getenv("DBXMETAGEN_USE_DSPY", "false").lower() == "true"
        dspy_patterns = os.getenv("DBXMETAGEN_DSPY_PATTERNS", "").split(",")
        dspy_patterns = [p.strip() for p in dspy_patterns if p.strip()]
        fallback_enabled = (
            os.getenv("DBXMETAGEN_DSPY_FALLBACK", "true").lower() == "true"
        )

        # DSPy configuration from environment
        dspy_config = {}
        if os.getenv("DBXMETAGEN_DSPY_OPTIMIZATION_ENABLED"):
            dspy_config["optimization_enabled"] = (
                os.getenv("DBXMETAGEN_DSPY_OPTIMIZATION_ENABLED").lower() == "true"
            )

        return EnhancedMetadataGeneratorFactory.create_generator(
            config=config,
            use_dspy=use_dspy,
            dspy_config=dspy_config,
            fallback_on_error=fallback_enabled,
        )


class DSPyMigrationHelper:
    """Helper class for managing DSPy migration."""

    @staticmethod
    def create_comparison_generators(config: MetadataConfig):
        """
        Create both traditional and DSPy generators for comparison.

        Args:
            config: MetadataConfig object

        Returns:
            Tuple of (traditional_generator, dspy_generator)
        """
        traditional = MetadataGeneratorFactory.create_generator(config)

        try:
            from src.dbxmetagen.dspy_comment_generator import (
                DSPyCommentGeneratorFactory,
            )

            dspy = DSPyCommentGeneratorFactory.create_generator(config)
        except ImportError:
            print("DSPy not available for comparison")
            dspy = None
        except Exception as e:
            print(f"Could not create DSPy generator: {e}")
            dspy = None

        return traditional, dspy

    @staticmethod
    def compare_responses(traditional_response, dspy_response, metrics=None):
        """
        Compare responses from traditional and DSPy generators.

        Args:
            traditional_response: Response from traditional generator
            dspy_response: Response from DSPy generator
            metrics: List of metrics to evaluate

        Returns:
            Dict with comparison results
        """
        if metrics is None:
            metrics = ["length", "completeness", "structure"]

        comparison = {
            "traditional": traditional_response,
            "dspy": dspy_response,
            "metrics": {},
        }

        try:
            # Length comparison
            if "length" in metrics:
                trad_len = len(str(traditional_response))
                dspy_len = len(str(dspy_response))
                comparison["metrics"]["length"] = {
                    "traditional": trad_len,
                    "dspy": dspy_len,
                    "difference": dspy_len - trad_len,
                }

            # Structure comparison (for CommentResponse objects)
            if "structure" in metrics:
                comparison["metrics"]["structure"] = {
                    "traditional_has_table": hasattr(traditional_response, "table"),
                    "dspy_has_table": hasattr(dspy_response, "table"),
                    "traditional_has_columns": hasattr(traditional_response, "columns"),
                    "dspy_has_columns": hasattr(dspy_response, "columns"),
                    "traditional_has_contents": hasattr(
                        traditional_response, "column_contents"
                    ),
                    "dspy_has_contents": hasattr(dspy_response, "column_contents"),
                }

            # Completeness comparison
            if "completeness" in metrics:
                trad_complete = DSPyMigrationHelper._check_completeness(
                    traditional_response
                )
                dspy_complete = DSPyMigrationHelper._check_completeness(dspy_response)
                comparison["metrics"]["completeness"] = {
                    "traditional": trad_complete,
                    "dspy": dspy_complete,
                }

        except (AttributeError, TypeError, ValueError) as e:
            comparison["comparison_error"] = str(e)

        return comparison

    @staticmethod
    def _check_completeness(response):
        """Check if response is complete."""
        if (
            hasattr(response, "table")
            and hasattr(response, "columns")
            and hasattr(response, "column_contents")
        ):
            return {
                "has_table_desc": bool(response.table),
                "has_columns": bool(response.columns),
                "has_column_contents": bool(response.column_contents),
                "column_count_match": (
                    len(response.columns) == len(response.column_contents)
                    if response.column_contents
                    else False
                ),
            }
        return {"error": "Response structure not recognized"}

    @staticmethod
    def run_migration_test(
        config: MetadataConfig, test_inputs: List[Any], save_results: bool = True
    ):
        """
        Run a migration test comparing traditional and DSPy generators.

        Args:
            config: MetadataConfig object
            test_inputs: List of test inputs
            save_results: Whether to save results to file

        Returns:
            Dict with test results
        """
        results = {
            "config": config.__dict__,
            "test_count": len(test_inputs),
            "results": [],
            "summary": {},
        }

        # Create generators
        traditional, dspy = DSPyMigrationHelper.create_comparison_generators(config)

        if not dspy:
            results["error"] = "DSPy generator not available"
            return results

        # Run tests
        for i, test_input in enumerate(test_inputs):
            test_result = {"test_id": i, "input": test_input}

            try:
                # Get traditional response
                if hasattr(traditional, "predict_chat_response"):
                    trad_response = traditional.predict_chat_response(test_input)
                else:
                    # Traditional generators don't have predict method - use get_responses if available
                    trad_response = str(traditional)
                test_result["traditional_response"] = trad_response
            except (AttributeError, ValueError, TypeError) as e:
                test_result["traditional_error"] = str(e)

            try:
                # Get DSPy response
                if hasattr(dspy, "predict_chat_response"):
                    dspy_response = dspy.predict_chat_response(test_input)
                elif hasattr(dspy, "predict"):
                    dspy_response = dspy.predict(test_input)
                else:
                    dspy_response = str(dspy)
                test_result["dspy_response"] = dspy_response
            except (AttributeError, ValueError, TypeError) as e:
                test_result["dspy_error"] = str(e)

            # Compare if both succeeded
            if "traditional_response" in test_result and "dspy_response" in test_result:
                comparison = DSPyMigrationHelper.compare_responses(
                    test_result["traditional_response"], test_result["dspy_response"]
                )
                test_result["comparison"] = comparison

            results["results"].append(test_result)

        # Generate summary
        successful_tests = [r for r in results["results"] if "comparison" in r]
        results["summary"] = {
            "total_tests": len(test_inputs),
            "successful_comparisons": len(successful_tests),
            "traditional_errors": len(
                [r for r in results["results"] if "traditional_error" in r]
            ),
            "dspy_errors": len([r for r in results["results"] if "dspy_error" in r]),
        }

        # Save results if requested
        if save_results:
            import datetime

            filename = f"migration_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                # Convert any non-serializable objects to strings
                serializable_results = DSPyMigrationHelper._make_serializable(results)
                json.dump(serializable_results, f, indent=2)
            results["saved_to"] = filename

        return results

    @staticmethod
    def _make_serializable(obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {
                key: DSPyMigrationHelper._make_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [DSPyMigrationHelper._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return {
                "__class__": obj.__class__.__name__,
                "__data__": DSPyMigrationHelper._make_serializable(obj.__dict__),
            }
        else:
            try:
                json.dumps(obj)  # Test if serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)


# Backward compatibility - can replace the original factory gradually
class MetadataGeneratorFactoryV2(EnhancedMetadataGeneratorFactory):
    """Alias for enhanced factory for version 2 compatibility."""
