"""
Example usage of the DSPy CommentGeneratorModel.

This example demonstrates how to use the DSPy-powered comment generator
while maintaining compatibility with the existing dbxmetagen system.
"""

import json
import mlflow
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.dspy_comment_generator import (
    DSPyCommentGeneratorModel,
    DSPyCommentGeneratorFactory,
)
from src.dbxmetagen.dspy_prompts import initialize_dspy_from_existing_prompts


# Example configuration
def create_example_config():
    """Create an example configuration for testing."""
    config = MetadataConfig(
        mode="comment",
        model="gpt-4",
        max_tokens=4000,
        temperature=0.7,
        allow_data_in_comments=True,
        max_prompt_length=50000,
        limit_prompt_based_on_cell_len=True,
        word_limit_per_cell=100,
        add_metadata=True,
        include_datatype_from_metadata=True,
        include_possible_data_fields_in_metadata=False,
        include_existing_table_comment=True,
        acro_content={
            "MRR": "Monthly Recurring Revenue",
            "EAP": "Enterprise Architecture Platform",
        },
    )
    return config


def example_basic_usage():
    """Example of basic DSPy model usage."""
    print("=== Basic DSPy Usage Example ===")

    # Create configuration
    config = create_example_config()

    # Create DSPy model
    dspy_model = DSPyCommentGeneratorFactory.create_generator(config)

    # Example input data (simulating what would come from Databricks table)
    example_input = [
        {
            "role": "user",
            "content": """Content is here - {"table_name": "sales.customers.customer_info", "column_contents": {"index": [0, 1], "columns": ["customer_id", "first_name", "last_name", "email", "signup_date"], "data": [["CUST001", "John", "Doe", "john.doe@email.com", "2023-01-15"], ["CUST002", "Jane", "Smith", "jane.smith@email.com", "2023-02-20"]], "column_metadata": {"customer_id": {"col_name": "customer_id", "data_type": "string", "num_nulls": "0", "distinct_count": "1000", "avg_col_len": "7", "max_col_len": "7"}, "first_name": {"col_name": "first_name", "data_type": "string", "num_nulls": "0", "distinct_count": "800", "avg_col_len": "6", "max_col_len": "15"}, "last_name": {"col_name": "last_name", "data_type": "string", "num_nulls": "0", "distinct_count": "900", "avg_col_len": "7", "max_col_len": "20"}, "email": {"col_name": "email", "data_type": "string", "num_nulls": "0", "distinct_count": "1000", "avg_col_len": "20", "max_col_len": "50"}, "signup_date": {"col_name": "signup_date", "data_type": "string", "num_nulls": "0", "distinct_count": "365", "avg_col_len": "10", "max_col_len": "10"}}}} and abbreviations are here - {"CUST": "customer"}""",
        }
    ]

    # Generate response
    try:
        response = dspy_model.predict(example_input)
        print("DSPy Response:")
        if hasattr(response, "choices") and response.choices:
            print(response.choices[0].message.content)
        else:
            print(response)
    except Exception as e:
        print(f"Error: {e}")


def example_structured_response():
    """Example of structured response generation."""
    print("\n=== Structured Response Example ===")

    # Create configuration
    config = create_example_config()

    # Create DSPy model
    dspy_model = DSPyCommentGeneratorFactory.create_generator(config)

    # Example prompt content
    prompt_content = [
        {"role": "system", "content": "Generate metadata for the given table."},
        {
            "role": "user",
            "content": """Content is here - {"table_name": "finance.transactions.payment_history", "column_contents": {"index": [0, 1], "columns": ["transaction_id", "customer_id", "amount", "payment_method", "transaction_date"], "data": [["TXN001", "CUST001", "99.99", "credit_card", "2023-03-01"], ["TXN002", "CUST002", "149.99", "bank_transfer", "2023-03-02"]], "column_metadata": {"transaction_id": {"col_name": "transaction_id", "data_type": "string", "num_nulls": "0", "distinct_count": "5000", "avg_col_len": "6", "max_col_len": "6"}, "customer_id": {"col_name": "customer_id", "data_type": "string", "num_nulls": "0", "distinct_count": "1000", "avg_col_len": "7", "max_col_len": "7"}, "amount": {"col_name": "amount", "data_type": "string", "num_nulls": "0", "distinct_count": "500", "avg_col_len": "5", "max_col_len": "10"}, "payment_method": {"col_name": "payment_method", "data_type": "string", "num_nulls": "0", "distinct_count": "4", "avg_col_len": "10", "max_col_len": "15"}, "transaction_date": {"col_name": "transaction_date", "data_type": "string", "num_nulls": "0", "distinct_count": "365", "avg_col_len": "10", "max_col_len": "10"}}}} and abbreviations are here - {"TXN": "transaction"}""",
        },
    ]

    # Generate structured response
    try:
        structured_response = dspy_model.predict_chat_response(prompt_content)
        print("Structured Response:")
        print(f"Table: {structured_response.table}")
        print(f"Columns: {structured_response.columns}")
        print("Column Contents:")
        for i, content in enumerate(structured_response.column_contents):
            column_name = (
                structured_response.columns[i]
                if i < len(structured_response.columns)
                else f"Column {i+1}"
            )
            print(f"  {column_name}: {content}")
    except Exception as e:
        print(f"Error: {e}")


def example_prompt_optimization():
    """Example of DSPy prompt optimization."""
    print("\n=== Prompt Optimization Example ===")

    # Create configuration
    config = create_example_config()

    # Create DSPy model
    dspy_model = DSPyCommentGeneratorFactory.create_generator(config)

    # Create training examples
    training_examples = [
        DSPyCommentGeneratorFactory.create_training_example(
            table_name="test.schema.sample_table",
            table_data={
                "index": [0, 1],
                "columns": ["id", "name", "email"],
                "data": [
                    ["1", "John", "john@test.com"],
                    ["2", "Jane", "jane@test.com"],
                ],
            },
            column_metadata={
                "id": {"data_type": "int", "num_nulls": "0"},
                "name": {"data_type": "string", "num_nulls": "0"},
                "email": {"data_type": "string", "num_nulls": "0"},
            },
            expected_response='{"table": "Sample test table for demonstration purposes.", "columns": ["id", "name", "email"], "column_contents": ["Unique identifier for records.", "Full name of the person.", "Email address for contact."]}',
            abbreviations={},
        )
        # Add more training examples as needed
    ]

    print("Starting prompt optimization...")
    try:
        # Optimize prompts (this would take time in a real scenario)
        optimized_module = dspy_model.optimize_prompts(
            train_examples=training_examples,
            num_threads=1,  # Use 1 thread for demonstration
        )
        print("Prompt optimization completed!")

        # Save optimized prompts
        dspy_model.save_optimized_prompts("optimized_comment_prompts.json")
        print("Optimized prompts saved to optimized_comment_prompts.json")

    except Exception as e:
        print(f"Optimization error: {e}")


def example_mlflow_integration():
    """Example of MLflow integration."""
    print("\n=== MLflow Integration Example ===")

    # Create configuration
    config = create_example_config()

    # Create and configure the model
    dspy_model = DSPyCommentGeneratorModel(config)
    dspy_model.from_context(config)

    # Example of logging the model with MLflow
    try:
        # Start MLflow run
        with mlflow.start_run():
            # Log model parameters
            mlflow.log_param("model_type", "dspy_comment_generator")
            mlflow.log_param("base_model", config.model)
            mlflow.log_param("temperature", config.temperature)
            mlflow.log_param("max_tokens", config.max_tokens)

            # Log the model (in a real scenario you'd want to define a proper signature)
            # mlflow.pyfunc.log_model(
            #     "dspy_comment_model",
            #     python_model=dspy_model,
            #     registered_model_name="DSPyCommentGenerator"
            # )

            print("Model logged to MLflow successfully!")

    except Exception as e:
        print(f"MLflow integration error: {e}")


def example_integration_path():
    """Example of how to integrate DSPy model into existing system."""
    print("\n=== Integration Path Example ===")

    # This shows how you could modify the MetadataGeneratorFactory
    # to optionally use the DSPy model without impacting existing functionality

    class EnhancedMetadataGeneratorFactory:
        @staticmethod
        def create_generator(config, use_dspy=False):
            if use_dspy and config.mode == "comment":
                # Use DSPy model
                generator = DSPyCommentGeneratorFactory.create_generator(config)
                print("Created DSPy-powered CommentGenerator")
                return generator
            else:
                # Use existing factory logic
                from src.dbxmetagen.metadata_generator import MetadataGeneratorFactory

                generator = MetadataGeneratorFactory.create_generator(config)
                print("Created traditional CommentGenerator")
                return generator

    # Example usage
    config = create_example_config()

    # Traditional generator
    traditional_generator = EnhancedMetadataGeneratorFactory.create_generator(
        config, use_dspy=False
    )

    # DSPy generator
    dspy_generator = EnhancedMetadataGeneratorFactory.create_generator(
        config, use_dspy=True
    )

    print("Both generators created successfully!")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_structured_response()
    example_prompt_optimization()
    example_mlflow_integration()
    example_integration_path()

    print("\n=== DSPy CommentGenerator Examples Completed ===")
