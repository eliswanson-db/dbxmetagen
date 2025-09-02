# DSPy Integration for CommentGenerator

This document explains how to use the DSPy-powered CommentGeneratorModel alongside the existing dbxmetagen system.

## Overview

The DSPy integration provides prompt optimization capabilities for the comment generation functionality while maintaining full compatibility with the existing system. This allows you to:

- Use DSPy's optimization framework to improve prompts
- Maintain the same interface as the existing CommentGenerator
- Keep MLflow pyfunc compatibility
- Use existing prompts as a starting point without modifying them

## Architecture

The DSPy integration consists of several key components:

- `DSPyCommentGeneratorModel`: Main model class that inherits from both `CommentGenerator` and `mlflow.pyfunc.PythonModel`
- `DSPyCommentModule`: DSPy module that handles the comment generation logic
- `DSPyPromptExtractor`: Extracts existing prompts and converts them to DSPy format
- `DSPyCommentGeneratorFactory`: Factory for creating DSPy-powered generators

## Installation

Add DSPy to your requirements:

```bash
pip install dspy-ai>=2.4.0
```

Or use the updated requirements.txt file that now includes DSPy.

## Basic Usage

### Using DSPy Model Directly

```python
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.dspy_comment_generator import DSPyCommentGeneratorFactory

# Create configuration
config = MetadataConfig(
    mode="comment",
    model="gpt-4",
    max_tokens=4000,
    temperature=0.7,
    allow_data_in_comments=True
)

# Create DSPy model
dspy_model = DSPyCommentGeneratorFactory.create_generator(config)

# Use like any other CommentGenerator
response = dspy_model.predict(input_data)
```

### Integration with Existing System

The DSPy model is designed to be a drop-in replacement for the existing CommentGenerator. You can modify the factory to optionally use DSPy:

```python
from src.dbxmetagen.metadata_generator import MetadataGeneratorFactory
from src.dbxmetagen.dspy_comment_generator import DSPyCommentGeneratorFactory

class EnhancedMetadataGeneratorFactory:
    @staticmethod
    def create_generator(config, use_dspy=False):
        if use_dspy and config.mode == "comment":
            return DSPyCommentGeneratorFactory.create_generator(config)
        else:
            return MetadataGeneratorFactory.create_generator(config)
```

## Prompt Optimization

### Creating Training Examples

```python
from src.dbxmetagen.dspy_comment_generator import DSPyCommentGeneratorFactory

# Create training examples
training_examples = [
    DSPyCommentGeneratorFactory.create_training_example(
        table_name="catalog.schema.table",
        table_data={
            "index": [0, 1],
            "columns": ["id", "name", "email"],
            "data": [["1", "John", "john@test.com"], ["2", "Jane", "jane@test.com"]]
        },
        column_metadata={
            "id": {"data_type": "int", "num_nulls": "0"},
            "name": {"data_type": "string", "num_nulls": "0"},
            "email": {"data_type": "string", "num_nulls": "0"}
        },
        expected_response='{"table": "Description of table", "columns": ["id", "name", "email"], "column_contents": ["ID column", "Name column", "Email column"]}',
        abbreviations={}
    )
]
```

### Running Optimization

```python
# Create DSPy model
dspy_model = DSPyCommentGeneratorFactory.create_generator(config)

# Optimize prompts
optimized_module = dspy_model.optimize_prompts(
    train_examples=training_examples,
    validation_examples=validation_examples,  # Optional
    num_threads=4
)

# Save optimized prompts
dspy_model.save_optimized_prompts("optimized_prompts.json")
```

### Loading Optimized Prompts

```python
# Load previously optimized prompts
dspy_model.load_optimized_prompts("optimized_prompts.json")
```

## MLflow Integration

The DSPy model maintains full MLflow pyfunc compatibility:

```python
import mlflow

# Log the model
with mlflow.start_run():
    mlflow.log_param("model_type", "dspy_comment_generator")
    mlflow.log_param("base_model", config.model)
    
    mlflow.pyfunc.log_model(
        "dspy_comment_model",
        python_model=dspy_model,
        registered_model_name="DSPyCommentGenerator"
    )
```

## Configuration Options

The DSPy model respects all existing configuration options:

- `allow_data_in_comments`: Controls whether actual data values are included in comments
- `model`: OpenAI model to use (gpt-4, gpt-3.5-turbo, etc.)
- `temperature`: Sampling temperature
- `max_tokens`: Maximum tokens for response
- `max_prompt_length`: Maximum prompt length
- All other existing configuration options

## Backwards Compatibility

The DSPy integration is designed to be completely backwards compatible:

- Inherits from existing `CommentGenerator` class
- Maintains the same interface and methods
- Falls back to traditional methods if DSPy fails
- Can be used as a drop-in replacement

## Example: Gradual Migration

To gradually migrate to DSPy without impacting the existing system:

1. **Phase 1**: Use DSPy alongside existing system for testing
2. **Phase 2**: Use DSPy for specific tables or schemas
3. **Phase 3**: Optimize prompts based on real data
4. **Phase 4**: Full migration once confidence is established

```python
def create_generator_with_fallback(config, table_name="", use_dspy_for_tables=None):
    """Create generator with selective DSPy usage."""
    if use_dspy_for_tables and any(pattern in table_name for pattern in use_dspy_for_tables):
        try:
            return DSPyCommentGeneratorFactory.create_generator(config)
        except Exception as e:
            print(f"DSPy failed, falling back to traditional: {e}")
    
    # Use traditional generator
    return MetadataGeneratorFactory.create_generator(config)

# Use DSPy only for specific schemas
generator = create_generator_with_fallback(
    config, 
    table_name="finance.customers.customer_data",
    use_dspy_for_tables=["finance.", "sales."]
)
```

## Monitoring and Evaluation

The DSPy integration includes built-in evaluation capabilities:

```python
# Evaluate model performance
from src.dbxmetagen.dspy_comment_generator import evaluate_model_performance

results = evaluate_model_performance(
    model=dspy_model,
    test_examples=test_examples,
    metrics=["accuracy", "completeness", "relevance"]
)
```

## Best Practices

1. **Start with Small Scale**: Begin with a few tables or schemas
2. **Create Good Training Data**: Use real examples from your data
3. **Validate Optimization**: Always test optimized prompts before production
4. **Monitor Performance**: Track quality metrics over time
5. **Gradual Rollout**: Implement in phases to minimize risk

## Troubleshooting

### Common Issues

1. **DSPy Import Errors**: Ensure `dspy-ai>=2.4.0` is installed
2. **Model Configuration**: Check OpenAI API keys and model access
3. **Prompt Length**: DSPy may generate longer prompts; adjust `max_prompt_length`
4. **Memory Usage**: DSPy optimization can be memory-intensive

### Fallback Behavior

The DSPy model is designed to gracefully fallback to traditional methods if:
- DSPy initialization fails
- Model optimization encounters errors
- Response parsing fails

### Debug Mode

Enable debug mode for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# DSPy will provide detailed logging
dspy_model = DSPyCommentGeneratorFactory.create_generator(config)
```

## Performance Considerations

- **Optimization Time**: Initial prompt optimization can take 30-60 minutes
- **Memory Usage**: DSPy requires additional memory for optimization
- **API Costs**: Optimization makes many API calls; monitor usage
- **Caching**: DSPy caches results to reduce API calls

## Future Enhancements

Potential improvements to the DSPy integration:

1. **Multi-modal Optimization**: Support for different prompt strategies
2. **Continuous Learning**: Update prompts based on feedback
3. **A/B Testing**: Built-in comparison between traditional and DSPy approaches
4. **Performance Metrics**: Automated quality assessment
5. **Domain-specific Optimization**: Table-type or schema-specific prompts

## Support

For issues with the DSPy integration:

1. Check the examples in `examples/dspy_usage_example.py`
2. Review the implementation in `src/dbxmetagen/dspy_comment_generator.py`
3. Ensure all dependencies are installed correctly
4. Test with simple examples before complex tables

The DSPy integration is designed to enhance the existing system while maintaining full compatibility and providing a safe migration path.
