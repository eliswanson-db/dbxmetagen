# DSPy Integration for dbxmetagen

This directory contains a complete DSPy integration for the dbxmetagen CommentGenerator that allows you to use DSPy's prompt optimization capabilities while maintaining full compatibility with the existing system.

## Quick Start

1. **Install DSPy**:
   ```bash
   pip install dspy-ai>=2.4.0
   ```

2. **Use DSPy CommentGenerator**:
   ```python
   from src.dbxmetagen.config import MetadataConfig
   from src.dbxmetagen.dspy_comment_generator import DSPyCommentGeneratorFactory
   
   config = MetadataConfig(mode="comment", model="gpt-4", allow_data_in_comments=True)
   generator = DSPyCommentGeneratorFactory.create_generator(config)
   
   # Use like any other CommentGenerator
   response = generator.predict_chat_response(messages)
   ```

3. **Integration without Impact**:
   ```python
   from src.dbxmetagen.enhanced_factory import EnhancedMetadataGeneratorFactory
   
   # Use DSPy selectively
   generator = EnhancedMetadataGeneratorFactory.create_generator_with_selective_dspy(
       config, 
       table_name="finance.customers.data",
       dspy_patterns=["finance.", "sales."]  # Use DSPy only for these schemas
   )
   ```

## Key Features

✅ **Full Inheritance**: DSPy model inherits from existing CommentGenerator  
✅ **MLflow Compatible**: Maintains pyfunc interface  
✅ **Prompt Optimization**: Uses DSPy for automatic prompt improvement  
✅ **No Original Modification**: Existing prompts.py remains unchanged  
✅ **Graceful Fallback**: Falls back to traditional methods on errors  
✅ **Selective Usage**: Use DSPy only for specific tables/schemas  
✅ **Environment Config**: Control via environment variables  

## Files Created

- `src/dbxmetagen/dspy_comment_generator.py` - Main DSPy model implementation
- `src/dbxmetagen/dspy_prompts.py` - DSPy prompt templates based on existing prompts
- `src/dbxmetagen/enhanced_factory.py` - Integration factory for gradual migration
- `examples/dspy_usage_example.py` - Complete usage examples
- `tests/test_dspy_integration.py` - Comprehensive test suite
- `docs/DSPY_INTEGRATION.md` - Detailed documentation

## Integration Paths

### 1. Drop-in Replacement
```python
# Replace existing factory
generator = DSPyCommentGeneratorFactory.create_generator(config)
```

### 2. Selective Usage
```python
# Use DSPy only for specific patterns
generator = EnhancedMetadataGeneratorFactory.create_generator_with_selective_dspy(
    config, table_name, dspy_patterns=["finance.", "customer."]
)
```

### 3. Environment-Controlled
```bash
export DBXMETAGEN_USE_DSPY=true
export DBXMETAGEN_DSPY_PATTERNS="finance.,sales."
```

### 4. A/B Testing
```python
# Compare traditional vs DSPy responses
traditional, dspy = DSPyMigrationHelper.create_comparison_generators(config)
comparison = DSPyMigrationHelper.compare_responses(trad_response, dspy_response)
```

## Prompt Optimization

The DSPy integration allows you to optimize prompts using real data:

```python
# Create training examples
training_examples = [
    DSPyCommentGeneratorFactory.create_training_example(
        table_name="catalog.schema.table",
        table_data={"columns": ["id", "name"], "data": [...]},
        column_metadata={"id": {...}, "name": {...}},
        expected_response='{"table": "...", "columns": [...], "column_contents": [...]}'
    )
]

# Optimize prompts
optimized_module = dspy_model.optimize_prompts(training_examples)

# Save for reuse
dspy_model.save_optimized_prompts("optimized_prompts.json")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all DSPy tests
pytest tests/test_dspy_integration.py -v

# Run specific test categories
pytest tests/test_dspy_integration.py::TestDSPyCommentGeneratorModel -v
pytest tests/test_dspy_integration.py::TestEnhancedFactory -v
```

## Migration Strategy

1. **Phase 1**: Install DSPy and test basic functionality
2. **Phase 2**: Use for specific tables/schemas with fallback enabled
3. **Phase 3**: Collect data and optimize prompts
4. **Phase 4**: Gradually expand usage based on results
5. **Phase 5**: Full migration once confidence is established

## Environment Variables

Control DSPy usage through environment variables:

- `DBXMETAGEN_USE_DSPY=true/false` - Enable/disable DSPy
- `DBXMETAGEN_DSPY_PATTERNS=pattern1,pattern2` - Table patterns for DSPy usage
- `DBXMETAGEN_DSPY_FALLBACK=true/false` - Enable fallback to traditional methods

## Benefits

- **Improved Quality**: DSPy optimization can improve comment quality
- **Consistency**: More consistent outputs through optimization
- **Adaptability**: Can adapt to specific domain vocabularies
- **Data-Driven**: Uses real examples to improve prompts
- **Safe Migration**: No impact on existing functionality

## Requirements

- Python 3.8+
- dspy-ai>=2.4.0
- All existing dbxmetagen dependencies
- OpenAI API access (for prompt optimization)

## Support

The DSPy integration is designed to be completely backwards compatible. If you encounter any issues:

1. DSPy will gracefully fallback to traditional methods
2. All existing functionality remains unchanged
3. You can disable DSPy usage at any time
4. Traditional and DSPy generators can coexist

For detailed documentation, see `docs/DSPY_INTEGRATION.md`.
For examples, see `examples/dspy_usage_example.py`.
For tests, see `tests/test_dspy_integration.py`.

## Next Steps

Once you confirm DSPy is working well, you can:

1. Collect examples from your actual data usage
2. Run prompt optimization on domain-specific examples
3. Compare quality between traditional and optimized approaches
4. Gradually increase DSPy usage based on results
5. Eventually replace the traditional approach if desired

The integration provides a safe, gradual path to adopting DSPy optimization while maintaining full system compatibility.
