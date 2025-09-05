# DSPy Training in Databricks

## Quick Start

Upload the `dspy_training_quickstart.py` notebook to your Databricks workspace and run it cell by cell.

## Training Process Overview

### 1. **Setup** (Cells 1-2)
- Install DSPy: `%pip install dspy-ai>=2.4.0`
- Import required libraries
- Initialize Spark session

### 2. **Configuration** (Cell 3)
```python
# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("your-scope", "openai-api-key")

# Or use environment variable
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### 3. **Create Training Data** (Cells 4-5)
```python
# Option 1: Use your real tables
sample_tables = [
    ("your_catalog", "your_schema", "your_table"),
    ("catalog2", "schema2", "table2")
]

# Option 2: Use mock data (included in notebook)
```

### 4. **Generate Expected Responses** (Cell 6)
The notebook uses the traditional generator to create baseline responses for training.

### 5. **Train DSPy Model** (Cells 8-9)
```python
# This is where the magic happens!
optimized_module = dspy_model.optimize_prompts(
    train_examples=train_examples,
    validation_examples=val_examples,
    num_threads=2  # Adjust based on API rate limits
)
```

### 6. **Test and Compare** (Cells 10-11)
Compare traditional vs DSPy optimized responses side-by-side.

### 7. **Save Model** (Cell 12)
```python
# Save to DBFS for persistence
dspy_model.save_optimized_prompts("/dbfs/tmp/dspy_comment_generator_optimized.json")
```

## Key Training Parameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `num_threads` | 2-4 | Lower if hitting API rate limits |
| `model` | `gpt-4o-mini` | Cheaper for training, use `gpt-4` for production |
| `temperature` | 0.3 | Lower for consistent training |
| `max_tokens` | 2000 | Sufficient for most comment generation |

## Training Data Requirements

### Minimum Requirements
- **5-10 examples** for basic optimization
- **15-25 examples** for good performance
- **50+ examples** for production-quality optimization

### Quality Guidelines
- **Diverse Tables**: Include different domains (finance, HR, sales, etc.)
- **Varied Column Types**: String, numeric, date, boolean columns
- **Different Complexities**: Simple lookup tables to complex fact tables
- **Real Metadata**: Use actual column statistics and data types

## Production Integration

### Option 1: Environment-Controlled
```python
# Set environment variables in your cluster/job
os.environ['DBXMETAGEN_USE_DSPY'] = 'true'
os.environ['DBXMETAGEN_DSPY_PATTERNS'] = 'finance.,sales.,marketing.'

# Use enhanced factory
generator = EnhancedMetadataGeneratorFactory.create_generator_with_env_config(config)
```

### Option 2: Direct Integration
```python
# Load optimized model
generator = DSPyCommentGeneratorFactory.create_generator(config)
generator.load_optimized_prompts("/dbfs/path/to/optimized_model.json")

# Use like any other generator
response = generator.predict_chat_response(messages)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **API Rate Limits** | Reduce `num_threads` to 1-2 |
| **Memory Errors** | Process examples in smaller batches |
| **Poor Quality** | Review and improve training examples |
| **Import Errors** | Ensure DSPy is installed: `%pip install dspy-ai>=2.4.0` |

### Performance Tips

1. **Start Small**: Begin with 5-10 high-quality examples
2. **Iterate**: Add more examples and re-train periodically  
3. **Domain-Specific**: Create separate models for different data domains
4. **Monitor**: Track response quality in production

## Expected Training Time

| Examples | Time | Cost (est.) |
|----------|------|-------------|
| 5-10 | 5-10 min | $2-5 |
| 15-25 | 15-30 min | $5-15 |
| 50+ | 30-60 min | $15-40 |

*Times and costs depend on API response times and model choice*

## File Structure After Training

```
/dbfs/tmp/
├── dspy_comment_generator_optimized.json  # Your optimized model
└── training_examples.json                 # Backup of training data
```

## Next Steps

1. **Run the notebook** with your data
2. **Review results** and compare traditional vs DSPy
3. **Save your optimized model** to DBFS
4. **Integrate into production** using environment variables
5. **Monitor performance** and re-train as needed

## Questions?

- Check the main documentation: `docs/DSPY_INTEGRATION.md`
- Review examples: `examples/dspy_usage_example.py`
- Run tests: `tests/test_dspy_integration.py`
