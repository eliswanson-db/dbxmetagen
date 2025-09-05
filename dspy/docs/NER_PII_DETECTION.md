# NER and PII/PHI Detection Implementation

This document describes the NER and PII/PHI detection solution added to dbxmetagen.

## Overview

The solution provides:
- Named Entity Recognition (NER) using GLiNER-biomed model from HuggingFace
- PII/PHI detection and redaction using both GLiNER and Presidio
- MLflow pyfunc model registration to Unity Catalog
- Efficient batch processing with pandas UDF and iterator pattern
- Synthetic data generation for testing and evaluation

## Architecture

### Components

1. **GLiNERNERModel**: MLflow pyfunc model that wraps the GLiNER-biomed model
2. **BatchNERProcessor**: Handles efficient batch processing using pandas UDF
3. **SyntheticDataGenerator**: Generates test data with ground truth labels
4. **Performance optimizations**: HuggingFace caching, iterator pattern, deduplication

### Model Features

- **Dual Detection Engine**: Combines GLiNER-biomed with Presidio for comprehensive coverage
- **Domain-Specific Labels**: Pre-configured for PII (personal identifiable information) and PHI (protected health information)
- **Intelligent Deduplication**: Removes overlapping entities based on span overlap
- **Configurable Thresholds**: Adjustable confidence thresholds for entity detection

## Usage

### Basic Usage

```python
# Initialize configuration
config = NERConfig(
    model_name="Ihor/gliner-biomed-base-v1.0",
    cache_dir="/dbfs/tmp/hf_cache",
    threshold=0.5
)

# Train and register model
model_uri = train_and_register_model(config, "catalog.schema.ner_model")

# Process data
processor = BatchNERProcessor(
    model_uri_path=f"models:/catalog.schema.ner_model/latest",
    ner_config=config
)
results = processor.process_dataframe(source_df, "text_column")
```

### Notebook Parameters

The notebook accepts these widget parameters:

- `catalog_name`: Unity Catalog name
- `schema_name`: Schema name  
- `source_table`: Source table with text data
- `results_table`: Table to store NER results
- `model_name`: Registered model name
- `hf_cache_dir`: HuggingFace model cache directory
- `generate_data`: Whether to generate synthetic test data

## Performance Optimizations

### Caching Strategy
- HuggingFace models cached using Unity Catalog Volumes (not DBFS)
- Default cache location: `/Volumes/dbxmetagen/default/models/hf_cache`
- Model loaded once per executor using iterator pattern
- Persistent cache across notebook runs with proper UC Volume permissions

### Batch Processing
- Pandas UDF with iterator pattern for memory efficiency
- Configurable batch sizes
- Parallel processing across Spark partitions

### Memory Management
- Streaming processing of large datasets
- Efficient entity deduplication algorithms
- Minimal memory footprint per batch

## Entity Types

### PII Labels
- person, email, phone number
- Social Security Number, driver licence, passport number
- credit card, full address, date of birth
- personally identifiable information

### PHI Labels  
- patient, medical record number, diagnosis
- medication, doctor, hospital, medical condition
- treatment, lab test, lab test value, dosage
- drug, prescription

## Data Schema

### Input Schema
```
- id: integer
- clean_text: string (no PII/PHI)
- pii_text: string (contains PII)  
- phi_text: string (contains PHI/medical data)
```

### Output Schema
```
- Original columns plus:
- detected_entities: string (JSON array of entities)
- redacted_text: string (text with entities replaced)
- entity_count: integer (number of detected entities)
```

## Evaluation and Metrics

The solution includes:
- Ground truth generation for evaluation
- Precision/recall metrics calculation
- Entity detection accuracy assessment  
- Redaction quality validation

## Best Practices

1. **Threshold Tuning**: Start with 0.5, adjust based on precision/recall needs
2. **Batch Size**: Optimize based on cluster memory (16-32 recommended)
3. **Cache Management**: Use persistent storage for production deployments
4. **Model Updates**: Regularly update to latest GLiNER-biomed versions
5. **Monitoring**: Track entity detection rates and model performance

## Deployment Considerations

### Development
- Use smaller batch sizes for faster iteration
- Enable verbose logging for debugging
- Test with synthetic data first
- Ensure Unity Catalog Volume access for caching

### Production
- Use GPU-enabled clusters for faster inference
- Implement model monitoring and drift detection
- Set up automated cache warming with UC Volumes
- Configure appropriate retry strategies
- Set proper UC permissions on volumes, schema, and catalog

## Dependencies

Core libraries:
- `gliner==0.2.5`: GLiNER model framework
- `transformers==4.44.0`: HuggingFace transformers
- `torch==2.4.0`: PyTorch backend
- `presidio-analyzer==2.2.358`: Microsoft Presidio PII detection
- `presidio-anonymizer==2.2.358`: Microsoft Presidio anonymization

## References

- [GLiNER-BioMed Paper](https://arxiv.org/abs/2504.00676)
- [GLiNER-BioMed Model](https://huggingface.co/Ihor/gliner-biomed-base-v1.0)
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
