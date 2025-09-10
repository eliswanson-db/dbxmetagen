# PHI Redaction Models

## Overview

This notebook implements five different approaches for detecting and redacting PHI in medical text, each with specific strengths and use cases.

## Models & Approaches

### 1. GLiNER Biomed (`Ihor/gliner-biomed-base-v1.0`)
- **Type**: Medical-domain generalist NER
- **Strengths**: Flexible entity detection, medical context understanding
- **Use Case**: When you need custom PHI entity lists with medical context
- **Detection**: Person names, locations, organizations, custom PHI identifiers

### 2. BioBERT + Presidio (`dmis-lab/biobert-v1.1`)
- **Type**: Hybrid approach combining medical NER + PII detection
- **Strengths**: Medical terminology + structured PII patterns
- **Use Case**: Maximum coverage by combining domain expertise with PII-specific tools
- **Detection**: Medical entities (BioBERT) + PII patterns (Presidio) + custom regex

### 3. Presidio-Only (Microsoft Presidio)
- **Type**: PII-focused detection library
- **Strengths**: Purpose-built for PII, extensive pattern library, custom recognizers
- **Use Case**: When you need comprehensive PII detection with minimal false positives
- **Detection**: Phone, email, SSN, addresses, medical record numbers, dates

### 4. DistilBERT NER (`dslim/distilbert-NER`)
- **Type**: General-purpose NER optimized for PII-relevant entities
- **Strengths**: Fast, efficient, trained on CoNLL-2003 (PER, LOC, ORG, MISC)
- **Use Case**: Best balance of speed and accuracy for standard PII detection
- **Detection**: Person names, locations, organizations, miscellaneous identifiers

### 5. Claude Sonnet (Databricks Foundation Models)
- **Type**: Large Language Model via API
- **Strengths**: Context understanding, flexible detection, natural language reasoning
- **Use Case**: Complex edge cases requiring contextual understanding
- **Detection**: All PHI types via prompt-based instruction following

## Community Model Risk Assessment

### Realistic Risk Evaluation

**Low-Medium Risk** for PHI applications when properly implemented:

#### Why Community Models Are Generally Acceptable:
- **Standard Practice**: Most enterprise NER solutions use community/open-source models as base
- **Transparency**: Open models allow inspection and validation vs. black box commercial solutions
- **Validation**: Model behavior can be tested extensively on your specific data
- **Established Models**: GLiNER, DistilBERT, BioBERT have extensive usage and validation

#### Risk Mitigation Strategies:
1. **Thorough Testing**: Validate all models on representative PHI data before production
2. **Human Review**: Always include human validation loop for PHI compliance
3. **Multiple Models**: Use ensemble approaches to catch edge cases
4. **Regular Auditing**: Monitor model performance over time
5. **Data Handling**: Focus on secure data processing pipelines vs. model provenance

#### Model-Specific Considerations:

**GLiNER Biomed**:
- Risk: Community contribution, newer model
- Mitigation: Extensive medical training data, active development

**DistilBERT NER**:
- Risk: General-purpose training
- Mitigation: Well-established, high download count, proven performance

**BioBERT**:
- Risk: Domain mismatch (research vs. clinical text)
- Mitigation: Combined with Presidio, strong medical understanding

**Presidio**:
- Risk: Minimal - Microsoft-backed open source
- Mitigation: Enterprise support, extensive validation

**Claude**:
- Risk: External API dependency, cost
- Mitigation: Databricks-hosted, enterprise SLA

### Recommendation

For PHI applications, prioritize:
1. **Validation over Provenance**: Test thoroughly on your data
2. **Multi-layered Approach**: Combine models + regex + human review
3. **Proper Infrastructure**: Secure processing environment
4. **Compliance Focus**: Align with HIPAA/GDPR requirements vs. model source

The risk from using community models is generally **lower** than the risk from inadequate PHI detection coverage, but this is a risk that every organization must assess for themselves.

## Performance Characteristics

| Model | Speed | Accuracy | Recall | Precision | Use Case |
|-------|-------|----------|--------|-----------|----------|
| DistilBERT | Fast | High | Good | High | Standard PII |
| GLiNER | Medium | High | High | Good | Medical PII |
| Presidio | Fast | Good | High | High | Structured PII |
| BioBERT+Presidio | Slow | High | High | Medium | Comprehensive |
| Claude | Slow | Variable | High | Medium | Complex cases |

## Usage Recommendations

- **Standard deployment**: DistilBERT + custom regex
- **Medical-heavy content**: GLiNER Biomed
- **Maximum coverage**: BioBERT + Presidio
- **High-volume processing**: Presidio-only
- **Complex edge cases**: Claude as fallback
