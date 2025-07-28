# DBX MetaGen Streamlit App

A production-quality Databricks app for AI-powered metadata generation, built with Streamlit.

## Features

- 🏗️ **Configuration Management**: Easy setup of dbxmetagen parameters through intuitive UI
- 📋 **Table Management**: Upload CSV files or manually specify tables for processing
- 🚀 **Job Management**: Create, run, and monitor metadata generation jobs
- 📊 **Results Viewing**: Browse generated metadata, download files, and review table comments
- 🔧 **Real-time Monitoring**: Track job progress with automatic status updates
- 📁 **File Processing**: View and download TSV, SQL, and CSV output files

## Quick Start

### 1. Deploy the App

Deploy using Databricks Asset Bundle:

```bash
cd dbxmetagen
databricks bundle deploy
```

### 2. Configure Secrets

Create a Databricks secret scope and token:

```bash
# Create secret scope
databricks secrets create-scope dbxmetagen

# Add your Databricks token 
databricks secrets put-secret --json '{"scope": "dbxmetagen", "key": "databricks_token", "string_value": "YOUR_TOKEN_HERE"}'
```

**Note**: This requires Databricks CLI version 0.234 or higher. If you're using an older version, please update:
```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

### 3. Access the App

After deployment, access your app through the Databricks workspace.

## Configuration

### Core Settings
- **Catalog Name**: Target catalog for storing metadata results
- **Databricks Host**: Your workspace URL
- **Schema Name**: Schema where results will be stored

### Data Processing Settings
- **Allow Data**: Whether to include actual data samples in LLM processing
- **Sample Size**: Number of data rows to sample per column (0 = no data sampling)
- **Mode**: Choose between generating comments, identifying PII, or both

### Advanced Settings
- **LLM Model**: Choose from available Databricks models
- **Apply DDL**: ⚠️ **WARNING** - This will directly modify your tables
- **Temperature**: Model creativity level (0.0 = deterministic, 1.0 = creative)

## Usage

### 1. Configure Settings
Use the sidebar to configure your processing parameters. You can:
- Load default settings from `variables.yml`
- Save and load custom configurations
- Validate settings before processing

### 2. Select Tables
Choose tables for processing by:
- **Manual Input**: Enter table names directly (one per line)
- **CSV Upload**: Upload a `table_names.csv` file

### 3. Create and Monitor Jobs
- Configure job parameters (name, cluster size)
- Create and run metadata generation jobs
- Monitor progress in real-time
- View detailed job information in Databricks

### 4. View Results
- Browse generated metadata files
- Download TSV, SQL, and CSV outputs
- View table metadata and comments
- Review processing statistics

## File Formats

### Input Files
- **table_names.csv**: CSV file with 'table_name' column containing fully qualified table names
- **Configuration YAML**: Saved configuration files for reuse

### Output Files
- **Generated TSV**: Tab-separated file with metadata results
- **Generated SQL**: DDL statements to apply metadata to tables
- **CSV Exports**: Filtered and processed data exports

## Architecture

```
dbxmetagen/app/
├── app.py              # Main Streamlit application
├── utils.py            # Utility classes and functions
├── config.yml          # App configuration
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Components

- **DBXMetaGenApp**: Main application class managing UI and orchestration
- **JobManager**: Handles Databricks job creation and monitoring
- **ConfigManager**: Manages configuration loading and validation
- **TableManager**: Handles table validation and processing
- **ResultsManager**: Manages results viewing and file processing

## Deployment Configuration

The app is configured for deployment via Databricks Asset Bundle in `resources/apps/dbxmetagen_app.yml`.

### Resource Requirements
- **CPU**: 1 core
- **Memory**: 2Gi
- **Runtime**: Streamlit with Python environment

### Environment Variables
- `DATABRICKS_HOST`: Automatically set from bundle variables
- `DATABRICKS_TOKEN`: Retrieved from secret scope

## Troubleshooting

### Authentication Issues
- Ensure Databricks token is properly configured in secret scope
- Verify host URL format in configuration

### Table Access
- Confirm read access to specified tables
- Check table name format (catalog.schema.table)

### Job Failures
- Review job logs in Databricks workspace for detailed errors
- Verify cluster configuration and resource availability
- Check table permissions and data access

### Performance Issues
- Reduce sample size for large tables
- Use smaller cluster configurations for testing
- Consider processing tables in smaller batches

## Security Considerations

- **Data Privacy**: Configure `allow_data` and `sample_size` based on your data sensitivity
- **Token Security**: Use Databricks secret scope for token storage
- **Table Access**: App uses your Databricks permissions for table access
- **DDL Application**: Use `apply_ddl=false` for testing to avoid modifying tables

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review job logs in Databricks workspace
3. Validate configuration settings
4. Ensure proper permissions and access

## Version

Version 1.0.0 - Initial release with core functionality for metadata generation and job management. 