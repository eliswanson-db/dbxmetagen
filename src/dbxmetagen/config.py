ACRO_CONTENT = "{'DBX': 'Databricks'}"
class MetadataConfig:
    ACRO_CONTENT = {"DBX": "Databricks"}
    SETUP_PARAMS = {
        "base_url": "https://adb-830292400663869.9.azuredatabricks.net",
        "catalog": "dbxmetagen",
        "catalog_tokenizable": "__CATALOG_NAME__", #"__CATALOG_NAME___{{env}}", # __CATALOG_NAME__
        "model": "databricks-meta-llama-3-1-70b-instruct",
        "max_prompt_length": 5000,
        "volume_name": "generated_metadata",
        "acro_content": ACRO_CONTENT,
        "columns_per_call": 100,
        "sample_size": 10,
        "max_tokens": 5000,
        "temperature": 0.1,
        "table_names_source": "csv_file_path", #could also use a list of table names in the config
        "source_file_path": "table_names.csv",
        "control_table": "metadata_control",
        "add_metadata": True,
        "ddl_application": "ddl" #direct other option
    }
    
    def __init__(self, **kwargs):
        self.setup_params = self.__class__.SETUP_PARAMS
        
        for key, value in kwargs.items():
            setattr(self, key, value)
