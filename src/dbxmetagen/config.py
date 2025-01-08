class MetadataConfig:
    ACRO_CONTENT = {
        "DBX": "Databricks"
        }
    SETUP_PARAMS = {
        "base_url": "https://adb-830292400663869.9.azuredatabricks.net",
        "catalog": "dbxmetagen",
        "catalog_tokenizable": "__CATALOG_NAME__", #"__CATALOG_NAME___{{env}}", # __CATALOG_NAME__
        "model": "databricks-meta-llama-3-1-70b-instruct",
        "registered_model_name": None,
        "model_type": None,
        "volume_name": "generated_metadata",
        "acro_content": ACRO_CONTENT,
        "table_names_source": "csv_file_path", #could also use a list of table names in the config
        "source_file_path": "table_names.csv",
        "control_table": "metadata_control",
        "apply_ddl": False, 
        "dry_run": False, #What should a dry run actually do? Provide test data?
        "pi_classification_rules": """if the content of a column is like {"appointment_text": "John Johnson has high blood pressure"}, then the classification should be "phi" and if a column appears to be a credit card number or a bank account number then it should be labeled as "pci". 'type' values allowed include 'name', 'location', 'national ID', 'email', and 'phone'. If the confidence is less than 0.5, then the classification should be 'none'."""
    }
    MODEL_PARAMS = {
        "max_prompt_length": 5000,
        "columns_per_call": 5,
        "sample_size": 5,
        "max_tokens": 5000,
        "temperature": 0.1,
        "add_metadata": True,
    }
    
    def __init__(self, **kwargs):
        self.setup_params = self.__class__.SETUP_PARAMS
        self.model_params = self.__class__.MODEL_PARAMS


        for key, value in self.setup_params.items():
            setattr(self, key, value)
        
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        

