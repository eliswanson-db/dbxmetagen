import yaml

class MetadataConfig:
    ACRO_CONTENT = {
        "DBX": "Databricks",
        "WHO": "World Health Organization",       
        }
    SETUP_PARAMS = {
        "yaml_file_path": "../resources/variables/variables.yml",
        "yaml_variable_names": ['dev_host', 'prod_host', 'catalog_name', 'schema_name'],
        "catalog_tokenizable": "__CATALOG_NAME__", #"__CATALOG_NAME___{{env}}", # __CATALOG_NAME__
        "model": "databricks-meta-llama-3-1-70b-instruct",
        "registered_model_name": None,
        "model_type": None,
        "volume_name": "generated_metadata",
        "acro_content": ACRO_CONTENT,
        "table_names_source": "csv_file_path", #could also use a list of table names in the config
        "source_file_path": "table_names.csv",
        "control_table": "metadata_control",
        "apply_ddl": False, # If true, the DDL for the tables will be applied. 
        "allow_data": True, # Determines whether data can be allowed to be displayed in a comment. Turn off if data is not allowed to be displayed. Note that comments still need to be reviewed by a human for any guarantees, but the prompt will be constructed to make a best effort at not including data in the comments.
        "dry_run": False, #What should a dry run actually do? Provide test data?
        "pi_classification_rules": """If a column has only PII, then it should be considered only PII, even if other columns in the table have medical information in them. A specific example of this is that if a column only contains name, or address, then it should be marked as pii, not as phi. However, the column containing medical information would be considered PHI because it pertains to the health status, provision of health care, or payment for health care that can be linked to an individual.""", ###The table as a whole though would be considered PHI because it contains both identifiable information and health-related information. These rules are only used in PI identification and classification.
        "allow_manual_override": True,
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
        print(kwargs)
        self.setup_params = self.__class__.SETUP_PARAMS
        self.model_params = self.__class__.MODEL_PARAMS


        for key, value in self.setup_params.items():
            setattr(self, key, value)
        
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        yaml_variables = self.load_yaml()
        print("yaml_variables:", yaml_variables)
        for key, value in yaml_variables.items():
            setattr(self, key, value)

        if self.env == "dev":
            self.base_url = self.dev_host
        elif self.env == "qa":
            self.base_url = self.qa_host
        elif self.env == "test":
            self.base_url = self.test_host
        elif self.env == "stg":
            self.base_url = self.stg_host
        elif self.env == "prod":
            self.base_url = self.prod_host
        else:
            raise Exception(f"Environment {self.env} does not match any provided host in variables.yml.")
    
    def load_yaml(self):
        with open(self.yaml_file_path, 'r') as file:
            variables = yaml.safe_load(file)
        print("variables:", variables)
        print("self yaml variable names", self.yaml_variable_names)
    
        selected_variables = {key: variables['variables'][key]['default'] for key in self.yaml_variable_names if key in variables['variables']}
        print("Selected variables:", selected_variables)
        return selected_variables
        

