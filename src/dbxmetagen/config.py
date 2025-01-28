import yaml

class MetadataConfig:
    ACRO_CONTENT = {
        "DBX": "Databricks",
        "WHO": "World Health Organization",       
        }
    SETUP_PARAMS = {
        "yaml_file_path": "../databricks_variables.yml",
        "yaml_variable_names": ['dev_host', 'prod_host', 'catalog_name', 'schema_name', 'catalog_tokenizable', 'model', 'registered_model_name', 'model_type', 'volume_name', 'table_names_source', 'source_file_path', 'control_table', 'apply_ddl', 'allow_data', 'dry_run', 'pi_classification_rules', 'allow_manual_override', 'tag_none_fields', 'pi_column_field_names', 'max_prompt_length', 'columns_per_call', 'sample_size', 'max_tokens', 'temperature', 'add_metadata', 'acro_content']
    }
    MODEL_PARAMS = {
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
        
        self.instantiate_environments()

    def load_yaml(self):
        with open(self.yaml_file_path, 'r') as file:
            variables = yaml.safe_load(file)
        selected_variables = {key: variables['variables'][key]['default'] for key in self.yaml_variable_names if key in variables['variables']}
        print("selected variables", selected_variables)
        return selected_variables
    
    def instantiate_environments(self):
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
        

