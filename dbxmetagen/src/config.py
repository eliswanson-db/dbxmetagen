class MetadataConfig:
    ACRO_CONTENT = {"DBX": "Databricks"}
    SETUP_PARAMS = {
        "base_url": "https://adb-830292400663869.9.azuredatabricks.net/serving-endpoints",
        "catalog": "eswanson_genai",
        "catalog_tokenizable": "eswanson_genai_{{env}}",
        "table_names": [],
        "model": "databricks-meta-llama-3-1-70b-instruct",
        "max_prompt_length": 3000,
        "volume_name": "generated_metadata",
        "acro_content": ACRO_CONTENT,
        "columns_per_call": 10,
        "sample_size": 5,
        "max_tokens": 3000,
        "temperature": 0.1,
    }
    
    def __init__(self, **kwargs):
        self.setup_params = self.__class__.SETUP_PARAMS
        
        for key, value in kwargs.items():
            setattr(self, key, value)