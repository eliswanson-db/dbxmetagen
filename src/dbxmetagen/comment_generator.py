import mlflow
from src.dbxmetagen.metadata_generator import CommentGenerator
from src.dbxmetagen.chat_client import ChatClientFactory


class CommentGeneratorModel(CommentGenerator, mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, model_input, params=None):
        if type(self.config) != dict:
            self.config = self.config.__dict__

        # Convert dict back to MetadataConfig-like object for chat client
        class TempConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

        temp_config = TempConfig(self.config)
        if not hasattr(self, "chat_client"):
            self.chat_client = ChatClientFactory.create_client(temp_config)

        self.chat_response = self.chat_client.create_completion(
            messages=model_input,
            model=self.config["model"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
        )
        return self.chat_response
