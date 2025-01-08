import mlflow
from src.dbxmetagen.metadata_generator import CommentGenerator
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from src.dbxmetagen.prompts import Prompt
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse


class CommentGeneratorModel(CommentGenerator, mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def predict(self, model_input, params=None):
        if type(self.config) != dict:
            self.config = self.config.__dict__
        self.chat_response = self.openai_client.chat.completions.create(
            messages=model_input,
            model=self.config['model'],
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature']
        )
        return self.chat_response
