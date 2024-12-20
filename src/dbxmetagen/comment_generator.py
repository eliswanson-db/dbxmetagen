import mlflow
from metadata_generator import CommentGenerator
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from prompts import create_prompt_template
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse


class CustomCommentGenerator(CommentGenerator, mlflow.pyfunc.ChatModel):
    def load_context(self, context):
        self.config = MetadataConfig(**context.model_config)
    
    def predict(self, context, prompt_content):
        return self.get_responses(self.config, prompt_content, prompt_content)
