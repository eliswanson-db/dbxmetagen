import os
from src.dbxmetagen.comment_generator import CommentGeneratorModel
from src.dbxmetagen.config import MetadataConfig
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow
import mlflow
from src.dbxmetagen.metadata_generator import CommentGenerator
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from mlflow.types.llm import ChatResponse, ChatChoice, ChatChoiceLogProbs
from src.dbxmetagen.prompts import Prompt
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse
from openai import OpenAI
import pandas as pd
from openai.types.chat.chat_completion import ChatCompletion
from mlflow.types.llm import ChatResponse, ChatChoice, ChatChoiceLogProbs, ChatMessage


def convert_to_chat_response(chat_completion):
    choices = [
        ChatChoice(
            index=choice.index,
            message=ChatMessage(role=choice.message.role, content=choice.message.content) if isinstance(choice.message, ChatCompletionMessage) else choice.message,
            finish_reason=choice.finish_reason
        )
        for choice in chat_completion.choices
    ]

    return ChatResponse(
        id=chat_completion.id,
        object=chat_completion.object,
        created=chat_completion.created,
        model=chat_completion.model,
        choices=choices,
        usage=chat_completion.usage.to_dict() if hasattr(chat_completion.usage, 'to_dict') else chat_completion.usage
    )

def get_latest_model_version(model_name: str) -> int:
    """
    Retrieves the latest version of the specified registered model.

    Args:
        model_name (str): The name of the registered model.

    Returns:
        int: The latest version number of the registered model.
    """
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    latest_version = max(int(version.version) for version in model_versions)
    return latest_version
