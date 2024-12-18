import mlflow
from metadata_generator import CommentChatCompletionResponse
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from prompts import create_prompt_template
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.metadata_generator import PIResponse


class CommentGeneratorModel(mlflow.pyfunc.ChatModel, CommentChatCompletionResponse, ABC):
    _openai_client = None

    @property
    def openai_client(self):
        return OpenAI(api_key=os.environ["DATABRICKS_TOKEN"],
                      base_url=self.config.base_url + "/serving-endpoints")

    @classmethod
    def from_context(cls, context: mlflow.pyfunc.PythonModelContext):
        chat = cls.__new__(cls)
        return chat
    
    @staticmethod
    def _format_prompt(prompt, content, acro_content):
        return create_prompt_template(prompt, content, acro_content)

    def get_openai_client(self):
       return OpenAI(
           api_key=os.environ["DATABRICKS_TOKEN"],
           base_url=os.environ["DATABRICKS_HOST"] + "/serving-endpoints")

    def load_context(self, context):
        """Instantiated OpenAI client cannot be added to load_context.
        """
        self.api_key = context.artifacts["api_key"]
        self.base_url = context.artifacts["base_url"]

    def get_response(config, content: str, prompt_content: str, model: str, max_tokens: str, temperature: str):
        client = OpenAI(
            api_key=os.environ['DATABRICKS_TOKEN'],
            base_url=config.base_url
        )
        chat_completion = client.chat.completions.create(
            messages=prompt_content,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return chat_completion

    def predict(self, context, messages):
        self.client = self.get_openai_client()
        prompt = self.create_prompt(messages)
        response = self.client.chat.completions.create(
            messages=prompt,
            model=context.get("model", "default-model"),
            max_tokens=context.get("max_tokens", 3000),
            temperature=context.get("temperature", 0.1)
        )
        text = response.choices[0].message["content"]

        prompt_tokens = len(self.client.tokenizer.encode(prompt))
        completion_tokens = len(self.client.tokenizer.encode(text))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        
        chat_response = {
            "id": f"response_{random.randint(0, 100)}",
            "model": "MyChatModel",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        
        return ChatResponse(**chat_response)
    
    def summarize_table_comments(self):


    def create_prompt(self, messages):
        # Convert the list of messages to the format expected by the OpenAI API
        return [{"role": message["role"], "content": message["content"]} for message in messages]
