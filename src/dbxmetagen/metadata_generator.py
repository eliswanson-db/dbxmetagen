import os
from abc import ABC, abstractmethod
import json
from pyspark.sql import SparkSession
from pydantic import ValidationError
from typing import Dict, Tuple
from openai.types.chat.chat_completion import Choice, ChatCompletion, ChatCompletionMessage
from mlflow.types.llm import TokenUsageStats, ChatResponse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from pydantic.types import List, Any
from src.dbxmetagen.config import MetadataConfig
from src.dbxmetagen.error_handling import exponential_backoff
from src.dbxmetagen.prompts import create_prompt_template



class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    columns: List[str]

    
class PIResponse(Response):
    model_config = ConfigDict(extra="forbid")

    column_contents: list[dict[str, Any]]


class CommentResponse(Response):
    model_config = ConfigDict(extra="forbid")

    column_contents: list[str]


class SummaryCommentResponse(Response):
    pass

class MetadataGenerator(ABC):
    def __init__(self, config, df, full_table_name):
        self.config = config
        self.df = df
        self.full_table_name = full_table_name

    @abstractmethod
    def get_responses(self) -> Tuple[Response, ChatCompletion]:
        pass


class CommentGenerator(MetadataGenerator):
    def __init__(self, config, df, full_table_name):
        super().__init__(config, df, full_table_name)
        self.comment_input_data = self.convert_to_comment_input()
        if self.config.add_metadata:
            self.add_metadata_to_comment_input()
    
    @property
    def openai_client(self):
        return OpenAI(api_key=os.environ["DATABRICKS_TOKEN"], base_url=os.environ["DATABRICKS_HOST"] + "/serving-endpoints")

    def convert_to_comment_input(self) -> Dict[str, Any]:
        return {
            "table_name": self.full_table_name,
            "column_contents": self.df.toPandas().to_dict(orient='split'),
        }

    def get_responses(self) -> Tuple[CommentResponse, ChatCompletion]:
        prompt_template = create_prompt_template(self.comment_input_data, self.config.acro_content)
        if len(prompt_template) > self.config.max_prompt_length:
            raise ValueError("The prompt template is too long. Please reduce the number of columns or increase the max_prompt_length.")
        
        comment_response, message_payload = self.get_comment_response(
            self.config, 
            content=self.comment_input_data, 
            prompt_content=prompt_template['comment'], 
            model=self.config.model, 
            max_tokens=self.config.max_tokens, 
            temperature=self.config.temperature
        )
        return comment_response, message_payload

    def add_metadata_to_comment_input(self) -> None:
        spark = SparkSession.builder.getOrCreate()
        column_metadata_dict = {}
        for column_name in self.comment_input_data['column_contents']['columns']:
            extended_metadata_df = spark.sql(
                f"DESCRIBE EXTENDED {self.full_table_name} {column_name}"
            )            
            filtered_metadata_df = extended_metadata_df.filter(extended_metadata_df["info_value"] != "NULL") \
                                                       .filter(extended_metadata_df["info_name"] != "description") \
                                                       .filter(extended_metadata_df["info_name"] != "comment")
            column_metadata = filtered_metadata_df.toPandas().to_dict(orient='list')
            combined_metadata = dict(zip(column_metadata['info_name'], column_metadata['info_value']))
            column_metadata_dict[column_name] = combined_metadata
            
        self.comment_input_data['column_contents']['column_metadata'] = column_metadata_dict

    def predict(self, prompt_content):
        self.chat_response = self.openai_client.chat.completions.create(
            messages=prompt_content,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return self.chat_response

    def get_comment_response(self, 
                             config: MetadataConfig,
                             content: str, 
                             prompt_content: str, 
                             model: str, 
                             max_tokens: int, 
                             temperature: float,
                             retries: int = 0, 
                             max_retries: int = 5) -> Tuple[CommentResponse, Dict[str, Any]]:
        try:
            chat_completion = self._get_chat_completion(config, prompt_content, model, max_tokens, temperature)
            response_payload = chat_completion.choices[0].message
            response_dict = self._parse_response(response_payload.content)
            self._validate_response(content, response_dict)
            chat_response = CommentResponse(**response_dict)
            return chat_response, response_payload
        except (ValidationError, json.JSONDecodeError, AttributeError, ValueError) as e:
            if retries < max_retries:
                print(f"Attempt {retries + 1} failed for {response_payload.content}, retrying due to {e}...")
                return self.get_comment_response(config, content, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print("Validation error - response")
                raise ValueError(f"Validation error after {max_retries} attempts: {e}")

    def _get_chat_completion(self, config: MetadataConfig, prompt_content: str, model: str, max_tokens: int, temperature: float, retries: int = 0, max_retries: int = 3) -> ChatCompletion:
        try:
            return self.predict(prompt_content)
        except Exception as e:
            if retries < max_retries:
                print(f"Error: {e}. Retrying in {2 ** retries} seconds...")
                exponential_backoff(retries)
                return self._get_chat_completion(config, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print(f"Failed after {max_retries} retries.")
                raise e

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response_dict = json.loads(response)
            if not isinstance(response_dict, dict):
                raise ValueError("Response is not a valid dict")
            return response_dict
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}")

    def _validate_response(self, content: str, response_dict: Dict[str, Any]) -> None:
        if not self._check_list_and_dict_keys_match(content['column_contents']['columns'], response_dict['columns']):
            raise ValueError("Column names do not match column contents")
    
    @staticmethod
    def _check_list_and_dict_keys_match(dict_list, string_list):
        if isinstance(dict_list, list):
            dict_keys = dict_list
        else:
            try:
                dict_keys = dict_list.keys()
            except: 
                raise TypeError("dict_list is not a list or a dictionary")
        list_matches_keys = all(item in dict_keys for item in string_list)
        keys_match_list = all(key in string_list for key in dict_keys)
        if not (list_matches_keys and keys_match_list):
            return False
        return True


class PIIdentifier(MetadataGenerator):
    def __init__(self, config, df, full_table_name):
        super().__init__(config, df, full_table_name)

    def get_responses(self) -> Tuple[PIResponse, PIResponse]:
        prompt_template = create_prompt_template(self.df, self.config.acro_content)
        if len(prompt_template) > self.config.max_prompt_length:
            raise ValueError("The prompt template is too long. Please reduce the number of columns or increase the max_prompt_length.")
        pi_response, message_payload = self.get_pi_response(
            self.config, 
            content=self.df, 
            prompt_content=prompt_template['pi'], 
            model=self.config.model, 
            max_tokens=self.config.max_tokens, 
            temperature=self.config.temperature
        )
        return pi_response, message_payload

    def get_pi_response(self, 
                        config: MetadataConfig,
                        content: str, 
                        prompt_content: str, 
                        model: str, 
                        max_tokens: int, 
                        temperature: float,
                        retries: int = 0, 
                        max_retries: int = 5) -> Tuple[PIResponse, Dict[str, Any]]:
        try:
            chat_completion = self._get_chat_completion(config, prompt_content, model, max_tokens, temperature)
            response_payload = chat_completion.choices[0].message
            response_dict = self._parse_response(response_payload.content)
            self._validate_response(content, response_dict)
            chat_response = PIResponse(**response_dict)
            return chat_response, response_payload
        except (ValidationError, json.JSONDecodeError, AttributeError, ValueError) as e:
            if retries < max_retries:
                print(f"Attempt {retries + 1} failed for {response_payload.content}, retrying due to {e}...")
                return self.get_pi_response(config, content, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print("Validation error - response")
                raise ValueError(f"Validation error after {max_retries} attempts: {e}")

    def _get_chat_completion(self, config: MetadataConfig, prompt_content: str, model: str, max_tokens: int, temperature: float, retries: int = 0, max_retries: int = 3) -> ChatCompletion:
        try:
            return self.predict(prompt_content)
        except Exception as e:
            if retries < max_retries:
                print(f"Error: {e}. Retrying in {2 ** retries} seconds...")
                exponential_backoff(retries)
                return self._get_chat_completion(config, prompt_content, model, max_tokens, temperature, retries + 1, max_retries)
            else:
                print(f"Failed after {max_retries} retries.")
                raise e

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            response_dict = json.loads(response)
            if not isinstance(response_dict, dict):
                raise ValueError("Response is not a valid dict")
            return response_dict
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e}")

    def _validate_response(self, content: str, response_dict: Dict[str, Any]) -> None:
        if not self._check_list_and_dict_keys_match(content['column_contents']['columns'], response_dict['columns']):
            raise ValueError("Column names do not match column contents")
    
    @staticmethod
    def _check_list_and_dict_keys_match(dict_list, string_list):
        if isinstance(dict_list, list):
            dict_keys = dict_list
        else:
            try:
                dict_keys = dict_list.keys()
            except: 
                raise TypeError("dict_list is not a list or a dictionary")
        list_matches_keys = all(item in dict_keys for item in string_list)
        keys_match_list = all(key in string_list for key in dict_keys)
        if not (list_matches_keys and keys_match_list):
            return False
        return True


class MetadataGeneratorFactory:
    @staticmethod
    def create_generator(config, df, full_table_name) -> MetadataGenerator:
        if config.mode == "comment":
            return CommentGenerator(config, df, full_table_name)
        elif config.mode == "pi":
            return PIIdentifier(config, df, full_table_name)
        else:
            raise ValueError("Invalid mode. Use 'pi' or 'comment'.")
