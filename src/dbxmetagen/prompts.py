import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import collect_list, struct

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Prompt(ABC):
    def __init__(self, config: Any, df: DataFrame, full_table_name: str):
        """
        Initialize the Prompt class.

        Args:
            config (Any): Configuration object.
            df (DataFrame): Spark DataFrame.
            full_table_name (str): Full table name in the format 'catalog.schema.table'.
        """
        self.spark = SparkSession.builder.getOrCreate()
        self.config = config
        self.df = df
        self.full_table_name = full_table_name
        self.prompt_content = self.convert_to_comment_input()
        if self.config.add_metadata:
            self.add_metadata_to_comment_input()
        logger.debug("Instantiating chat completion response...")

    @abstractmethod
    def convert_to_comment_input(self) -> Dict[str, Any]:
        """
        Convert DataFrame to a dictionary format suitable for comment input.

        Returns:
            Dict[str, Any]: Dictionary containing table and column contents.
        """
        pass

    @abstractmethod
    def add_metadata_to_comment_input(self) -> None:
        """
        Add metadata to the comment input.
        """
        pass
    
    def calculate_cell_length(self, pandas_df) -> pd.DataFrame:
        """
        Calculate the length of every cell in the original DataFrame and truncate values longer than the word limit specified in the config.

        Returns:
            pd.DataFrame: Modified Pandas DataFrame with truncated values.
        """
        def truncate_value(value: str, word_limit: int) -> str:
            words = value.split()
            if len(words) > word_limit:
                return ' '.join(words[:word_limit])
            return value

        word_limit = getattr(self.config, 'word_limit_per_cell', 100) 
        pandas_df = self.df.toPandas()
        truncated_count = 0

        for column in pandas_df.columns:
            truncated_values = pandas_df[column].apply(lambda x: truncate_value(str(x), word_limit))
            truncation_flags = pandas_df[column].apply(lambda x: len(str(x).split()) > word_limit)
            pandas_df[column] = truncated_values
            truncated_count += truncation_flags.sum()

        if truncated_count > 0:
            print(f"{truncated_count} cells were truncated.")
            logger.info(f"{truncated_count} cells were truncated.")
        
        return pandas_df

    def filter_extended_metadata_fields(self, extended_metadata_df: DataFrame) -> DataFrame:
        """
        Filter extended metadata fields based on the mode.

        Args:
            extended_metadata_df (DataFrame): DataFrame containing extended metadata.

        Returns:
            DataFrame: Filtered DataFrame.
        """
        if self.config.mode == "pi":
            return extended_metadata_df.filter(extended_metadata_df["info_value"] != "NULL")
        elif self.config.mode == "comment":
            return extended_metadata_df.filter(
                (extended_metadata_df["info_value"] != "NULL") &
                (extended_metadata_df["info_name"] != "description") &
                (extended_metadata_df["info_name"] != "comment")
            )
        else:
            raise ValueError("Invalid mode provided. Please provide either 'pi' or 'comment'.")

    def add_metadata_to_comment_input(self) -> None:
        """
        Add metadata to the comment input.
        """
        column_metadata_dict = self.extract_column_metadata()
        table_metadata = self.get_table_metadata()
        self.add_table_metadata_to_column_contents(table_metadata)
        self.prompt_content['column_contents']['column_metadata'] = column_metadata_dict

    def extract_column_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract metadata for each column.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing metadata for each column.
        """
        column_metadata_dict = {}
        for column_name in self.prompt_content['column_contents']['columns']:
            extended_metadata_df = self.spark.sql(
                f"DESCRIBE EXTENDED {self.full_table_name} `{column_name}`"
            )
            filtered_metadata_df = self.filter_extended_metadata_fields(extended_metadata_df)
            column_metadata = filtered_metadata_df.toPandas().to_dict(orient='list')
            combined_metadata = dict(zip(column_metadata['info_name'], column_metadata['info_value']))
            combined_metadata = self.add_column_metadata_to_column_contents(column_name, combined_metadata)
            column_metadata_dict[column_name] = combined_metadata
        return column_metadata_dict

    def add_column_metadata_to_column_contents(self, column_name: str, combined_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add column metadata to the column contents.

        Args:
            column_name (str): Name of the column.
            combined_metadata (Dict[str, Any]): Combined metadata for the column.

        Returns:
            Dict[str, Any]: Updated combined metadata with column tags.
        """
        column_tags = self.get_column_tags()
        if column_name in column_tags:
            combined_metadata['tags'] = column_tags[column_name]
        return combined_metadata

    def add_table_metadata_to_column_contents(self, table_metadata: Tuple[Dict[str, str], str, str, str]) -> None:
        """
        Add table metadata to the column contents.

        Args:
            table_metadata (Tuple[Dict[str, str], str, str, str]): Tuple containing column tags, table tags, table constraints, and table comments.
        """
        column_tags, table_tags, table_constraints, table_comments = table_metadata
        self.prompt_content['column_contents']['table_tags'] = table_tags
        self.prompt_content['column_contents']['table_constraints'] = table_constraints
        self.prompt_content['column_contents']['table_comments'] = table_comments
        logger.debug(self.prompt_content)

    def get_column_tags(self) -> Dict[str, Dict[str, str]]:
        """
        Get column tags from the information schema.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary containing column tags.
        """
        catalog_name, schema_name, table_name = self.full_table_name.split('.')
        query = f"""
        SELECT catalog_name, schema_name, table_name, column_name, tag_name, tag_value
        FROM system.information_schema.column_tags
        WHERE catalog_name = '{catalog_name}'
        AND schema_name = '{schema_name}'
        AND table_name = '{table_name}';
        """
        result_df = self.spark.sql(query)
        column_tags = result_df.groupBy("column_name").agg(
            collect_list(struct("tag_name", "tag_value")).alias("tags")
        ).collect()
        column_tags_dict = {row["column_name"]: {tag["tag_name"]: tag["tag_value"] for tag in row["tags"]} for row in column_tags}
        logger.debug("column tags dict: %s", column_tags_dict)
        return column_tags_dict

    def get_table_tags(self) -> str:
        """
        Get table tags from the information schema.

        Returns:
            str: JSON string containing table tags.
        """
        catalog_name, schema_name, table_name = self.full_table_name.split('.')
        query = f"""
        SELECT tag_name, tag_value
        FROM system.information_schema.table_tags
        WHERE catalog_name = '{catalog_name}'
        AND schema_name = '{schema_name}'
        AND table_name = '{table_name}';
        """
        result_df = self.spark.sql(query)
        #logger.debug("table tags result: %s", result_df)
        return self.df_to_json(result_df)

    def get_table_constraints(self) -> str:
        """
        Get table constraints from the information schema.

        Returns:
            str: JSON string containing table constraints.
        """
        catalog_name, schema_name, table_name = self.full_table_name.split('.')
        query = f"""
        SELECT table_name, constraint_type
        FROM system.information_schema.table_constraints
        WHERE table_catalog = '{catalog_name}'
        AND table_schema = '{schema_name}'
        AND table_name = '{table_name}';
        """
        return self.df_to_json(self.spark.sql(query))

    def get_table_comment(self) -> str:
        """
        Get table comment from the information schema.

        Returns:
            str: JSON string containing table comment.
        """
        catalog_name, schema_name, table_name = self.full_table_name.split('.')
        query = f"""
        SELECT table_name, comment
        FROM system.information_schema.tables
        WHERE table_catalog = '{catalog_name}'
        AND table_schema = '{schema_name}'
        AND table_name = '{table_name}';
        """
        return self.df_to_json(self.spark.sql(query))

    def get_table_metadata(self) -> Tuple[Dict[str, Dict[str, str]], str, str, str]:
        """
        Get table metadata including column tags, table tags, table constraints, and table comments.

        Returns:
            Tuple[Dict[str, Dict[str, str]], str, str, str]: Tuple containing column tags, table tags, table constraints, and table comments.
        """
        column_tags = self.get_column_tags()
        table_tags = self.get_table_tags()
        table_constraints = self.get_table_constraints()
        table_comments = self.get_table_comment()
        return column_tags, table_tags, table_constraints, table_comments

    @staticmethod
    def df_to_json(df: DataFrame) -> str:
        """
        Convert DataFrame to JSON string.

        Args:
            df (DataFrame): Spark DataFrame.

        Returns:
            str: JSON string representation of the DataFrame.
        """
        if df.isEmpty():
            return {}
        else:
            json_response = df.toJSON().reduce(lambda x, y: x + ',' + y)
            json_response = '[' + json_response + ']'
            logger.debug("json response in prompt: %s", json_response)
        return json_response



class CommentPrompt(Prompt):
    def convert_to_comment_input(self) -> Dict[str, Any]:
        pandas_df = self.df.toPandas()
        if self.config.limit_prompt_based_on_cell_len:
            truncated_pandas_df = self.calculate_cell_length(pandas_df)
        else:
            truncated_pandas_df = pandas_df
        return {
            "table_name": self.full_table_name,
            "column_contents": truncated_pandas_df.to_dict(orient='split'),
        }

    def create_prompt_template(self) -> Dict[str, Any]:
        print("Creating comment prompt template...")
        content = self.prompt_content
        acro_content = self.config.acro_content
        return {
              "comment":
              [
                {
                    "role": "system",
                    "content": """You are an AI assistant helping to generate metadata for tables and columns in Databricks.


                    ###
                    Input will come in in this format:

                    ###
                    Input format:
                    {"index": [0, 1], "columns": ["name", "address", "email", "MRR", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$1545.50", "2024-03-05", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$124555.32", "2023-01-03", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'MRR': {'col_name': 'MRR', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}, 'eap_created': {'col_name': 'eap_created', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}.

                    ###
                    Please provide a response in this format:
                    ###
                    {"table": "Predictable recurring revenue earned from subscriptions in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer name. Based on the data available in the sample, this field appears to contain both first and last name.", "Customer mailing address including both the number and street name, but at least in the sampled data not including the city, state, country, or zipcode. Stored as a string and populated in all cases. The sample has two distinct values for two rows, but the metadata makes it clear that there are 5 distinct values in the table.", "Customer email address with domain name. This is a common format for email addresses. Domains seen include MSN and AOL. These are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "MRR (Monthly recurring revenue) from the customer in United States dollars with two decimals for cents. This field is never null and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the EAP (Enterprise Architecture Platform) or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column - based on the sample and the metadata it appears that every value is set to False, but based on the column metadata as a string rather than as a boolean value."]}

                    ###
                    Specific instructions to remember:
                    1. Do not use the exact content from the examples given in the prompt unless it really makes sense to.
                    2. The top level index key that comes back is because this is a result from a Pandas to_dict call with a split type - this is not a column name unless it also shows up in "columns" list.
                    3. Generate descriptions based on the data and column names provided, as well as the metadata.
                    4. Ensure the descriptions are detailed but concise, using between 50 to 200 words for comments.
                    5. Use all information provided for context, including catalog name, schema name, table name, column name, data, and metadata. Consider knowledge of outside sources, for example, if the table is clearly an SAP table, Salesforce table, or synthetic test table.
                    6. Please unpack any acronyms and abbreviations if you are confident in the interpretation.
                    7. Column contents will only represent a subset of data in the table so please provide information about the data in the sample but but be cautious inferring too strongly about the entire column based on the sample.
                    8. Only provide a dictionary. Any response other than the dictionary will be considered invalid. Make sure the list of column_names in the response content match the dictionary keys in the prompt input in column_contents.

                    ###
                    Please generate a description for the table and columns in the following string.

                    ###
                    Please provide the contents of a comment string in four sentences:
                    1) a brief factual description of the table or column and its purpose
                    2) any inference or deduction about the column based on the table name and column names
                    3) a description of the data in the column contents and any deductions that can be made about the column from that
                    4) a description of the metadata and further inference from the metadata itself. Do not rely too heavily on the data in the column contents, but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, and do not provide any offensive or dangerous content.

                    Please ONLY provide the dictionary response. The response will be considered invalid if it contains any other content other than the dictionary.
                    """
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "finance.restricted.customer_monthly_recurring_revenue", "column_contents": {"index": [0,1], "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$355.45", "2024-01-01", "True"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$4850.00", "2024-12-01", "False"]}, "column_metadata": {"name": {"col_name": "name", "data_type": "string", "num_nulls": "0", "distinct_count": "5", "avg_col_len": "16", "max_col_len": "23"}, "address": {"col_name": "address", "data_type": "string", "num_nulls": "0", "distinct_count": "46", "avg_col_len": "4", "max_col_len": "4"}, "email": {"col_name": "email", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "15", "max_col_len": "15"}, "revenue": {"col_name": "revenue", "data_type": "string", "num_nulls": "0", "distinct_count": "10", "avg_col_len": "11", "max_col_len": "11"}, "eap_created": {"col_name": "eap_created", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "11", "max_col_len": "11"}, "delete_flag": {"col_name": "delete_flag", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "11", "max_col_len": "11"}}}} and abbreviations and acronyms are here - {"EAP - enterprise architecture platform"}"""
                },
                {   "role": "assistant",
                    "content": """{"table": "Predictable recurring revenue earned from customers in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer's first and last name.", "Customer mailing address including both the number and street name, but not including the city, state, country, or zipcode. Stored as a string and populated in all cases. At least 46 distinct values.", "Customer email address with domain name. This is a common format for email addresses. Domains seen include MSN and AOL. These are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "Monthly recurring revenue from the customer in United States dollars with two decimals for cents. This field is never null, and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the Enterprise Architecture Platform or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column - based on the sample and the metadata it appears that every value is set to False, but as a string rather than as a boolean value."]}"""
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "hr.employees.employee_performance_reviews", "column_contents": {"index": [0,1,2,3,4], "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "data": [["E123", "2023-06-15", "4.5", "Excellent work throughout the year", "Yes"], ["E456", "2023-06-15", "3.2", "Needs improvement in meeting deadlines", "No"], ["E789", "2023-06-15", "4.8", "Outstanding performance and leadership", "Yes"], ["E101", "2023-06-15", "2.9", "Struggles with teamwork", "No"], ["E112", "2023-06-15", "3.7", "Consistently meets expectations", "Yes"]], "column_metadata": {"employee_id": {"col_name": "employee_id", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "4", "max_col_len": "4"}, "review_date": {"col_name": "review_date", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "10", "max_col_len": "10"}, "performance_score": {"col_name": "performance_score", "data_type": "string", "num_nulls": "0", "distinct_count": "50", "avg_col_len": "3", "max_col_len": "3"}, "manager_comments": {"col_name": "manager_comments", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "30", "max_col_len": "100"}, "promotion_recommendation": {"col_name": "promotion_recommendation", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "3", "max_col_len": "3"}}}} and abbreviations and acronyms are here - {"EID - employee ID"}"""
                },
                {
                    "role": "assistant",
                    "content": """{"table": "Employee performance reviews conducted annually. This table includes employee IDs, review dates, performance scores, manager comments, and promotion recommendations.", "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "column_contents": ["Unique identifier for each employee. This field is always populated and has 100 distinct values. The average and maximum column lengths are both 4, indicating a consistent format for employee IDs.", "Date when the performance review was conducted. This field is always populated and has only one distinct value in the sample, suggesting that all reviews were conducted on the same date. The average and maximum column lengths are both 10, consistent with the date format 'YYYY-MM-DD'.", "Performance score given by the manager, typically on a scale of 1 to 5. This field is always populated and has 50 distinct values. The average and maximum column lengths are both 3, indicating a consistent format for performance scores.", "Comments provided by the manager during the performance review. This field is always populated and has 100 distinct values, one for each employee, so these are fairly unique comments for each employee. The average column length is 30 and the maximum column length is 100, indicating a wide range of comment lengths, though given the skew there are probably a large number of very short comments.", "Recommendation for promotion based on the performance review. This field is always populated and has two distinct values: 'Yes' and 'No'. The average and maximum column lengths are both 3, indicating a consistent format for promotion recommendations."]}"""
                },
                {
                    "role": "user",
                    "content": f"""Content is here - {content} and abbreviations are here - {acro_content}"""
                }

              ]
            }



class PIPrompt(Prompt):
    def convert_to_comment_input(self) -> Dict[str, Any]:
        pandas_df = self.df.toPandas()
        if self.config.limit_prompt_based_on_cell_len:
            truncated_pandas_df = self.calculate_cell_length(pandas_df)
        else:
            truncated_pandas_df = pandas_df
        return {
            "table_name": self.full_table_name,
            "column_contents": truncated_pandas_df.to_dict(orient='split'),
        }

    def create_prompt_template(self) -> Dict[str, Any]:
        content = self.prompt_content
        acro_content = self.config.acro_content
        return {
            "pi": [
                {
                    "role": "system",
                    "content": """You are an AI assistant trying to help identify personally identifying information. Consider the data in the sample, the column names, and the column metadata, but please only respond with a dictionary in the format given in the user instructions. Do NOT use content from the examples given in the prompts. The examples are provided to help you understand the format of the prompt. Do not add a note after the dictionary, and do not provide any offensive or dangerous content.
                    ### 
                    Input Format
                    {"index": [0, 1], "columns": ["name", "address", "email", "MRR", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$1545.50", "2024-03-05", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$124555.32", "2023-01-03", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {{'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'MRR': {'col_name': 'MRR', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}, 'eap_created': {'col_name': 'eap_created', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}

                    ###
                    Please provide a response in this format, putting under classification either "pi" or "None", and under type either "pii", "pci", "medical_information", or "phi". Do not add additional fields.
                    ###

                    {"table": "pi", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": [{"classification": "pi", "type": "pii", "confidence": 0.85}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.98]}

                    ###

                    Specific Considerations
                    1. pi and pii are synonyms for our purposes, and not used as a legal term but as a way to distinguish individuals from one another.
                    2. Please don't respond with anything other than the dictionary.
                    3. Attempt to classify into None, PII, PCI, medical information, and PHI based on common definitions. If definitions are provided in the content then use them. Otherwise, assume that PII is Personally Identifiable Information, PHI is Protected Health Information, and PCI is Protected Confidential Information. PII: This includes any data that could potentially identify a specific individual. Medical Information: This includes any infromation in a medical record that cannot be used to identify an individual. PHI: This includes any information in a medical record that can be used to identify an individual and that was created, used, or disclosed in the course of providing a health care service such as diagnosis or treatment. PCI: Payment Card Industry Information, including the primary account number (PAN) typically found on the front of the card, the card's security code, full track data stored in the card's chip or magnetic stripe, cardholder PIN, cardholder name, or card expiration date.
                    4. When health or medical information is not linked to any identifiers, it's not considered PHI under HIPAA, it is considered medical infrormation. It only becomes PHI when it's combined with personally identifiable information in a way that could reasonably identify an individual. Thus, a column with a diagnosis is not PHI unless there are also patient names or other PII in the column as well, but a column or table that has both medical information and patient names is considered PHI.
                    5. The value for classification should always be either "pi" or "None". The value for type should always be either "pii", "pci", "medical_information", or "phi".
                    6. Any freeform medical information should be considered PHI, because it's not possible to guarantee it has no personal information. For example, "medical_notes", "chart_information", "doctor_notes".

                    ###
                    """
                    +
                    f"""PI Classification Rules: {self.config.pi_classification_rules}."""
                },
                {
                    "role": "user",
                    "content": """{"index": [0, 1], "columns": ["name", "address", "email", "credit_card", "medical_record", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "4111 1111 1111 1111", "MR12345", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "5500 0000 0000 0004", "MR67890", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'credit_card': {'col_name': 'credit_card', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '19', 'max_col_len': '19'}, 'medical_record': {'col_name': 'medical_record', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '7', 'max_col_len': '7'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}"""
                },
                {
                    "role": "assistant",
                    "content": """{"table": "pi", "columns": ["name", "address", "email", "credit_card", "medical_record", "delete_flag"], "column_contents": [{"classification": "pi", "type": "pii", "confidence": 0.95}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "pi", "type": "pci", "confidence": 0.95}, {"classification": "pi", "type": "phi", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.98}]}"""
                },
                {
                    "role": "user",
                    "content": """{"index": [0, 1], "columns": ["username", "ip_address", "session_id", "purchase_amount", "transaction_date", "is_active"], "data": [["user123", "192.168.1.1", "sess123", "$100.00", "2024-03-05", "True"], ["user456", "192.168.1.2", "sess456", "$200.00", "2024-03-06", "False"]], "column_metadata": {'username': {'col_name': 'username', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '7', 'max_col_len': '7'}, 'ip_address': {'col_name': 'ip_address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '11', 'max_col_len': '11'}, 'session_id': {'col_name': 'session_id', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '7', 'max_col_len': '7'}, 'purchase_amount': {'col_name': 'purchase_amount', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '7', 'max_col_len': '7'}, 'transaction_date': {'col_name': 'transaction_date', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '10', 'max_col_len': '10'}, 'is_active': {'col_name': 'is_active', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}"""
                },
                {
                    "role": "assistant",
                    "content": """{"table": "pi", "columns": ["username", "ip_address", "session_id", "purchase_amount", "transaction_date", "is_active"], "column_contents": [{"classification": "pi", "type": "pii", "confidence": 0.85}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.8}, {"classification": "None", "type": "None", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.85}, {"classification": "None", "type": "None", "confidence": 0.95}]}"""
                },
                {
                    "role": "user",
                    "content": """{"index": [0, 1], "columns": ["patient_name", "patient_id", "diagnosis", "treatment", "doctor_notes", "appointment_date"], "data": [["Jane Doe", "P12345", "Diabetes", "10 mg Insulin twice/day", "Ms. Doe is responding well to treatment", "2023-06-15"], ["John Smith", "P67890", "Hypertension", "Medication", "John's blood pressure is under control", "2023-06-16"]], "column_metadata": {'patient_name': {'col_name': 'patient_name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '8', 'max_col_len': '8'}, 'patient_id': {'col_name': 'patient_id', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '6', 'max_col_len': '6'}, 'diagnosis': {'col_name': 'diagnosis', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '8', 'max_col_len': '8'}, 'treatment': {'col_name': 'treatment', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '7', 'max_col_len': '7'}, 'doctor_notes': {'col_name': 'doctor_notes', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '30', 'max_col_len': '30'}, 'appointment_date': {'col_name': 'appointment_date', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '10', 'max_col_len': '10'}}}"""
                },
                {
                    "role": "assistant",
                    "content": """{"table": "phi", "columns": ["patient_name", "patient_id", "diagnosis", "treatment", "doctor_notes", "appointment_date"], "column_contents": [{"classification": "pi", "type": "pii", "confidence": 0.95}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "pi", "type": "medical_information", "confidence": 0.9}, {"classification": "pi", "type": "medical_information", "confidence": 0.7}, {"classification": "pi", "type": "phi", "confidence": 0.9}, {"classification": "pi", "type": "medical_information", "confidence": 0.7}]}"""
                },
                {
                    "role": "user",
                    "content": f"{content}"
                }
              ]
        }


class CommentNoDataPrompt(Prompt):
    def convert_to_comment_input(self) -> Dict[str, Any]:
        pandas_df = self.df.toPandas()
        if self.config.limit_prompt_based_on_cell_len:
            truncated_pandas_df = self.calculate_cell_length(pandas_df)
        else:
            truncated_pandas_df = pandas_df
        return {
            "table_name": self.full_table_name,
            "column_contents": truncated_pandas_df.to_dict(orient='split'),
        }

    def create_prompt_template(self) -> Dict[str, Any]:
        print("Creating comment prompt template with no data in comments...")
        content = self.prompt_content
        acro_content = self.config.acro_content
        return {
            "comment":
              [
                {
                    "role": "system",
                    "content": """You are an AI assistant helping to generate metadata for tables and columns in Databricks. The data you are working with may be sensitive so you don't want to include any data whatsoever in table or column descriptions.


                    ###
                    Input will come in in this format:

                    ###
                    Input format:
                    {"index": [0, 1], "columns": ["name", "address", "email", "MRR", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$1545.50", "2024-03-05", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$124555.32", "2023-01-03", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'MRR': {'col_name': 'MRR', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}, 'eap_created': {'col_name': 'eap_created', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}.

                    ###
                    Please provide a response in this format:
                    ###
                    {"table": "Predictable recurring revenue earned from subscriptions in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer name. Based on the data available in the sample, this field appears to contain both first and last name.", "Customer mailing address including both the number and street name, but at least in the sampled data not including the city, state, country, or zipcode. Stored as a string and populated in all cases. The sample has two distinct values for two rows, but the metadata makes it clear that there are 5 distinct values in the table.", "Customer email address with domain name. This is a common format for email addresses. Based on the domains they are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "MRR (Monthly recurring revenue) from the customer in United States dollars with two decimals for cents. This field is never null and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the EAP (Enterprise Architecture Platform) or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column, stored as a string rather than as a boolean value."]}

                    ###
                    Specific instructions to remember:
                    1. Do not use the exact content from the examples given in the prompt unless it really makes sense to.
                    2. The top level index key that comes back is because this is a result from a Pandas to_dict call with a split type - this is not a column name unless it also shows up in "columns" list.
                    3. Generate descriptions based on the data and column names provided, as well as the metadata.
                    4. Ensure the descriptions are detailed but concise, using between 50 to 200 words for comments.
                    5. Use all information provided for context, including catalog name, schema name, table name, column name, data, and metadata. Consider knowledge of outside sources, for example, if the table is clearly an SAP table, Salesforce table, or synthetic test table.
                    6. Please unpack any acronyms and abbreviations if you are confident in the interpretation.
                    7. Column contents will only represent a subset of data in the table so please provide aggregated information about the data in the sample but but be cautious inferring too strongly about the entire column based on the sample. Do not provide any actual data in the comment.
                    8. Only provide a dictionary. Any response other than the dictionary will be considered invalid. Make sure the list of column_names in the response content match the dictionary keys in the prompt input in column_contents.
                    9. Do not provide any actual data in the comment.

                    ###
                    Please generate a description for the table and columns in the following string.

                    ###
                    Please provide the contents of a comment string in four sentences:
                    1) a brief factual description of the table or column and its purpose
                    2) any inference or deduction about the column based on the table name and column names
                    3) a description of the data in the column contents and any deductions that can be made about the column from that
                    4) a description of the metadata and further inference from the metadata itself. Do not rely too heavily on the data in the column contents, but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, and do not provide any offensive or dangerous content.

                    Please ONLY provide the dictionary response. The response will be considered invalid if it contains any other content other than the dictionary.
                    """
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "finance.restricted.customer_monthly_recurring_revenue", "column_contents": {"index": [0,1], "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$355.45", "2024-01-01", "True"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$4850.00", "2024-12-01", "False"]}, "column_metadata": {"name": {"col_name": "name", "data_type": "string", "num_nulls": "0", "distinct_count": "5", "avg_col_len": "16", "max_col_len": "23"}, "address": {"col_name": "address", "data_type": "string", "num_nulls": "0", "distinct_count": "46", "avg_col_len": "4", "max_col_len": "4"}, "email": {"col_name": "email", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "15", "max_col_len": "15"}, "revenue": {"col_name": "revenue", "data_type": "string", "num_nulls": "0", "distinct_count": "10", "avg_col_len": "11", "max_col_len": "11"}, "eap_created": {"col_name": "eap_created", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "11", "max_col_len": "11"}, "delete_flag": {"col_name": "delete_flag", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "11", "max_col_len": "11"}}}} and abbreviations and acronyms are here - {"EAP - enterprise architecture platform"}"""
                },
                {   "role": "assistant",
                    "content": """{"table": "Predictable recurring revenue earned from customers in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer's first and last name.", "Customer mailing address including both the number and street name, but not including the city, state, country, or zipcode. Stored as a string and populated in all cases. At least 46 distinct values.", "Customer email address with domain name. This is a common format for email addresses. These are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "Monthly recurring revenue from the customer in United States dollars with two decimals for cents. This field is never null, and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the Enterprise Architecture Platform or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column - based on the sample and the metadata it appears that every value is set to the same value, but as a string rather than as a boolean value."]}"""
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "hr.employees.employee_performance_reviews", "column_contents": {"index": [0,1,2,3,4], "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "data": [["E123", "2023-06-15", "4.5", "Excellent work throughout the year", "Yes"], ["E456", "2023-06-15", "3.2", "Needs improvement in meeting deadlines", "No"], ["E789", "2023-06-15", "4.8", "Outstanding performance and leadership", "Yes"], ["E101", "2023-06-15", "2.9", "Struggles with teamwork", "No"], ["E112", "2023-06-15", "3.7", "Consistently meets expectations", "Yes"]], "column_metadata": {"employee_id": {"col_name": "employee_id", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "4", "max_col_len": "4"}, "review_date": {"col_name": "review_date", "data_type": "string", "num_nulls": "0", "distinct_count": "1", "avg_col_len": "10", "max_col_len": "10"}, "performance_score": {"col_name": "performance_score", "data_type": "string", "num_nulls": "0", "distinct_count": "50", "avg_col_len": "3", "max_col_len": "3"}, "manager_comments": {"col_name": "manager_comments", "data_type": "string", "num_nulls": "0", "distinct_count": "100", "avg_col_len": "30", "max_col_len": "100"}, "promotion_recommendation": {"col_name": "promotion_recommendation", "data_type": "string", "num_nulls": "0", "distinct_count": "2", "avg_col_len": "3", "max_col_len": "3"}}}} and abbreviations and acronyms are here - {"EID - employee ID"}"""
                },
                {
                    "role": "assistant",
                    "content": """{"table": "Employee performance reviews conducted annually. This table includes employee IDs, review dates, performance scores, manager comments, and promotion recommendations.", "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "column_contents": ["Unique identifier for each employee. This field is always populated and has 100 distinct values. The average and maximum column lengths are both 4, indicating a consistent format for employee IDs.", "Date when the performance review was conducted. This field is always populated and has only one distinct value in the sample, suggesting that all reviews were conducted on the same date. The average and maximum column lengths are both 10, consistent with the date format 'YYYY-MM-DD'.", "Performance score given by the manager, representing single digit integers. This field is always populated and has 50 distinct values. The average and maximum column lengths are both 3, indicating a consistent format for performance scores.", "Comments provided by the manager during the performance review. This field is always populated and has 100 distinct values, one for each employee, so these are fairly unique comments for each employee. The average column length is 30 and the maximum column length is 100, indicating a wide range of comment lengths, though given the skew there are probably a large number of very short comments.", "Recommendation for promotion based on the performance review. This field is always populated and has two distinct values. The average and maximum column lengths are both 3, indicating a consistent format for promotion recommendations."]}"""
                },
                {
                    "role": "user",
                    "content": f"""Content is here - {content} and abbreviations are here - {acro_content}"""
                }

              ]
            }



class PromptFactory:
    @staticmethod
    def create_prompt(config, df, full_table_name) -> Prompt:
        if config.mode == "comment" and config.allow_data:
            return CommentPrompt(config, df, full_table_name)
        elif config.mode == "comment":
            return CommentNoDataPrompt(config, df, full_table_name)
        elif config.mode == "pi":
            return PIPrompt(config, df, full_table_name)
        else:
            raise ValueError("Invalid mode. Use 'pi' or 'comment'.")
