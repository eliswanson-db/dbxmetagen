from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from pyspark.sql import SparkSession


class Prompt(ABC):
    def __init__(self, config, df, full_table_name):
        self.config = config
        self.df = df
        self.full_table_name = full_table_name
        self.prompt_content = self.convert_to_comment_input()
        if self.config.add_metadata:
            self.add_metadata_to_comment_input()
        print("Instantiating chat completion response...")

    @abstractmethod
    def convert_to_comment_input(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def add_metadata_to_comment_input(self) -> None:
        pass




class CommentPrompt(Prompt):
    def convert_to_comment_input(self) -> Dict[str, Any]:
        return {
            "table_name": self.full_table_name,
            "column_contents": self.df.toPandas().to_dict(orient='split'),
        }

    def add_metadata_to_comment_input(self) -> None:
        spark = SparkSession.builder.getOrCreate()
        column_metadata_dict = {}
        for column_name in self.prompt_content['column_contents']['columns']:
            extended_metadata_df = spark.sql(
                f"DESCRIBE EXTENDED {self.full_table_name} {column_name}"
            )            
            filtered_metadata_df = extended_metadata_df.filter(
                (extended_metadata_df["info_value"] != "NULL") &
                (extended_metadata_df["info_name"] != "description") &
                (extended_metadata_df["info_name"] != "comment")
            )
            column_metadata = filtered_metadata_df.toPandas().to_dict(orient='list')
            combined_metadata = dict(zip(column_metadata['info_name'], column_metadata['info_value']))
            column_metadata_dict[column_name] = combined_metadata
        
        self.prompt_content['column_contents']['column_metadata'] = column_metadata_dict

    def create_prompt_template(self) -> Dict[str, Any]:
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
        return {
            "table_name": self.full_table_name,
            "column_contents": self.df.toPandas().to_dict(orient='split'),
        }

    def add_metadata_to_comment_input(self) -> None:
        spark = SparkSession.builder.getOrCreate()
        column_metadata_dict = {}
        for column_name in self.prompt_content['column_contents']['columns']:
            extended_metadata_df = spark.sql(
                f"DESCRIBE EXTENDED {self.full_table_name} {column_name}"
            )            
            filtered_metadata_df = extended_metadata_df.filter(
                (extended_metadata_df["info_value"] != "NULL")
            )
            column_metadata = filtered_metadata_df.toPandas().to_dict(orient='list')
            combined_metadata = dict(zip(column_metadata['info_name'], column_metadata['info_value']))
            column_metadata_dict[column_name] = combined_metadata
            
        self.prompt_content['column_contents']['column_metadata'] = column_metadata_dict

    def create_prompt_template(self) -> Dict[str, Any]:
        content = self.prompt_content
        acro_content = self.config.acro_content
        return {
            "pi": [
                {
                    "role": "system",
                    "content": """You are an AI assistant trying to help identify personally identifying information. Consider the data in the sample, but please only respond with a dictionary in the format given in the user instructions. Do NOT use content from the examples given in the prompts. The examples are provided to help you understand the format of the prompt. Do not add a note after the dictionary, and do not provide any offensive or dangerous content.
                    
                    ### 
                    Input Format
                    {"index": [0, 1], "columns": ["name", "address", "email", "MRR", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$1545.50", "2024-03-05", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$124555.32", "2023-01-03", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {{'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'MRR': {'col_name': 'MRR', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}, 'eap_created': {'col_name': 'eap_created', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}

                    ###
                    Please provide a response in this format:
                    ###

                    {"table": "pi", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": [{"classification": "pi", "type": "pii", "confidence": 0.85}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "pi", "type": "pii", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.9}, {"classification": "None", "type": "None", "confidence": 0.98]}

                    ### 
                    
                    Specific Considerations
                    1. pi and pii are synonyms for our purposes, and not used as a legal term but as a way to distinguish individuals from one another.
                    2. Please don't respond with anything other than the dictionary.
                    3. Attempt to classify into PI, PII, PCI, and PHI based on common definitions. If definitions are provided in the content then use them.

                    ### 
                    
                    PI Classification Rules: {pi_classification_rules}

                    """ 
                },
                {
                    "role": "user",
                    "content": f"""Please look at each column in {content} and identify if the content represents a person, an address, an email, or a potentially valid national ID for a real country and provide a probability that it represents personally identifying information - confidence - scaled from 0 to 1. In addition, provide a classification for PCI or PHI if there is some probability that these are true. 
                    
                    Content will be provided as a Python dictionary in a string, formatted like this example, but the table name and column names might vary: {{"table_name": "finance.restricted.monthly_recurring_revenue", "column_contents": {{"index": [0, 1], "columns": ["name", "address", "email", "ssn", "religion"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "664-35-1234", "Buddhist"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "664-35-1234", "Episcopalian"]], "column_metadata": {{'name': {{'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}}, 'address': {{'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}}, 'email': {{'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}}, 'ssn': {{'col_name': 'ssn', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}}, 'religion': {{'col_name': 'religion', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}}}}}}}}. Please provide the response as a dictionary in a string. Options for classification are 'phi', 'none', or 'pci'. phi is health information that is tied to pi.

                    Example: if the content is the example above, then the response should look like {{"table": "pi", "columns": ["name", "address", "email", "ssn", "religion"], "column_contents": [{{"classification": "pi", "pi_type": "pii", "confidence": 0.8}}, {{"classification": "pi", "pi_type": "pii", "confidence": 0.7}}, {{"classification": "pi", "pi_type": "pii", "confidence": 0.9}}, {{"classification": "pi", "pi_type": "pii", "confidence": 0.5}}, {{"classification": "none", "pi_type": "none", "confidence": 0.95}}]}}. 
                    
                    Please don't respond with any other content other than the dictionary.
                    """
                }
              ]
        }


class CommentNoDataPrompt(Prompt):
    def convert_to_comment_input(self) -> Dict[str, Any]:
        return {
            "table_name": self.full_table_name,
            "column_contents": self.df.toPandas().to_dict(orient='split'),
        }

    def add_metadata_to_comment_input(self) -> None:
        spark = SparkSession.builder.getOrCreate()
        column_metadata_dict = {}
        for column_name in self.prompt_content['column_contents']['columns']:
            extended_metadata_df = spark.sql(
                f"DESCRIBE EXTENDED {self.full_table_name} {column_name}"
            )            
            filtered_metadata_df = extended_metadata_df.filter(
                (extended_metadata_df["info_value"] != "NULL") &
                (extended_metadata_df["info_name"] != "description") &
                (extended_metadata_df["info_name"] != "comment")
            )
            column_metadata = filtered_metadata_df.toPandas().to_dict(orient='list')
            combined_metadata = dict(zip(column_metadata['info_name'], column_metadata['info_value']))
            column_metadata_dict[column_name] = combined_metadata
            
        self.prompt_content['column_contents']['column_metadata'] = column_metadata_dict

    def create_prompt_template(self) -> Dict[str, Any]:
        content = self.prompt_content
        acro_content = self.config.acro_content
        return {
            "comment_no_data":
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
        if config.mode == "comment":
            return CommentPrompt(config, df, full_table_name)
        elif config.mode == "comment" and config.allow_data:
            return CommentNoDataPrompt(config, df, full_table_name)
        elif config.mode == "pi":
            return PIPrompt(config, df, full_table_name)
        else:
            raise ValueError("Invalid mode. Use 'pi' or 'comment'.")

