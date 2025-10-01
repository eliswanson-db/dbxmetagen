"""
DSPy Prompts based on existing CommentPrompt and CommentNoDataPrompt templates.

This module extracts the existing prompt templates and converts them to DSPy format
without modifying the original prompts.py file. This allows for independent optimization
while maintaining the original system's functionality.
"""

from typing import Dict, Any

# Try to import optional dependencies
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None


class DSPyPromptExtractor:
    """Extracts existing prompts and converts them to DSPy format."""

    @staticmethod
    def extract_comment_prompt_instructions() -> str:
        """Extract system instructions from CommentPrompt."""
        # Base system message from CommentPrompt
        system_content = """You are an AI assistant helping to generate metadata for tables and columns in Databricks. You are very careful to properly identify PII, PCI, and PHI, and you care deeply about ensuring high quality responses.

Input will come in this format:
{"index": [0, 1], "columns": ["name", "address", "email", "MRR", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$1545.50", "2024-03-05", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$124555.32", "2023-01-03", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'MRR': {'col_name': 'MRR', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}, 'eap_created': {'col_name': 'eap_created', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}.

Please provide a response in this format:
{"table": "Predictable recurring revenue earned from subscriptions in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer name. Based on the data available in the sample, this field appears to contain both first and last name.", "Customer mailing address including both the number and street name, but at least in the sampled data not including the city, state, country, or zipcode. Stored as a string and populated in all cases. The sample has two distinct values for two rows, but the metadata makes it clear that there are 5 distinct values in the table.", "Customer email address with domain name. This is a common format for email addresses. Domains seen include MSN and AOL. These are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "MRR (Monthly recurring revenue) from the customer in United States dollars with two decimals for cents. This field is never null and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the EAP (Enterprise Architecture Platform) or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column - based on the sample and the metadata it appears that every value is set to False, but based on the column metadata as a string rather than as a boolean value."]}

Specific instructions to remember:
1. Do not use the exact content from the examples given in the prompt unless it really makes sense to.
2. The top level index key that comes back is because this is a result from a Pandas to_dict call with a split type - this is not a column name unless it also shows up in "columns" list.
3. Generate descriptions based on the data and column names provided, as well as the metadata.
4. Ensure the descriptions are detailed but concise, using between 50 to 200 words for comments.
5. Use all information provided for context, including catalog name, schema name, table name, column name, data, and metadata. Consider knowledge of outside sources, for example, if the table is clearly an SAP table, Salesforce table, or synthetic test table.
6. Please unpack any acronyms and abbreviations if you are confident in the interpretation.
7. Column contents will only represent a subset of data in the table so please provide information about the data in the sample but but be cautious inferring too strongly about the entire column based on the sample.
8. Only provide a dictionary. Any response other than the dictionary will be considered invalid. Make sure the list of column_names in the response content match the dictionary keys in the prompt input in column_contents.
9. Consider that any DDL generated could be run directly in SQL or in Python, and make sure that it's directly runnable without error. Outer strings should be double quoted. Within strings, use single quotes as would be needed if the comment or column_content would be used as a Python string or in SQL DDL. For example format responses like: "The column 'scope' is a summary column.", rather than "The column "scope" is a summary column."
10. Use double quotes to enclose all comments. Apostrophe's need to be escaped.

Please generate a description for the table and columns in the following string.

Please provide the contents of a comment string in four sentences:
1) a brief factual description of the table or column and its purpose
2) any inference or deduction about the column based on the table name and column names
3) a description of the data in the column contents and any deductions that can be made about the column from that
4) a description of the metadata and further inference from the metadata itself. Do not rely too heavily on the data in the column contents, but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, and do not provide any offensive or dangerous content.

Please ONLY provide the dictionary response. The response will be considered invalid if it contains any other content other than the dictionary."""

        return system_content

    @staticmethod
    def extract_comment_no_data_prompt_instructions() -> str:
        """Extract system instructions from CommentNoDataPrompt."""
        system_content = """You are an AI assistant helping to generate metadata for tables and columns in Databricks. The data you are working with may be sensitive so you don't want to include any data whatsoever in table or column descriptions. It doesn't matter if data are coming from data pulls or from stored metadata, do not include any data in your outputs.

Input will come in this format:
{"index": [0, 1], "columns": ["name", "address", "email", "MRR", "eap_created", "delete_flag"], "data": [["John Johnson", "123 Main St", "jj@msn.com", "$1545.50", "2024-03-05", "False"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "$124555.32", "2023-01-03", "False"]], "column_metadata": {'name': {'col_name': 'name', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '5', 'avg_col_len': '16', 'max_col_len': '23'}, 'address': {'col_name': 'address', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '46', 'avg_col_len': '4', 'max_col_len': '4'}, 'email': {'col_name': 'email', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '15', 'max_col_len': '15'}, 'MRR': {'col_name': 'MRR', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '1', 'avg_col_len': '11', 'max_col_len': '11'}, 'eap_created': {'col_name': 'eap_created', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '3', 'avg_col_len': '12', 'max_col_len': '12'}, 'delete_flag': {'col_name': 'delete_flag', 'data_type': 'string', 'num_nulls': '0', 'distinct_count': '2', 'avg_col_len': '5', 'max_col_len': '5'}}}.

Please provide a response in this format:
{"table": "Predictable recurring revenue earned from subscriptions in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer name. Based on the data available in the sample, this field appears to contain both first and last name.", "Customer mailing address including both the number and street name, but at least in the sampled data not including the city, state, country, or zipcode. Stored as a string and populated in all cases. The sample has two distinct values for two rows, but the metadata makes it clear that there are 5 distinct values in the table.", "Customer email address with domain name. This is a common format for email addresses. Based on the domains they are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "MRR (Monthly recurring revenue) from the customer in United States dollars with two decimals for cents. This field is never null and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the EAP (Enterprise Architecture Platform) or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column, stored as a string rather than as a boolean value."]}

Specific instructions to remember:
1. Do not use the exact content from the examples given in the prompt unless it really makes sense to.
2. The top level index key that comes back is because this is a result from a Pandas to_dict call with a split type - this is not a column name unless it also shows up in "columns" list.
3. Generate descriptions based on the data and column names provided, as well as the metadata.
4. Ensure the descriptions are detailed but concise, using between 50 to 200 words for comments.
5. Use all information provided for context, including catalog name, schema name, table name, column name, data, and metadata. Consider knowledge of outside sources, for example, if the table is clearly an SAP table, Salesforce table, or synthetic test table.
6. Please unpack any acronyms and abbreviations if you are confident in the interpretation.
7. Column contents will only represent a subset of data in the table so please provide aggregated information about the data in the sample but but be cautious inferring too strongly about the entire column based on the sample. Do not provide any actual data in the comment.
8. Only provide a dictionary. Any response other than the dictionary will be considered invalid. Make sure the list of column_names in the response content match the dictionary keys in the prompt input in column_contents.
9. Within strings, use single quotes as would be needed if the comment or column_content would be used as a Python string or in SQL DDL. For example format responses like: "The column 'scope' is a summary column.", rather than "The column "scope" is a summary column."
10. Do not provide any actual data in the comment.
11. Consider that any DDL generated could be run directly in SQL or in Python, and make sure that it's directly runnable without error. Outer strings should be double quoted. Within strings, use single quotes as would be needed if the comment or column_content would be used as a Python string or in SQL DDL. For example format responses like: "The column 'scope' is a summary column.", rather than "The column "scope" is a summary column."
12. Try not to generate apostrophes. When you must use them, determine if the generated text is DDL or not. If it is DDL or a comment that might be inserted into DDL, escape it like: '', with two single quotes in a row, for a SQL escape. If it is used in another context, you can escape it using a back slash as a standard Python escape for an apostrophe.

Please generate a description for the table and columns in the following string.

Please provide the contents of a comment string in four sentences:
1) a brief factual description of the table or column and its purpose
2) any inference or deduction about the column based on the table name and column names
3) a description of the data in the column contents and any deductions that can be made about the column from that
4) a description of the metadata and further inference from the metadata itself. Do not rely too heavily on the data in the column contents, but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, and do not provide any offensive or dangerous content.

Please ONLY provide the dictionary response. The response will be considered invalid if it contains any other content other than the dictionary."""

        return system_content

    @staticmethod
    def get_comment_examples() -> list:
        """Extract few-shot examples from CommentPrompt."""
        return [
            {
                "input": {
                    "table_name": "finance.restricted.customer_monthly_recurring_revenue",
                    "table_data": {
                        "index": [0, 1],
                        "columns": [
                            "name",
                            "address",
                            "email",
                            "revenue",
                            "eap_created",
                            "delete_flag",
                        ],
                        "data": [
                            [
                                "John Johnson",
                                "123 Main St",
                                "jj@msn.com",
                                "$355.45",
                                "2024-01-01",
                                "True",
                            ],
                            [
                                "Alice Ericks",
                                "6789 Fake Ave",
                                "alice.ericks@aol.com",
                                "$4850.00",
                                "2024-12-01",
                                "False",
                            ],
                        ],
                    },
                    "column_metadata": {
                        "name": {
                            "col_name": "name",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "5",
                            "avg_col_len": "16",
                            "max_col_len": "23",
                        },
                        "address": {
                            "col_name": "address",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "46",
                            "avg_col_len": "4",
                            "max_col_len": "4",
                        },
                        "email": {
                            "col_name": "email",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "2",
                            "avg_col_len": "15",
                            "max_col_len": "15",
                        },
                        "revenue": {
                            "col_name": "revenue",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "10",
                            "avg_col_len": "11",
                            "max_col_len": "11",
                        },
                        "eap_created": {
                            "col_name": "eap_created",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "1",
                            "avg_col_len": "11",
                            "max_col_len": "11",
                        },
                        "delete_flag": {
                            "col_name": "delete_flag",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "1",
                            "avg_col_len": "11",
                            "max_col_len": "11",
                        },
                    },
                    "abbreviations": {"EAP": "enterprise architecture platform"},
                },
                "output": '{"table": "Predictable recurring revenue earned from customers in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "columns": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer\'s first and last name.", "Customer mailing address including both the number and street name, but not including the city, state, country, or zipcode. Stored as a string and populated in all cases. At least 46 distinct values.", "Customer email address with domain name. This is a common format for email addresses. These are not likely domains for company email addresses. Email field is always populated, although there appears to be very few distinct values in the table.", "Monthly recurring revenue from the customer in United States dollars with two decimals for cents. This field is never null, and only has 10 distinct values, odd for an MRR field.", "Date when the record was created in the Enterprise Architecture Platform or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system. Every value appears to be the same in this column - based on the sample and the metadata it appears that every value is set to the same value, but as a string rather than as a boolean value."]}',
            },
            {
                "input": {
                    "table_name": "hr.employees.employee_performance_reviews",
                    "table_data": {
                        "index": [0, 1, 2, 3, 4],
                        "columns": [
                            "employee_id",
                            "review_date",
                            "performance_score",
                            "manager_comments",
                            "promotion_recommendation",
                        ],
                        "data": [
                            [
                                "E123",
                                "2023-06-15",
                                "4.5",
                                "Excellent work throughout the year",
                                "Yes",
                            ],
                            [
                                "E456",
                                "2023-06-15",
                                "3.2",
                                "Needs improvement in meeting deadlines",
                                "No",
                            ],
                            [
                                "E789",
                                "2023-06-15",
                                "4.8",
                                "Outstanding performance and leadership",
                                "Yes",
                            ],
                            [
                                "E101",
                                "2023-06-15",
                                "2.9",
                                "Struggles with teamwork",
                                "No",
                            ],
                            [
                                "E112",
                                "2023-06-15",
                                "3.7",
                                "Consistently meets expectations",
                                "Yes",
                            ],
                        ],
                    },
                    "column_metadata": {
                        "employee_id": {
                            "col_name": "employee_id",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "100",
                            "avg_col_len": "4",
                            "max_col_len": "4",
                        },
                        "review_date": {
                            "col_name": "review_date",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "1",
                            "avg_col_len": "10",
                            "max_col_len": "10",
                        },
                        "performance_score": {
                            "col_name": "performance_score",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "50",
                            "avg_col_len": "3",
                            "max_col_len": "3",
                        },
                        "manager_comments": {
                            "col_name": "manager_comments",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "100",
                            "avg_col_len": "30",
                            "max_col_len": "100",
                        },
                        "promotion_recommendation": {
                            "col_name": "promotion_recommendation",
                            "data_type": "string",
                            "num_nulls": "0",
                            "distinct_count": "2",
                            "avg_col_len": "3",
                            "max_col_len": "3",
                        },
                    },
                    "abbreviations": {"EID": "employee ID"},
                },
                "output": '{"table": "Employee performance reviews conducted annually. This table includes employee IDs, review dates, performance scores, manager comments, and promotion recommendations.", "columns": ["employee_id", "review_date", "performance_score", "manager_comments", "promotion_recommendation"], "column_contents": ["Unique identifier for each employee. This field is always populated and has 100 distinct values. The average and maximum column lengths are both 4, indicating a consistent format for employee IDs.", "Date when the performance review was conducted. This field is always populated and has only one distinct value in the sample, suggesting that all reviews were conducted on the same date. The average and maximum column lengths are both 10, consistent with the date format \'YYYY-MM-DD\'.", "Performance score given by the manager, representing single digit integers. This field is always populated and has 50 distinct values. The average and maximum column lengths are both 3, indicating a consistent format for performance scores.", "Comments provided by the manager during the performance review. This field is always populated and has 100 distinct values, one for each employee, so these are fairly unique comments for each employee. The average column length is 30 and the maximum column length is 100, indicating a wide range of comment lengths, though given the skew there are probably a large number of very short comments.", "Recommendation for promotion based on the performance review. This field is always populated and has two distinct values. The average and maximum column lengths are both 3, indicating a consistent format for promotion recommendations."]}',
            },
        ]


class DSPyPromptInitializer:
    """Initializes DSPy with prompts based on existing templates."""

    def __init__(self, allow_data_in_comments=True):
        self.allow_data_in_comments = allow_data_in_comments
        self.extractor = DSPyPromptExtractor()

    def initialize_dspy_with_existing_prompts(self, lm_config=None):
        """Initialize DSPy with prompts extracted from existing system."""
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy not available. Install with: pip install dspy-ai>=2.4.0"
            )

        # Configure language model
        if lm_config:
            dspy.settings.configure(lm=lm_config)
        else:
            # Default OpenAI configuration
            dspy.settings.configure(lm=dspy.OpenAI())

        # Get instructions based on data policy
        if self.allow_data_in_comments:
            instructions = self.extractor.extract_comment_prompt_instructions()
        else:
            instructions = self.extractor.extract_comment_no_data_prompt_instructions()

        # Get few-shot examples
        examples = self.extractor.get_comment_examples()

        return instructions, examples

    def create_optimized_signature(self, instructions):
        """Create a DSPy signature with extracted instructions."""
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available")

        class OptimizedCommentSignature(dspy.Signature):
            """Generate table and column comments for database metadata using optimized prompts."""

            table_name: str = dspy.InputField(
                desc="Full table name (catalog.schema.table)"
            )
            table_data: str = dspy.InputField(
                desc="JSON representation of table data with columns and sample rows"
            )
            column_metadata: str = dspy.InputField(
                desc="JSON metadata for columns including data types, null counts, etc."
            )
            abbreviations: str = dspy.InputField(
                desc="JSON containing abbreviations and their expansions"
            )

            metadata_response: str = dspy.OutputField(
                desc=instructions[:500]
                + "... Return JSON dictionary with 'table', 'columns', and 'column_contents' fields."
            )

        return OptimizedCommentSignature


def initialize_dspy_from_existing_prompts(config, allow_data_in_comments=True):
    """
    Convenience function to initialize DSPy using existing prompt templates.

    Args:
        config: MetadataConfig object with model configuration
        allow_data_in_comments: Whether to allow actual data in comments

    Returns:
        Tuple of (instructions, examples, signature)
    """
    if not DSPY_AVAILABLE:
        raise ImportError(
            "DSPy not available. Install with: pip install dspy-ai>=2.4.0"
        )

    initializer = DSPyPromptInitializer(allow_data_in_comments)

    # Configure DSPy with model from config
    if hasattr(config, "model"):
        lm = dspy.OpenAI(
            model=config.model,
            max_tokens=getattr(config, "max_tokens", 4000),
            temperature=getattr(config, "temperature", 0.7),
        )
    else:
        lm = dspy.OpenAI()

    instructions, examples = initializer.initialize_dspy_with_existing_prompts(lm)
    signature = initializer.create_optimized_signature(instructions)

    return instructions, examples, signature
