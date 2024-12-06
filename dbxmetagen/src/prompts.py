def create_prompt_template(content, acro_content):
    return {"pi":
              [
                {
                    "role": "system",
                    "content": f"You are an AI assistant trying to help identify personally identifying information. Please only respond with a dictionary in the format given in the user instructions. Do NOT use content from the examples given in the prompts. The examples are provided to help you understand the format of the prompt. Do not add a note after the dictionary, and do not provide any offensive or dangerous content." 
                },
                {
                    "role": "user",
                    "content": f"""Please look at each column in {content} and identify if the content represents a person, an address, an email, or a potentially valid national ID for a real country and provide a probability that it represents personally identifying information - confidence - scaled from 0 to 1. In addition, provide a classification for PCI or PHI if there is some probability that these are true. 
                    
                    Content will be provided as a Python dictionary in a string, formatted like this example, but the table name and column names might vary: {{"table_name": "finance.restricted.monthly_recurring_revenue", "column_names": ["name", "address", "email", "ssn", "religion"], "column_contents": [["John Johnson", "123 Main St", "jj@msn.com", "664-35-1234", "Buddhist"], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "664-35-1234", "Episcopalian"]]}}. Please provide the response as a dictionary in a string. Options for classification are 'phi', 'none', or 'pci'. phi is health information that is tied to pi.

                    pi and pii are synonyms for our purposes, and not used as a legal term but as a way to distinguish individuals from one another. Example: if the content is the example above, then the response should look like {{"table": "pi", "column_names": ["name", "address", "email", "ssn", "religion"], "column_contents": [{{"classification": "pi", "type": "name", "confidence": 0.8}}, {{"classification": "pi", "type": "address", "confidence": 0.7}}, {{"classification": "pi", "type": "email", "confidence": 0.9}}, {{"classification": "pi", "type": "national ID", "confidence": 0.5}}, {{"classification": "none", "type": "none", "confidence": 0.95}}]}}. Modifier: if the content of a column is like {{"appointment_text": "John Johnson has high blood pressure"}}, then the classification should be "phi" and if a column appears to be a credit card number or a bank account number then it should be labeled as "pci". 'type' values allowed include 'name', 'location', 'national ID', 'email', and 'phone'. If the confidence is less than 0.5, then the classification should be 'none'. 
                    
                    Please don't respond with any other content other than the dictionary.
                    """
                }
              ],
              "comment":
              [
                {
                    "role": "system",
                    "content": """You are an AI assistant helping to generate metadata for tables and columns in Databricks. Please respond only with a dictionary in the specified format - {{"table": "finance.restricted.customer_monthly_recurring_revenue", "column_names": ["name", "address", "email", "revenue"], "column_contents": [["John Johnson", "123 Main St", "jj@msn.com", "2024-01-05", NULL], ["Alice Ericks", "6789 Fake Ave", "alice.ericks@aol.com", "2024-01-05", NULL]}}. Do not use the exact content from the examples given in the prompt unless it really makes sense to. Generate descriptions based on the data and column names provided, as well as the metadata. Ensure the descriptions are detailed but concise, using between 50 to 200 for comments. If column names are informative, prioritize them; otherwise, rely more on the data content. Use the metadata as additional context. Please unpack any acronyms, initialisms, and abbreviations unless they are in common parlance like SCUBA. Column contents will only represent a subset of data in the table so please provide information about the data in the sample but but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, any response other than the dictionary will be considered invalid. Make sure the list of column_names in the response content match the dictionary keys in the prompt input in column_contents. Be aware that the column_contents will include only a sample of the data, not the entire content of the data.
                    
                    Please generate a description for the table and columns in the following string. The content string will be provided as a Python dictionary in a string, formatted like in the examples, but the table name and column names might vary. Please generate table descriptions between 50 and 200 words considering the catalog, schema, table, as well as column names and content. Please generate column descriptions between 10 and 50 words. Please provide the contents of a comment string in four sentences: 1) a brief factual description of the table or column and its purpose, 2) any inference or deduction about the column based on the table name and column names, 3) a description of the data in the column contents and any deductions that can be made about the column from that, and 4) a description of the metadata and further inference from the metadata itself. Do not rely too heavily on the data in the column contents, but be cautious inferring too strongly about the entire column based on the sample. Do not add a note after the dictionary, and do not provide any offensive or dangerous content. 
                    
                    Please NEVER add notes or other info to the response. The response will be considered invalid if it contains any other content other than the dictionary.
                    """
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "finance.restricted.customer_monthly_recurring_revenue", "column_contents": [{"name": ["John Johnson", "Alice Ericks"]}, {"address": ["123 Main St", "6789 Fake Ave"]}, {"email": ["jj@msn.com", "alice.ericks@aol.com"]}, {"revenue": ["$355.45", "$4850.00"]}, {"eap_created": ["2024-01-01", "2024-12-01"]}, {"delete_flag": ["True", NULL]}} and abbreviations and acronyms are here - {"EAP - enterprise architecture platform"}"""
                },
                {   "role": "assistant",
                    "content": """{"table": "Predictable recurring revenue earned from subscriptions in a specific period. Monthly recurring revenue, or MRR, is calculated on a monthly duration and in this case aggregated at a customer level. This table includes customer names, addresses, emails, and other identifying information as well as system colums.", "column_names": ["name", "address", "email", "revenue", "eap_created", "delete_flag"], "column_contents": ["Customer's first and last name.", "Customer mailing address including both the number and street name, but not including the city, state, country, or zipcode.", "Customer email address with domain name. This is a common format for email addresses. Domains seen include MSN and AOL. These are not likely domains for company email addresses.", "Monthly recurring revenue from the customer in United States dollars with two decimals for cents.", "Date when the record was created in the Enterprise Architecture Platform or by the Enterprise Architecture Platform team.", "Flag indicating whether the record has been deleted from the system. Most likely this is a soft delete flag, indicating a hard delete in an upstream system."]}""" 
                },
                {
                    "role": "user",
                    "content": """Content is here - {"table_name": "enterprise.master_data.customer_master", "column_contents": [{"name": ["John Johnson", "Alice Ericks", "Charlie J. Berens", None]}, {"address": ["123 Main St", "6789 Fake Ave", "42909 Johnsone Street, Dallas, Texas 44411-1111]}, {"cid": ["429184984", "443345555"]}, {"PN": ["(214) 555-0100", "(214) 555-0101"]}, {"email": ["jj@msn.com", "alice.ericks@aol.com"]}, {"dob": ["1980-01-01", "1980-01-02"]}} and abbreviations and acronyms are here - {"EAP: enterprise architecture platform, BU: business unit, PN: phone number, dob: date of birth"}"""
                },
                {   "role": "assistant",
                    "content": """{"table": "Master data for customers. Customer master data is a non-transactional information that identifies and describes a customer within a business's database. Contains customer names, addresses, phone numbers, email addresses, and date of birth. This table appears to be customer master data used enterprise-wide. There appears to be some risk of personally identifying information appearing in this table.", "column_names": ["name", "address", "cid", "PN", "email", "dob"], "column_contents": ["Customer's first and last name. In some cases, middle initial is included so it's possible that this is a free form entry field or that a variety of options are available for name.", "Customer mailing address including both the number and street name, and in some cases, but not all the city, state, and zipcode. It's possible this is free entry or taken from a variety of sources. The one zipcode in the sample data apppears to be the zipcode +4.", "Customer's unique identifier. This is a 9-digit number that appears to be a customer identifier. This is a column that appears to be a customer identifier. There is a small risk that these could be social security numbers in the United States despite not being labeled as such, as the number of digits match and they're in a customer table.", "Customer's phone number, including the area code and formatted with puncuation.", "Customer's email address with domain name.", "Customer's date of birth in the form of a string, but generally formatted as yyyy-mm-dd or yyyy-dd-mm, unclear which."]}""" 
                },
                ### Could add a third 'few-shot example' specific to each workspace or domain it's run in. It's important that you get the schema exactly right.
                {
                    "role": "user",
                    "content": f"""Content is here - {content} and abbreviations are here - {acro_content}"""                    
                }

              ]
            }


COMMENT_PROMPT = ""
