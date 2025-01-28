import re
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from commonregex import CommonRegex
import pandas as pd

class PIIClassifier:
    def __init__(self, df: DataFrame):
        self.df = df
        self.regex_patterns = {
            'PII': {
                'email': CommonRegex().email,
                'phone': CommonRegex().phone,
                'ip': CommonRegex().ip,
                'ssn': re.compile(r'\b\d-\d-\d\b')
            },
            'PCI': {
                'credit_card': CommonRegex().credit_card
            },
            'PHI': {
                'date': CommonRegex().date,
                'medical_record_number': re.compile(r'\b\d{8,10}\b'),
                'health_plan_beneficiary_number': re.compile(r'\b\d\b'),
                'account_number': re.compile(r'\b\d{10,12}\b')
            },
            'Medical_Info': {
                'diagnosis': re.compile(r'\b(?:diabetes|hypertension|asthma|cancer|stroke)\b', re.IGNORECASE),
                'medication': re.compile(r'\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril)\b', re.IGNORECASE),
                'procedure': re.compile(r'\b(?:surgery|biopsy|chemotherapy|radiation|transplant)\b', re.IGNORECASE)
            }
        }

    def classify(self):
        def classify_text(text_series: pd.Series) -> pd.Series:
            classifications = []
            for text in text_series:
                if text is None:
                    classifications.append(None)
                    continue
                classified = 'None'
                for category, patterns in self.regex_patterns.items():
                    for pattern_name, pattern in patterns.items():
                        if pattern.search(text):
                            classified = category
                            break
                    if classified != 'None':
                        break
                classifications.append(classified)
            return pd.Series(classifications)

        classify_udf = pandas_udf(classify_text, returnType=StringType())
        classified_df = self.df.withColumn('classification', classify_udf(self.df['text_column']))
        return classified_df

# Example usage:
# df = spark.createDataFrame([("example@example.com",), ("123-45-6789",), ("diabetes",)], ["text_column"])
# classifier = PIIClassifier(df)
# classified_df = classifier.classify()
# display(classified_df)