"""
Delta table generation utilities for cost benchmarking.
"""

from typing import List, Any
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
import random
from datetime import datetime
from faker import Faker


class BenchmarkTableGenerator:
    """Generates test tables for benchmarking dbxmetagen performance and cost."""

    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        """
        Initialize the table generator.

        Args:
            spark: SparkSession instance
            catalog: Target catalog name
            schema: Target schema name
        """
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.fake = Faker()

    def create_tables_for_scenario(
        self,
        scenario_name: str,
        num_tables: int,
        num_columns: int,
        num_rows: int,
        include_sensitive_data: bool = True,
    ) -> List[str]:
        """
        Create tables for a specific benchmarking scenario.

        Args:
            scenario_name: Name prefix for the scenario tables
            num_tables: Number of tables to create
            num_columns: Number of columns per table
            num_rows: Number of rows per table
            include_sensitive_data: Whether to include sensitive data types

        Returns:
            List of created table names
        """
        table_names = []

        for i in range(num_tables):
            table_name = f"{scenario_name}_table_{i+1}"
            full_table_name = f"{self.catalog}.{self.schema}.{table_name}"

            # Generate schema and data
            schema = self._generate_table_schema(num_columns, include_sensitive_data)
            data = self._generate_table_data(schema, num_rows)

            # Create DataFrame and save as Delta table
            df = self.spark.createDataFrame(data, schema)
            df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)

            table_names.append(full_table_name)

        return table_names

    def _generate_table_schema(
        self, num_columns: int, include_sensitive_data: bool
    ) -> StructType:
        """Generate a realistic table schema with various data types."""
        fields = []

        # Always include an ID column
        fields.append(StructField("id", IntegerType(), False))

        # Column types to choose from
        basic_types = [
            ("name", StringType()),
            ("description", StringType()),
            ("amount", DoubleType()),
            ("quantity", IntegerType()),
            ("created_at", TimestampType()),
            ("updated_at", TimestampType()),
            ("is_active", BooleanType()),
            ("category", StringType()),
            ("status", StringType()),
        ]

        sensitive_types = [
            ("email", StringType()),
            ("phone_number", StringType()),
            ("ssn", StringType()),
            ("credit_card", StringType()),
            ("patient_name", StringType()),
            ("medical_record_number", StringType()),
            ("diagnosis", StringType()),
            ("prescription", StringType()),
        ]

        # Choose column types
        available_types = basic_types + (
            sensitive_types if include_sensitive_data else []
        )

        # Select columns (excluding the id column already added)
        selected_columns = random.sample(
            available_types, min(num_columns - 1, len(available_types))
        )

        for idx, (base_name, data_type) in enumerate(selected_columns):
            col_name = f"{base_name}_{idx+1}" if idx > 0 else base_name
            fields.append(StructField(col_name, data_type, True))

        # Fill remaining columns with generic string columns if needed
        while len(fields) < num_columns:
            col_name = f"column_{len(fields)}"
            fields.append(StructField(col_name, StringType(), True))

        return StructType(fields)

    def _generate_table_data(self, schema: StructType, num_rows: int) -> List[Row]:
        """Generate realistic data for the table schema."""
        data = []

        for i in range(num_rows):
            row_data = {}

            for field in schema.fields:
                row_data[field.name] = self._generate_field_value(field)

            data.append(Row(**row_data))

        return data

    def _generate_field_value(self, field: StructField) -> Any:
        """Generate a realistic value for a field based on its name and type."""
        field_name = field.name.lower()

        if field_name == "id":
            return random.randint(1, 1000000)
        elif "email" in field_name:
            return self.fake.email()
        elif "phone" in field_name:
            return self.fake.phone_number()
        elif "ssn" in field_name:
            return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
        elif "credit_card" in field_name:
            return self.fake.credit_card_number()
        elif "name" in field_name:
            return self.fake.name()
        elif "patient_name" in field_name:
            return self.fake.name()
        elif "medical_record" in field_name:
            return f"MRN{random.randint(100000, 999999)}"
        elif "diagnosis" in field_name:
            diagnoses = [
                "Hypertension",
                "Diabetes Type 2",
                "Asthma",
                "Depression",
                "Arthritis",
            ]
            return random.choice(diagnoses)
        elif "prescription" in field_name:
            medications = [
                "Lisinopril",
                "Metformin",
                "Albuterol",
                "Sertraline",
                "Ibuprofen",
            ]
            return random.choice(medications)
        elif "description" in field_name:
            return self.fake.text(max_nb_chars=200)
        elif "amount" in field_name:
            return round(random.uniform(10.0, 10000.0), 2)
        elif "quantity" in field_name:
            return random.randint(1, 100)
        elif "created_at" in field_name or "updated_at" in field_name:
            return self.fake.date_time_between(start_date="-2y", end_date="now")
        elif "is_active" in field_name:
            return random.choice([True, False])
        elif "category" in field_name:
            categories = ["A", "B", "C", "Premium", "Standard", "Basic"]
            return random.choice(categories)
        elif "status" in field_name:
            statuses = ["Active", "Inactive", "Pending", "Completed", "Cancelled"]
            return random.choice(statuses)
        elif isinstance(field.dataType, StringType):
            return self.fake.word()
        elif isinstance(field.dataType, IntegerType):
            return random.randint(1, 1000)
        elif isinstance(field.dataType, DoubleType):
            return round(random.uniform(1.0, 1000.0), 2)
        elif isinstance(field.dataType, BooleanType):
            return random.choice([True, False])
        elif isinstance(field.dataType, TimestampType):
            return self.fake.date_time_between(start_date="-1y", end_date="now")
        else:
            return None

    def cleanup_scenario_tables(self, table_names: List[str]) -> None:
        """Clean up tables created for a scenario."""
        for table_name in table_names:
            try:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")
            except Exception as e:
                print(f"Error dropping table {table_name}: {e}")  # noqa: W0703
