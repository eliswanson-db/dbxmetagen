# Databricks notebook source
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# COMMAND ----------

schema = "dbx_metagen.default"
volume_name = "datasets"

# COMMAND ----------

files = dbutils.fs.ls(f"/Volumes/eswanson_genai/default/{volume_name}")

def replace_punctuation_with_underscore(text):
    return re.sub(r'\W+', '_', text)

for file_info in files:
    file_path = file_info.path
    file_name = replace_punctuation_with_underscore(os.path.basename(file_path).split('.')[0])
    df = spark.read.format("csv").option("header", "true").load(file_path)
    df.write.format("delta").mode("overwrite").saveAsTable(f"{schema}.{file_name}")
    print(f"File {file_name} has been written to Delta table in schema {schema}")
    spark.sql(f"ANALYZE TABLE {schema}.{file_name} COMPUTE STATISTICS FOR ALL COLUMNS;")

# COMMAND ----------

num_rows = 100

first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"]
last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
emails = [f"{first.lower()}.{last.lower()}@example.com" for first in first_names for last in last_names]

string_data = {
    "first_name": np.random.choice(first_names, num_rows),
    "last_name": np.random.choice(last_names, num_rows),
    "city": np.random.choice(cities, num_rows),
    "email": np.random.choice(emails, num_rows),
    "phone_number": [f"+1-555-{str(i).zfill(4)}" for i in range(num_rows)],
    "address": [f"{i} Main St" for i in range(num_rows)],
    "company": [f"Company_{i}" for i in range(num_rows)],
    "job_title": [f"Job_{i}" for i in range(num_rows)],
    "department": [f"Department_{i}" for i in range(num_rows)],
    "country": ["USA"] * num_rows
}

timestamp_data = {
    f"timestamp_col_{i}": [datetime.now() - timedelta(days=j) for j in range(num_rows)] for i in range(1, 6)
}

numeric_data = {
    "salary": np.random.randint(50000, 150000, num_rows),
    "bonus": np.random.randint(5000, 20000, num_rows),
    "years_experience": np.random.randint(1, 30, num_rows),
    "age": np.random.randint(22, 65, num_rows),
    "employee_id": np.arange(1, num_rows + 1)
}

data = {**string_data, **timestamp_data, **numeric_data}
employees_df = pd.DataFrame(data)

spark_employees_df = spark.createDataFrame(employees_df)

# COMMAND ----------

account_data = {
    "account_id": np.arange(1, num_rows + 1),
    "account_name": [f"Account_{i}" for i in range(num_rows)],
    "account_type": np.random.choice(["Savings", "Checking", "Credit"], num_rows),
    "balance": np.random.uniform(1000, 10000, num_rows),
    "created_at": [datetime.now() - timedelta(days=np.random.randint(1, 1000)) for _ in range(num_rows)]
}

accounts_df = pd.DataFrame(account_data)
spark_accounts_df = spark.createDataFrame(accounts_df)

# COMMAND ----------

order_data = {
    "order_id": np.arange(1, num_rows + 1),
    "customer_id": np.random.randint(1, num_rows + 1, num_rows),
    "order_date": [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(num_rows)],
    "order_amount": np.random.uniform(50, 500, num_rows),
    "order_status": np.random.choice(["Pending", "Shipped", "Delivered", "Cancelled"], num_rows)
}

orders_df = pd.DataFrame(order_data)
spark_orders_df = spark.createDataFrame(orders_df)

# COMMAND ----------

product_data = {
    "product_id": np.arange(1, num_rows + 1),
    "product_name": [f"Product_{i}" for i in range(num_rows)],
    "category": np.random.choice(["Electronics", "Clothing", "Home", "Beauty", "Sports"], num_rows),
    "price": np.random.uniform(10, 1000, num_rows),
    "stock_quantity": np.random.randint(1, 100, num_rows)
}

products_df = pd.DataFrame(product_data)
spark_products_df = spark.createDataFrame(products_df)

# COMMAND ----------

schema = "eswanson_genai.default"
employees_table = f"{schema}.employees"
accounts_table = f"{schema}.accounts"
orders_table = f"{schema}.orders"
products_table = f"{schema}.products"

spark_employees_df.write.format("delta").mode("overwrite").saveAsTable(employees_table)
spark_accounts_df.write.format("delta").mode("overwrite").saveAsTable(accounts_table)
spark_orders_df.write.format("delta").mode("overwrite").saveAsTable(orders_table)
spark_products_df.write.format("delta").mode("overwrite").saveAsTable(products_table)



# COMMAND ----------

spark.sql(f"ANALYZE TABLE {accounts_table} COMPUTE STATISTICS FOR ALL COLUMNS;")
spark.sql(f"ANALYZE TABLE {employees_table} COMPUTE STATISTICS FOR ALL COLUMNS;")
spark.sql(f"ANALYZE TABLE {orders_table} COMPUTE STATISTICS FOR ALL COLUMNS;")
spark.sql(f"ANALYZE TABLE {products_table} COMPUTE STATISTICS FOR ALL COLUMNS;")

# COMMAND ----------


