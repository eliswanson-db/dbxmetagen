# Databricks notebook source
# MAGIC %md
# MAGIC # Life Sciences Sample Data Generator
# MAGIC #### Generates realistic life sciences tables for testing metadata classification
# MAGIC
# MAGIC This notebook creates sample tables relevant to life sciences customers including:
# MAGIC - Customer accounts and product orders
# MAGIC - Bulk RNA sequencing experiments
# MAGIC - Single cell datasets and drug compounds
# MAGIC - Clinical trials data

# COMMAND ----------

dbutils.widgets.text("environment", "dev", "Environment (dev/stg/prd)")
dbutils.widgets.text(
    "catalog_name", "", "Target Catalog (leave empty to use config default)"
)
dbutils.widgets.text(
    "schema_name", "", "Target Schema (leave empty to use config default)"
)
dbutils.widgets.text("num_records", "1000", "Number of sample records per table")
dbutils.widgets.dropdown(
    "include_sample_data", "true", ["true", "false"], "Generate Sample Data"
)

# COMMAND ----------

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql.types import *
import random
from datetime import datetime, timedelta
import uuid

# Get configuration
environment = dbutils.widgets.get("environment")
catalog_name_param = dbutils.widgets.get("catalog_name")
schema_name_param = dbutils.widgets.get("schema_name")
num_records = int(dbutils.widgets.get("num_records"))
include_sample_data = dbutils.widgets.get("include_sample_data").lower() == "true"

# Use config defaults if parameters are empty
if not catalog_name_param:
    # Load from dbxmetagen config
    import sys

    sys.path.append("/Workspace/Repos/dbxmetagen/src")
    from dbxmetagen.config import MetadataConfig

    config = MetadataConfig()
    catalog_name = config.catalog_name
    schema_name = (
        schema_name_param if schema_name_param else f"{config.schema_name}_demo"
    )
else:
    catalog_name = catalog_name_param
    schema_name = schema_name_param if schema_name_param else "life_sciences_demo"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark = SparkSession.getActiveSession()

logger.info(f"Generating life sciences demo data for {catalog_name}.{schema_name}")

# COMMAND ----------

# Create the schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
logger.info(f"Created schema: {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Customer Accounts Table

# COMMAND ----------


def create_accounts_table():
    """Create customer accounts table for life sciences customers"""

    # Define schema
    accounts_schema = StructType(
        [
            StructField("account_id", StringType(), False),
            StructField("account_name", StringType(), False),
            StructField("account_type", StringType(), False),
            StructField("industry_sector", StringType(), True),
            StructField("primary_contact_email", StringType(), True),
            StructField("billing_address", StringType(), True),
            StructField("country", StringType(), True),
            StructField("annual_revenue_usd", LongType(), True),
            StructField("employee_count", IntegerType(), True),
            StructField("research_focus_areas", StringType(), True),
            StructField("account_status", StringType(), False),
            StructField("created_date", TimestampType(), False),
            StructField("last_updated", TimestampType(), False),
        ]
    )

    # Sample data
    if include_sample_data:
        # Sample life sciences companies and research institutions
        sample_accounts = [
            (
                "ACC_001",
                "Moderna Inc",
                "Pharmaceutical",
                "Biotechnology",
                "research@moderna.com",
                "200 Technology Square, Cambridge, MA",
                "USA",
                18471000000,
                4900,
                "mRNA therapeutics, vaccines, infectious diseases",
                "Active",
            ),
            (
                "ACC_002",
                "Broad Institute",
                "Research Institution",
                "Genomics",
                "info@broadinstitute.org",
                "415 Main Street, Cambridge, MA",
                "USA",
                500000000,
                3000,
                "genomics, cancer research, infectious disease",
                "Active",
            ),
            (
                "ACC_003",
                "Genentech Inc",
                "Pharmaceutical",
                "Biotechnology",
                "partnerships@gene.com",
                "1 DNA Way, South San Francisco, CA",
                "USA",
                13850000000,
                13500,
                "oncology, immunology, neuroscience",
                "Active",
            ),
            (
                "ACC_004",
                "Sanger Institute",
                "Research Institution",
                "Genomics",
                "contact@sanger.ac.uk",
                "Wellcome Genome Campus, Hinxton, Cambridge",
                "UK",
                120000000,
                900,
                "genome sequencing, single cell genomics, cancer genomics",
                "Active",
            ),
            (
                "ACC_005",
                "10x Genomics",
                "Biotechnology",
                "Single Cell",
                "sales@10xgenomics.com",
                "6230 Stoneridge Mall Rd, Pleasanton, CA",
                "USA",
                501000000,
                2200,
                "single cell analysis, spatial biology, in situ analysis",
                "Active",
            ),
            (
                "ACC_006",
                "Illumina Inc",
                "Biotechnology",
                "Sequencing",
                "info@illumina.com",
                "5200 Illumina Way, San Diego, CA",
                "USA",
                4526000000,
                9000,
                "DNA sequencing, array-based technologies, genomic services",
                "Active",
            ),
            (
                "ACC_007",
                "Memorial Sloan Kettering",
                "Healthcare",
                "Cancer Research",
                "info@mskcc.org",
                "1275 York Avenue, New York, NY",
                "USA",
                5200000000,
                21000,
                "cancer treatment, oncology research, precision medicine",
                "Active",
            ),
            (
                "ACC_008",
                "Roche Diagnostics",
                "Pharmaceutical",
                "Diagnostics",
                "global.info@roche.com",
                "Grenzacherstrasse 124, Basel",
                "Switzerland",
                15615000000,
                15000,
                "diagnostics, personalized healthcare, oncology",
                "Active",
            ),
            (
                "ACC_009",
                "Chan Zuckerberg Initiative",
                "Research Institution",
                "Biomedical Research",
                "science@chanzuckerberg.com",
                "801 Jefferson Avenue, Redwood City, CA",
                "USA",
                300000000,
                500,
                "single cell biology, infectious disease, neurodegeneration",
                "Active",
            ),
            (
                "ACC_010",
                "Vertex Pharmaceuticals",
                "Pharmaceutical",
                "Rare Diseases",
                "info@vrtx.com",
                "50 Northern Avenue, Boston, MA",
                "USA",
                9687000000,
                5000,
                "cystic fibrosis, pain, alpha-1 antitrypsin deficiency",
                "Active",
            ),
        ]

        # Create sample records
        sample_data = []
        base_date = datetime(2020, 1, 1)

        for i, (
            acc_id,
            name,
            acc_type,
            sector,
            email,
            address,
            country,
            revenue,
            employees,
            focus,
            status,
        ) in enumerate(sample_accounts):
            created = base_date + timedelta(days=random.randint(0, 1000))
            updated = created + timedelta(days=random.randint(0, 100))

            sample_data.append(
                (
                    acc_id,
                    name,
                    acc_type,
                    sector,
                    email,
                    address,
                    country,
                    revenue,
                    employees,
                    focus,
                    status,
                    created,
                    updated,
                )
            )

        # Generate additional random records
        account_types = [
            "Pharmaceutical",
            "Biotechnology",
            "Research Institution",
            "Healthcare",
            "CRO",
        ]
        sectors = [
            "Oncology",
            "Immunology",
            "Neuroscience",
            "Genomics",
            "Rare Diseases",
            "Infectious Disease",
        ]
        statuses = ["Active", "Inactive", "Pending", "Suspended"]
        countries = ["USA", "UK", "Germany", "Switzerland", "Canada", "Japan", "France"]

        for i in range(
            len(sample_accounts), min(num_records, 100)
        ):  # Limit to reasonable number
            acc_id = f"ACC_{i+1:03d}"
            name = f"Life Sciences Corp {i+1}"
            acc_type = random.choice(account_types)
            sector = random.choice(sectors)
            email = f"contact{i+1}@company{i+1}.com"
            country = random.choice(countries)
            revenue = random.randint(10000000, 50000000000)
            employees = random.randint(50, 50000)
            focus = (
                f"{random.choice(sectors).lower()}, {random.choice(sectors).lower()}"
            )
            status = random.choice(statuses)
            created = base_date + timedelta(days=random.randint(0, 1000))
            updated = created + timedelta(days=random.randint(0, 100))

            sample_data.append(
                (
                    acc_id,
                    name,
                    acc_type,
                    sector,
                    email,
                    f"Address {i+1}",
                    country,
                    revenue,
                    employees,
                    focus,
                    status,
                    created,
                    updated,
                )
            )

        accounts_df = spark.createDataFrame(sample_data, accounts_schema)
    else:
        # Create empty table with schema
        accounts_df = spark.createDataFrame([], accounts_schema)

    # Write table
    table_name = f"{catalog_name}.{schema_name}.customer_accounts"
    accounts_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
        table_name
    )

    # Add table comment
    spark.sql(
        f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'comment' = 'Customer accounts including pharmaceutical companies, biotechnology firms, and research institutions. Contains contact information, financial data, and research focus areas for sales and partnership management.'
        )
    """
    )

    # Add column comments
    column_comments = {
        "account_id": "Unique identifier for customer account",
        "account_name": "Official name of the organization or institution",
        "account_type": "Category of organization: Pharmaceutical, Biotechnology, Research Institution, Healthcare, CRO",
        "industry_sector": "Primary industry focus area within life sciences",
        "primary_contact_email": "Main email address for business communications",
        "billing_address": "Primary billing and correspondence address",
        "country": "Country where the organization is headquartered",
        "annual_revenue_usd": "Annual revenue in US dollars for commercial entities",
        "employee_count": "Total number of employees across all locations",
        "research_focus_areas": "Comma-separated list of primary research and development areas",
        "account_status": "Current status: Active, Inactive, Pending, Suspended",
        "created_date": "Date when the account was first created in the system",
        "last_updated": "Timestamp of the most recent account information update",
    }

    for column, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{comment}'")

    logger.info(f"Created customer_accounts table with {accounts_df.count()} records")
    return accounts_df


accounts_df = create_accounts_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Products and Orders Table

# COMMAND ----------


def create_products_orders_table():
    """Create products and orders table for life sciences equipment and services"""

    # Define schema
    orders_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("account_id", StringType(), False),
            StructField("product_category", StringType(), False),
            StructField("product_name", StringType(), False),
            StructField("product_sku", StringType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("unit_price_usd", DoubleType(), False),
            StructField("total_amount_usd", DoubleType(), False),
            StructField("order_date", TimestampType(), False),
            StructField("delivery_date", TimestampType(), True),
            StructField("order_status", StringType(), False),
            StructField("sales_rep", StringType(), True),
            StructField("special_requirements", StringType(), True),
            StructField("contract_type", StringType(), True),
            StructField("research_application", StringType(), True),
        ]
    )

    if include_sample_data:
        # Product categories and names
        products = [
            (
                "Sequencing",
                "NovaSeq 6000 System",
                "SEQ-NS6000",
                1,
                985000.00,
                "DNA sequencing, whole genome sequencing",
            ),
            (
                "Sequencing",
                "MiSeq v3 Reagent Kit",
                "REG-MIV3-150",
                10,
                1350.00,
                "targeted sequencing, amplicon sequencing",
            ),
            (
                "Single Cell",
                "Chromium Controller",
                "SC-CHR-CTRL",
                1,
                125000.00,
                "single cell RNA sequencing",
            ),
            (
                "Single Cell",
                "Chromium Single Cell 3' v3.1",
                "SC-3P-V31",
                5,
                2100.00,
                "single cell gene expression",
            ),
            (
                "Single Cell",
                "Chromium Single Cell ATAC",
                "SC-ATAC-V11",
                3,
                3200.00,
                "single cell chromatin accessibility",
            ),
            (
                "Reagents",
                "TruSeq Stranded mRNA Library Prep",
                "REG-TS-MRNA",
                20,
                425.00,
                "RNA library preparation",
            ),
            (
                "Reagents",
                "KAPA HyperPlus Kit",
                "REG-KAPA-HP",
                15,
                780.00,
                "DNA library preparation",
            ),
            (
                "Instruments",
                "HiSeq 4000 System",
                "SEQ-HS4000",
                1,
                740000.00,
                "high-throughput sequencing",
            ),
            (
                "Services",
                "Whole Genome Sequencing Service",
                "SVC-WGS-30X",
                50,
                850.00,
                "genomic services, population studies",
            ),
            (
                "Services",
                "RNA-seq Analysis Pipeline",
                "SVC-RNA-PIPE",
                25,
                1200.00,
                "bioinformatics, differential expression",
            ),
            (
                "Spatial",
                "Visium Spatial Gene Expression",
                "SP-VIS-GEX",
                4,
                4500.00,
                "spatial transcriptomics",
            ),
            (
                "Reagents",
                "AMPure XP Beads",
                "REG-AMP-XP",
                100,
                95.00,
                "DNA/RNA purification",
            ),
            (
                "Instruments",
                "Element AVITI System",
                "SEQ-AVITI",
                1,
                395000.00,
                "benchtop sequencing",
            ),
            (
                "Single Cell",
                "CellPlex Kit",
                "SC-CPLX-KIT",
                8,
                850.00,
                "cell multiplexing, sample barcoding",
            ),
            (
                "Services",
                "Single Cell Analysis Service",
                "SVC-SC-ANAL",
                12,
                2500.00,
                "single cell bioinformatics",
            ),
        ]

        # Get account IDs from the accounts table we just created
        account_ids = [
            row.account_id for row in accounts_df.select("account_id").collect()
        ]

        sample_data = []
        order_statuses = [
            "Completed",
            "Processing",
            "Shipped",
            "Delivered",
            "Cancelled",
            "Pending",
        ]
        contract_types = [
            "Standard",
            "Academic Discount",
            "Volume Agreement",
            "Service Contract",
            "Lease",
        ]
        sales_reps = [
            "Alice Chen",
            "Robert Martinez",
            "Sarah Johnson",
            "Michael Kim",
            "Jessica Wong",
            "David Brown",
        ]

        base_date = datetime(2023, 1, 1)

        for i in range(min(num_records, 500)):  # Limit to reasonable number
            product_cat, product_name, sku, base_qty, unit_price, application = (
                random.choice(products)
            )

            order_id = f"ORD-{i+1:06d}"
            account_id = (
                random.choice(account_ids)
                if account_ids
                else f"ACC_{random.randint(1, 10):03d}"
            )
            quantity = base_qty + random.randint(0, 10)
            actual_unit_price = unit_price * random.uniform(0.8, 1.2)  # Price variation
            total_amount = quantity * actual_unit_price

            order_date = base_date + timedelta(days=random.randint(0, 365))
            delivery_offset = random.randint(5, 45)
            delivery_date = (
                order_date + timedelta(days=delivery_offset)
                if random.random() > 0.1
                else None
            )

            status = random.choice(order_statuses)
            sales_rep = random.choice(sales_reps)
            contract = random.choice(contract_types)

            special_req = None
            if random.random() < 0.3:  # 30% chance of special requirements
                special_reqs = [
                    "Cold chain shipping",
                    "Express delivery",
                    "Custom installation",
                    "Training included",
                    "Extended warranty",
                ]
                special_req = random.choice(special_reqs)

            sample_data.append(
                (
                    order_id,
                    account_id,
                    product_cat,
                    product_name,
                    sku,
                    quantity,
                    actual_unit_price,
                    total_amount,
                    order_date,
                    delivery_date,
                    status,
                    sales_rep,
                    special_req,
                    contract,
                    application,
                )
            )

        orders_df = spark.createDataFrame(sample_data, orders_schema)
    else:
        orders_df = spark.createDataFrame([], orders_schema)

    # Write table
    table_name = f"{catalog_name}.{schema_name}.products_orders"
    orders_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
        table_name
    )

    # Add table comment
    spark.sql(
        f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'comment' = 'Product orders and sales data for life sciences equipment, reagents, and services. Includes sequencing systems, single cell analysis tools, reagents, and bioinformatics services purchased by research institutions and pharmaceutical companies.'
        )
    """
    )

    # Add column comments
    column_comments = {
        "order_id": "Unique identifier for each product order",
        "account_id": "Foreign key reference to customer_accounts table",
        "product_category": "High-level product category: Sequencing, Single Cell, Reagents, Instruments, Services, Spatial",
        "product_name": "Specific name of the product or service ordered",
        "product_sku": "Stock keeping unit identifier for inventory management",
        "quantity": "Number of units ordered",
        "unit_price_usd": "Price per unit in US dollars at time of order",
        "total_amount_usd": "Total order value calculated as quantity × unit_price_usd",
        "order_date": "Date when the order was placed by the customer",
        "delivery_date": "Actual or expected delivery date, null if not yet scheduled",
        "order_status": "Current order status: Completed, Processing, Shipped, Delivered, Cancelled, Pending",
        "sales_rep": "Name of the sales representative who handled the order",
        "special_requirements": "Any special shipping, installation, or service requirements",
        "contract_type": "Type of commercial agreement: Standard, Academic Discount, Volume Agreement, etc.",
        "research_application": "Intended research use case or application area",
    }

    for column, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{comment}'")

    logger.info(f"Created products_orders table with {orders_df.count()} records")
    return orders_df


orders_df = create_products_orders_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bulk RNA Sequencing Experiments

# COMMAND ----------


def create_bulk_rnaseq_table():
    """Create bulk RNA sequencing experiments table"""

    # Define schema
    rnaseq_schema = StructType(
        [
            StructField("experiment_id", StringType(), False),
            StructField("project_name", StringType(), False),
            StructField("principal_investigator", StringType(), False),
            StructField("research_team", StringType(), True),
            StructField("experiment_type", StringType(), False),
            StructField("tissue_type", StringType(), True),
            StructField("organism", StringType(), False),
            StructField("strain_genotype", StringType(), True),
            StructField("treatment_condition", StringType(), True),
            StructField("sample_count", IntegerType(), False),
            StructField("replicate_count", IntegerType(), False),
            StructField("sequencing_platform", StringType(), False),
            StructField("read_length", IntegerType(), False),
            StructField("read_type", StringType(), False),
            StructField("target_depth_million_reads", DoubleType(), False),
            StructField("library_prep_protocol", StringType(), False),
            StructField("experiment_date", TimestampType(), False),
            StructField("sequencing_date", TimestampType(), True),
            StructField("data_analysis_status", StringType(), False),
            StructField("raw_data_path", StringType(), True),
            StructField("processed_data_path", StringType(), True),
            StructField("publication_status", StringType(), True),
            StructField("funding_source", StringType(), True),
            StructField("experimental_notes", StringType(), True),
        ]
    )

    if include_sample_data:
        # Sample experiments
        sample_experiments = [
            (
                "EXP-RNA-001",
                "Cancer Drug Response",
                "Dr. Sarah Chen",
                "Chen Lab Oncology Team",
                "Drug Response",
                "Tumor Tissue",
                "Homo sapiens",
                "Primary breast cancer",
                "Doxorubicin treatment vs control",
                48,
                6,
                "Illumina NovaSeq 6000",
                150,
                "Paired-end",
                30.0,
                "TruSeq Stranded mRNA",
                "RNA expression profiling of breast cancer cells",
            ),
            (
                "EXP-RNA-002",
                "Neurodegeneration Study",
                "Dr. Michael Rodriguez",
                "Rodriguez Neuroscience Group",
                "Disease Progression",
                "Brain Tissue",
                "Mus musculus",
                "5xFAD (Alzheimer model)",
                "Age progression: 3, 6, 12 months",
                72,
                8,
                "Illumina HiSeq 4000",
                100,
                "Single-end",
                25.0,
                "KAPA mRNA HyperPrep",
                "Temporal analysis of Alzheimer's disease progression",
            ),
            (
                "EXP-RNA-003",
                "Immune Response Profiling",
                "Dr. Jennifer Wang",
                "Wang Immunology Lab",
                "Immune Activation",
                "PBMC",
                "Homo sapiens",
                "Healthy volunteers",
                "LPS stimulation time course",
                36,
                6,
                "Illumina MiSeq",
                75,
                "Paired-end",
                15.0,
                "NEBNext Ultra II",
                "Innate immune response to bacterial endotoxin",
            ),
            (
                "EXP-RNA-004",
                "Stem Cell Differentiation",
                "Dr. Robert Kim",
                "Kim Developmental Biology Lab",
                "Differentiation",
                "iPSC-derived",
                "Homo sapiens",
                "iPSC lines from 3 donors",
                "Cardiac differentiation protocol",
                60,
                5,
                "Illumina NovaSeq 6000",
                150,
                "Paired-end",
                40.0,
                "TruSeq Stranded Total RNA",
                "Cardiac differentiation from induced pluripotent stem cells",
            ),
            (
                "EXP-RNA-005",
                "Drug Toxicity Screen",
                "Dr. Lisa Thompson",
                "Thompson Pharmacology Team",
                "Toxicology",
                "Hepatocytes",
                "Homo sapiens",
                "Primary hepatocytes",
                "Acetaminophen dose response",
                30,
                5,
                "Illumina HiSeq 4000",
                100,
                "Single-end",
                20.0,
                "KAPA mRNA HyperPrep",
                "Hepatotoxicity mechanisms of acetaminophen overdose",
            ),
        ]

        sample_data = []
        platforms = [
            "Illumina NovaSeq 6000",
            "Illumina HiSeq 4000",
            "Illumina MiSeq",
            "BGI DNBSEQ-T7",
        ]
        organisms = [
            "Homo sapiens",
            "Mus musculus",
            "Rattus norvegicus",
            "Macaca mulatta",
        ]
        tissues = [
            "Brain Tissue",
            "Liver",
            "Heart",
            "Kidney",
            "Lung",
            "PBMC",
            "Tumor Tissue",
            "Muscle",
            "Skin",
            "iPSC-derived",
        ]
        exp_types = [
            "Disease Progression",
            "Drug Response",
            "Immune Activation",
            "Differentiation",
            "Toxicology",
            "Aging",
            "Stress Response",
        ]
        analysis_statuses = [
            "Raw Data",
            "Quality Control",
            "Alignment",
            "Quantification",
            "Differential Expression",
            "Pathway Analysis",
            "Complete",
        ]
        pub_statuses = [
            "Unpublished",
            "In Preparation",
            "Submitted",
            "Published",
            "Preprint",
        ]
        funding = [
            "NIH R01",
            "NSF Grant",
            "Industry Collaboration",
            "Internal Funding",
            "NIH R21",
            "DOD Grant",
            "Foundation Grant",
        ]

        base_date = datetime(2022, 1, 1)

        for i, (
            exp_id,
            project,
            pi,
            team,
            exp_type,
            tissue,
            organism,
            strain,
            treatment,
            samples,
            reps,
            platform,
            read_len,
            read_type,
            depth,
            lib_prep,
            notes,
        ) in enumerate(sample_experiments):
            exp_date = base_date + timedelta(days=random.randint(0, 500))
            seq_date = exp_date + timedelta(days=random.randint(7, 30))

            raw_path = f"/mnt/data/raw_rnaseq/{exp_id}/"
            processed_path = f"/mnt/data/processed_rnaseq/{exp_id}/"

            sample_data.append(
                (
                    exp_id,
                    project,
                    pi,
                    team,
                    exp_type,
                    tissue,
                    organism,
                    strain,
                    treatment,
                    samples,
                    reps,
                    platform,
                    read_len,
                    read_type,
                    depth,
                    lib_prep,
                    exp_date,
                    seq_date,
                    random.choice(analysis_statuses),
                    raw_path,
                    processed_path,
                    random.choice(pub_statuses),
                    random.choice(funding),
                    notes,
                )
            )

        # Generate additional random experiments
        for i in range(len(sample_experiments), min(num_records, 50)):
            exp_id = f"EXP-RNA-{i+1:03d}"
            project = f"Research Project {i+1}"
            pi = f"Dr. Investigator {i+1}"
            team = f"Research Team {i+1}"
            exp_type = random.choice(exp_types)
            tissue = random.choice(tissues)
            organism = random.choice(organisms)
            strain = f"Strain {random.randint(1, 10)}"
            treatment = f"Treatment condition {random.randint(1, 5)}"
            samples = random.randint(12, 96)
            reps = random.randint(3, 8)
            platform = random.choice(platforms)
            read_len = random.choice([75, 100, 150])
            read_type = random.choice(["Single-end", "Paired-end"])
            depth = random.uniform(15.0, 50.0)
            lib_prep = random.choice(
                ["TruSeq Stranded mRNA", "KAPA mRNA HyperPrep", "NEBNext Ultra II"]
            )

            exp_date = base_date + timedelta(days=random.randint(0, 600))
            seq_date = exp_date + timedelta(days=random.randint(7, 30))

            raw_path = f"/mnt/data/raw_rnaseq/{exp_id}/"
            processed_path = f"/mnt/data/processed_rnaseq/{exp_id}/"
            notes = f"RNA sequencing experiment {i+1} for {exp_type.lower()} studies"

            sample_data.append(
                (
                    exp_id,
                    project,
                    pi,
                    team,
                    exp_type,
                    tissue,
                    organism,
                    strain,
                    treatment,
                    samples,
                    reps,
                    platform,
                    read_len,
                    read_type,
                    depth,
                    lib_prep,
                    exp_date,
                    seq_date,
                    random.choice(analysis_statuses),
                    raw_path,
                    processed_path,
                    random.choice(pub_statuses),
                    random.choice(funding),
                    notes,
                )
            )

        rnaseq_df = spark.createDataFrame(sample_data, rnaseq_schema)
    else:
        rnaseq_df = spark.createDataFrame([], rnaseq_schema)

    # Write table
    table_name = f"{catalog_name}.{schema_name}.bulk_rnaseq_experiments"
    rnaseq_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
        table_name
    )

    # Add table comment
    spark.sql(
        f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'comment' = 'Bulk RNA sequencing experiments conducted by research teams. Contains experiment metadata, sample information, sequencing parameters, and analysis status for gene expression studies across various biological conditions and treatments.'
        )
    """
    )

    # Add column comments
    column_comments = {
        "experiment_id": "Unique identifier for each RNA sequencing experiment",
        "project_name": "Name of the research project or study",
        "principal_investigator": "Lead researcher responsible for the experiment",
        "research_team": "Name of the laboratory or research group conducting the study",
        "experiment_type": "Category of experiment: Disease Progression, Drug Response, Immune Activation, etc.",
        "tissue_type": "Type of tissue or cell type used in the experiment",
        "organism": "Scientific name of the organism being studied",
        "strain_genotype": "Specific strain, genotype, or genetic background information",
        "treatment_condition": "Experimental treatment, drug, or condition applied to samples",
        "sample_count": "Total number of biological samples in the experiment",
        "replicate_count": "Number of biological replicates per condition",
        "sequencing_platform": "Sequencing instrument used (e.g., Illumina NovaSeq 6000)",
        "read_length": "Length of sequencing reads in base pairs",
        "read_type": "Single-end or paired-end sequencing configuration",
        "target_depth_million_reads": "Target sequencing depth in millions of reads per sample",
        "library_prep_protocol": "RNA library preparation kit or protocol used",
        "experiment_date": "Date when the biological experiment was conducted",
        "sequencing_date": "Date when sequencing was performed",
        "data_analysis_status": "Current stage of bioinformatics analysis pipeline",
        "raw_data_path": "File system path to raw FASTQ sequencing files",
        "processed_data_path": "File system path to processed analysis results",
        "publication_status": "Publication status: Unpublished, In Preparation, Submitted, Published",
        "funding_source": "Primary funding agency or grant supporting the research",
        "experimental_notes": "Additional notes about experimental design or methodology",
    }

    for column, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{comment}'")

    logger.info(
        f"Created bulk_rnaseq_experiments table with {rnaseq_df.count()} records"
    )
    return rnaseq_df


rnaseq_df = create_bulk_rnaseq_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Single Cell Datasets

# COMMAND ----------


def create_single_cell_datasets_table():
    """Create single cell datasets table"""

    # Define schema
    sc_schema = StructType(
        [
            StructField("dataset_id", StringType(), False),
            StructField("dataset_name", StringType(), False),
            StructField("technology_platform", StringType(), False),
            StructField("assay_type", StringType(), False),
            StructField("organism", StringType(), False),
            StructField("tissue_organ", StringType(), False),
            StructField("cell_type_annotation", StringType(), True),
            StructField("disease_condition", StringType(), True),
            StructField("treatment_perturbation", StringType(), True),
            StructField("donor_count", IntegerType(), False),
            StructField("total_cell_count", IntegerType(), False),
            StructField("median_genes_per_cell", IntegerType(), True),
            StructField("median_umis_per_cell", IntegerType(), True),
            StructField("sequencing_chemistry", StringType(), False),
            StructField("library_prep_date", TimestampType(), False),
            StructField("sequencing_completion_date", TimestampType(), True),
            StructField("primary_investigator", StringType(), False),
            StructField("institution", StringType(), False),
            StructField("data_processing_pipeline", StringType(), True),
            StructField("quality_control_metrics", StringType(), True),
            StructField("raw_data_location", StringType(), True),
            StructField("processed_data_location", StringType(), True),
            StructField("public_repository_id", StringType(), True),
            StructField("associated_publication", StringType(), True),
            StructField("data_access_level", StringType(), False),
        ]
    )

    if include_sample_data:
        # Sample single cell datasets
        sample_datasets = [
            (
                "SC-001",
                "Human Brain Cell Atlas",
                "10x Genomics Chromium",
                "scRNA-seq",
                "Homo sapiens",
                "Brain",
                "Neurons, Astrocytes, Microglia, Oligodendrocytes",
                "Healthy",
                "None",
                12,
                156000,
                2400,
                8500,
                "Single Cell 3' v3.1",
                "Cell Ranger 6.0",
                "Ambient RNA removal, doublet detection, batch correction",
            ),
            (
                "SC-002",
                "COVID-19 PBMC Response",
                "10x Genomics Chromium",
                "scRNA-seq",
                "Homo sapiens",
                "Blood",
                "T cells, B cells, Monocytes, NK cells, Dendritic cells",
                "COVID-19",
                "SARS-CoV-2 infection",
                24,
                89000,
                1800,
                6200,
                "Single Cell 3' v3.1",
                "Cell Ranger 6.1",
                "Standard quality filtering, cell type annotation",
            ),
            (
                "SC-003",
                "Cardiac Development Atlas",
                "10x Genomics Chromium",
                "scRNA-seq",
                "Mus musculus",
                "Heart",
                "Cardiomyocytes, Fibroblasts, Endothelial cells",
                "Healthy",
                "Developmental time course",
                8,
                67000,
                2100,
                7800,
                "Single Cell 3' v3",
                "Cell Ranger 5.0",
                "Trajectory analysis, pseudotime ordering",
            ),
            (
                "SC-004",
                "Drug Perturbation Screen",
                "10x Genomics Chromium",
                "scRNA-seq",
                "Homo sapiens",
                "Liver",
                "Hepatocytes, Kupffer cells, Stellate cells",
                "Healthy",
                "Compound library (50 drugs)",
                6,
                45000,
                1950,
                7100,
                "Single Cell 3' v3.1",
                "Cell Ranger 6.0",
                "Drug response scoring, pathway analysis",
            ),
            (
                "SC-005",
                "Tumor Microenvironment",
                "10x Genomics Chromium",
                "scRNA-seq",
                "Homo sapiens",
                "Tumor",
                "Cancer cells, CAFs, TAMs, T cells, B cells",
                "Breast cancer",
                "Immunotherapy treatment",
                18,
                125000,
                2200,
                8900,
                "Single Cell 3' v3.1",
                "Cell Ranger 6.1",
                "Cancer cell scoring, immune infiltration analysis",
            ),
            (
                "SC-006",
                "Chromatin Accessibility",
                "10x Genomics Chromium",
                "scATAC-seq",
                "Homo sapiens",
                "Brain",
                "Excitatory neurons, Inhibitory neurons, Glia",
                "Alzheimer's disease",
                "Disease progression",
                15,
                78000,
                1200,
                5000,
                "Single Cell ATAC v1.1",
                "Cell Ranger ATAC 2.0",
                "Peak calling, motif analysis, gene activity scoring",
            ),
            (
                "SC-007",
                "Immune Cell Diversity",
                "10x Genomics Chromium",
                "scRNA-seq",
                "Mus musculus",
                "Spleen",
                "T cells, B cells, Macrophages, Dendritic cells",
                "Healthy",
                "Age comparison",
                16,
                92000,
                2000,
                7500,
                "Single Cell 3' v3",
                "Cell Ranger 5.0",
                "Cell type annotation, aging signatures",
            ),
            (
                "SC-008",
                "Spatial Transcriptomics",
                "10x Genomics Visium",
                "Spatial",
                "Homo sapiens",
                "Kidney",
                "Glomeruli, Tubules, Interstitium",
                "Chronic kidney disease",
                "Disease progression",
                10,
                35000,
                3500,
                12000,
                "Visium v1",
                "Space Ranger 1.3",
                "Spatial clustering, pathway enrichment",
            ),
        ]

        sample_data = []
        platforms = [
            "10x Genomics Chromium",
            "Smart-seq2",
            "Drop-seq",
            "inDrop",
            "MARS-seq",
        ]
        assays = ["scRNA-seq", "scATAC-seq", "scMultiome", "Spatial", "CITE-seq"]
        organisms = [
            "Homo sapiens",
            "Mus musculus",
            "Rattus norvegicus",
            "Macaca mulatta",
        ]
        tissues = [
            "Brain",
            "Heart",
            "Liver",
            "Lung",
            "Kidney",
            "Blood",
            "Skin",
            "Muscle",
            "Tumor",
            "Spleen",
            "Bone marrow",
        ]
        diseases = [
            "Healthy",
            "Cancer",
            "Alzheimer's disease",
            "Diabetes",
            "COVID-19",
            "Autoimmune",
            "Cardiovascular disease",
        ]
        access_levels = ["Public", "Controlled", "Private", "Embargo"]
        institutions = [
            "Harvard Medical School",
            "Stanford University",
            "MIT",
            "UCSF",
            "Broad Institute",
            "Sanger Institute",
        ]

        base_date = datetime(2022, 6, 1)

        for i, (
            ds_id,
            name,
            platform,
            assay,
            organism,
            tissue,
            cells,
            disease,
            treatment,
            donors,
            total_cells,
            genes,
            umis,
            chemistry,
            pipeline,
            qc,
        ) in enumerate(sample_datasets):
            lib_date = base_date + timedelta(days=random.randint(0, 400))
            seq_date = lib_date + timedelta(days=random.randint(3, 14))

            pi = f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller'])}"
            institution = random.choice(institutions)

            raw_location = f"/mnt/single_cell/raw/{ds_id}/"
            processed_location = f"/mnt/single_cell/processed/{ds_id}/"
            repo_id = (
                f"GSE{random.randint(100000, 999999)}"
                if random.random() > 0.3
                else None
            )
            publication = f"Nature Medicine 2023" if random.random() > 0.6 else None
            access = random.choice(access_levels)

            sample_data.append(
                (
                    ds_id,
                    name,
                    platform,
                    assay,
                    organism,
                    tissue,
                    cells,
                    disease,
                    treatment,
                    donors,
                    total_cells,
                    genes,
                    umis,
                    chemistry,
                    lib_date,
                    seq_date,
                    pi,
                    institution,
                    pipeline,
                    qc,
                    raw_location,
                    processed_location,
                    repo_id,
                    publication,
                    access,
                )
            )

        # Generate additional random datasets
        for i in range(len(sample_datasets), min(num_records, 30)):
            ds_id = f"SC-{i+1:03d}"
            name = f"Single Cell Study {i+1}"
            platform = random.choice(platforms)
            assay = random.choice(assays)
            organism = random.choice(organisms)
            tissue = random.choice(tissues)
            disease = random.choice(diseases)
            treatment = "Treatment A" if random.random() > 0.5 else "None"
            donors = random.randint(3, 30)
            total_cells = random.randint(10000, 200000)
            genes = random.randint(1000, 4000)
            umis = random.randint(3000, 15000)
            chemistry = "Single Cell 3' v3.1"

            lib_date = base_date + timedelta(days=random.randint(0, 500))
            seq_date = lib_date + timedelta(days=random.randint(3, 14))

            pi = f"Dr. Investigator {i+1}"
            institution = random.choice(institutions)
            pipeline = "Standard pipeline"
            qc = "Quality control applied"

            raw_location = f"/mnt/single_cell/raw/{ds_id}/"
            processed_location = f"/mnt/single_cell/processed/{ds_id}/"
            repo_id = (
                f"GSE{random.randint(100000, 999999)}"
                if random.random() > 0.4
                else None
            )
            publication = None
            access = random.choice(access_levels)

            cells = "Mixed cell types"

            sample_data.append(
                (
                    ds_id,
                    name,
                    platform,
                    assay,
                    organism,
                    tissue,
                    cells,
                    disease,
                    treatment,
                    donors,
                    total_cells,
                    genes,
                    umis,
                    chemistry,
                    lib_date,
                    seq_date,
                    pi,
                    institution,
                    pipeline,
                    qc,
                    raw_location,
                    processed_location,
                    repo_id,
                    publication,
                    access,
                )
            )

        sc_df = spark.createDataFrame(sample_data, sc_schema)
    else:
        sc_df = spark.createDataFrame([], sc_schema)

    # Write table
    table_name = f"{catalog_name}.{schema_name}.single_cell_datasets"
    sc_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(table_name)

    # Add table comment
    spark.sql(
        f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'comment' = 'Single cell genomics datasets including scRNA-seq, scATAC-seq, and spatial transcriptomics experiments. Contains comprehensive metadata about cell populations, experimental conditions, quality metrics, and data processing information for single cell analysis studies.'
        )
    """
    )

    # Add column comments
    column_comments = {
        "dataset_id": "Unique identifier for each single cell dataset",
        "dataset_name": "Descriptive name of the single cell study or experiment",
        "technology_platform": "Single cell technology platform used (e.g., 10x Genomics Chromium, Smart-seq2)",
        "assay_type": "Type of single cell assay: scRNA-seq, scATAC-seq, scMultiome, Spatial, CITE-seq",
        "organism": "Scientific name of the organism studied",
        "tissue_organ": "Tissue or organ from which cells were isolated",
        "cell_type_annotation": "Identified cell types and populations in the dataset",
        "disease_condition": "Disease state or health condition of the samples",
        "treatment_perturbation": "Experimental treatment, drug, or perturbation applied",
        "donor_count": "Number of individual donors or biological samples",
        "total_cell_count": "Total number of single cells profiled in the dataset",
        "median_genes_per_cell": "Median number of genes detected per cell",
        "median_umis_per_cell": "Median number of unique molecular identifiers (UMIs) per cell",
        "sequencing_chemistry": "Single cell chemistry version used for library preparation",
        "library_prep_date": "Date when single cell libraries were prepared",
        "sequencing_completion_date": "Date when sequencing was completed",
        "primary_investigator": "Lead researcher responsible for the study",
        "institution": "Research institution where the study was conducted",
        "data_processing_pipeline": "Bioinformatics pipeline and software used for data processing",
        "quality_control_metrics": "Quality control measures and filtering criteria applied",
        "raw_data_location": "File system path to raw sequencing data (FASTQ files)",
        "processed_data_location": "File system path to processed count matrices and analysis results",
        "public_repository_id": "Public repository accession number (e.g., GEO, SRA)",
        "associated_publication": "Published paper associated with the dataset",
        "data_access_level": "Data access restrictions: Public, Controlled, Private, Embargo",
    }

    for column, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{comment}'")

    logger.info(f"Created single_cell_datasets table with {sc_df.count()} records")
    return sc_df


sc_df = create_single_cell_datasets_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Drug Compounds

# COMMAND ----------


def create_drug_compounds_table():
    """Create drug compounds table linked to single cell datasets"""

    # Define schema
    drug_schema = StructType(
        [
            StructField("compound_id", StringType(), False),
            StructField("compound_name", StringType(), False),
            StructField("chemical_formula", StringType(), True),
            StructField("molecular_weight", DoubleType(), True),
            StructField("smiles_string", StringType(), True),
            StructField("inchi_key", StringType(), True),
            StructField("drug_class", StringType(), False),
            StructField("mechanism_of_action", StringType(), True),
            StructField("target_protein", StringType(), True),
            StructField("therapeutic_area", StringType(), False),
            StructField("development_stage", StringType(), False),
            StructField("approval_status", StringType(), False),
            StructField("approved_indications", StringType(), True),
            StructField("dosage_form", StringType(), True),
            StructField("route_of_administration", StringType(), True),
            StructField("half_life_hours", DoubleType(), True),
            StructField("bioavailability_percent", DoubleType(), True),
            StructField("manufacturer", StringType(), True),
            StructField("patent_expiry_date", TimestampType(), True),
            StructField("first_approval_date", TimestampType(), True),
            StructField("associated_single_cell_datasets", StringType(), True),
            StructField("preclinical_studies_count", IntegerType(), True),
            StructField("clinical_trials_count", IntegerType(), True),
            StructField("safety_profile", StringType(), True),
            StructField("drug_interactions", StringType(), True),
        ]
    )

    if include_sample_data:
        # Sample drug compounds
        sample_drugs = [
            (
                "CMP-001",
                "Doxorubicin",
                "C27H29NO11",
                543.52,
                "CC1C(C(CC(O1)OC2C(CC(C(C2O)O)N)O)N)O",
                "AOJJSUZBOXZQNB-TZSSRYMLSA-N",
                "Anthracycline",
                "DNA intercalation, topoisomerase II inhibition",
                "Topoisomerase II",
                "Oncology",
                "Approved",
                "FDA Approved",
                "Breast cancer, lymphoma, sarcoma",
                "Injectable",
                "Intravenous",
                20.0,
                5.0,
                "Pfizer",
                None,
                "1974-08-01",
                "SC-002,SC-005",
                150,
                45,
                "Cardiotoxicity, myelosuppression",
                "Cyclosporine, phenytoin",
            ),
            (
                "CMP-002",
                "Pembrolizumab",
                "C6374H9788N1724O1988S52",
                148134.0,
                None,
                None,
                "Monoclonal Antibody",
                "PD-1 checkpoint inhibition",
                "PD-1 receptor",
                "Oncology",
                "Approved",
                "FDA Approved",
                "Melanoma, lung cancer, renal cell carcinoma",
                "Injectable",
                "Intravenous",
                500.0,
                100.0,
                "Merck",
                "2028-12-31",
                "2014-09-04",
                "SC-005,SC-006",
                85,
                120,
                "Immune-related adverse events",
                "None significant",
            ),
            (
                "CMP-003",
                "Metformin",
                "C4H11N5",
                129.16,
                "CN(C)C(=N)NC(=N)N",
                "XZWYZXLIPXDOLR-UHFFFAOYSA-N",
                "Biguanide",
                "AMPK activation, gluconeogenesis inhibition",
                "AMPK",
                "Endocrinology",
                "Approved",
                "FDA Approved",
                "Type 2 diabetes mellitus",
                "Tablet",
                "Oral",
                6.2,
                50.0,
                "Teva",
                None,
                "1995-03-03",
                "SC-007",
                200,
                80,
                "Gastrointestinal effects, lactic acidosis (rare)",
                "Cimetidine, furosemide",
            ),
            (
                "CMP-004",
                "Adalimumab",
                "C6428H9912N1694O2018S46",
                148000.0,
                None,
                None,
                "Monoclonal Antibody",
                "TNF-alpha inhibition",
                "TNF-alpha",
                "Immunology",
                "Approved",
                "FDA Approved",
                "Rheumatoid arthritis, Crohn's disease, psoriasis",
                "Injectable",
                "Subcutaneous",
                336.0,
                64.0,
                "AbbVie",
                "2016-12-31",
                "2002-12-31",
                "SC-007,SC-008",
                95,
                200,
                "Increased infection risk, injection site reactions",
                "Live vaccines contraindicated",
            ),
            (
                "CMP-005",
                "Remdesivir",
                "C27H35N6O8P",
                602.58,
                "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1[C@@H]([C@H]([C@@H](O1)N2C=CC(=NC2=O)N)O)O)OC3=CC=CC=C3",
                "RWWYLEGWBNMMLJ-YSOARWBDSA-N",
                "Nucleotide Analog",
                "RNA polymerase inhibition",
                "Viral RNA polymerase",
                "Infectious Disease",
                "Approved",
                "FDA Approved",
                "COVID-19",
                "Injectable",
                "Intravenous",
                1.0,
                100.0,
                "Gilead Sciences",
                "2030-05-01",
                "2020-05-01",
                "SC-002",
                25,
                15,
                "Hepatotoxicity, renal impairment",
                "Chloroquine, hydroxychloroquine",
            ),
            (
                "CMP-006",
                "Osimertinib",
                "C28H33N7O2",
                499.61,
                "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C",
                "VZJUSQOVDMKTLR-UHFFFAOYSA-N",
                "EGFR Inhibitor",
                "EGFR tyrosine kinase inhibition",
                "EGFR",
                "Oncology",
                "Approved",
                "FDA Approved",
                "Non-small cell lung cancer",
                "Tablet",
                "Oral",
                48.0,
                70.0,
                "AstraZeneca",
                "2025-11-13",
                "2015-11-13",
                "SC-004,SC-005",
                40,
                25,
                "Diarrhea, rash, decreased appetite",
                "CYP3A4 inducers",
            ),
            (
                "CMP-007",
                "Rituximab",
                "C6416H9874N1688O1987S44",
                145531.5,
                None,
                None,
                "Monoclonal Antibody",
                "CD20 B-cell depletion",
                "CD20",
                "Oncology",
                "Approved",
                "FDA Approved",
                "Non-Hodgkin lymphoma, chronic lymphocytic leukemia",
                "Injectable",
                "Intravenous",
                206.0,
                100.0,
                "Roche",
                "2015-11-26",
                "1997-11-26",
                "SC-002,SC-007",
                120,
                180,
                "Infusion reactions, tumor lysis syndrome",
                "None significant",
            ),
            (
                "CMP-008",
                "Donepezil",
                "C24H29NO3",
                379.49,
                "COC1=CC=C(C=C1)C(C2=CC=CC=C2)N3CCN(CC3)CC(=O)C4=CC=CC=C4",
                "ADEBDKJRFQWKAU-UHFFFAOYSA-N",
                "Acetylcholinesterase Inhibitor",
                "Acetylcholinesterase inhibition",
                "Acetylcholinesterase",
                "Neurology",
                "Approved",
                "FDA Approved",
                "Alzheimer's disease",
                "Tablet",
                "Oral",
                70.0,
                100.0,
                "Eisai",
                "2010-11-25",
                "1996-11-25",
                "SC-001,SC-006",
                85,
                60,
                "Nausea, diarrhea, insomnia",
                "Ketoconazole, quinidine",
            ),
        ]

        sample_data = []
        drug_classes = [
            "Kinase Inhibitor",
            "Monoclonal Antibody",
            "Small Molecule",
            "Protein Therapy",
            "Gene Therapy",
            "Chemotherapy",
            "Immunotherapy",
        ]
        therapeutic_areas = [
            "Oncology",
            "Immunology",
            "Neurology",
            "Cardiology",
            "Infectious Disease",
            "Endocrinology",
            "Dermatology",
        ]
        dev_stages = [
            "Preclinical",
            "Phase I",
            "Phase II",
            "Phase III",
            "Approved",
            "Discontinued",
        ]
        approval_statuses = [
            "FDA Approved",
            "EMA Approved",
            "Investigational",
            "Orphan Drug",
            "Fast Track",
        ]
        manufacturers = [
            "Pfizer",
            "Roche",
            "Novartis",
            "Merck",
            "AstraZeneca",
            "Bristol Myers Squibb",
            "Gilead Sciences",
            "AbbVie",
        ]

        # Get single cell dataset IDs
        sc_dataset_ids = [
            row.dataset_id for row in sc_df.select("dataset_id").collect()
        ]

        base_date = datetime(1990, 1, 1)

        for i, (
            cmp_id,
            name,
            formula,
            mw,
            smiles,
            inchi,
            drug_class,
            moa,
            target,
            area,
            stage,
            approval,
            indications,
            dosage,
            route,
            half_life,
            bioavail,
            mfg,
            patent,
            first_approval,
            sc_datasets,
            preclin,
            trials,
            safety,
            interactions,
        ) in enumerate(sample_drugs):

            if patent:
                patent_date = (
                    datetime.strptime(patent, "%Y-%m-%d")
                    if isinstance(patent, str)
                    else patent
                )
            else:
                patent_date = None

            approval_date = (
                datetime.strptime(first_approval, "%Y-%m-%d")
                if isinstance(first_approval, str)
                else first_approval
            )

            sample_data.append(
                (
                    cmp_id,
                    name,
                    formula,
                    mw,
                    smiles,
                    inchi,
                    drug_class,
                    moa,
                    target,
                    area,
                    stage,
                    approval,
                    indications,
                    dosage,
                    route,
                    half_life,
                    bioavail,
                    mfg,
                    patent_date,
                    approval_date,
                    sc_datasets,
                    preclin,
                    trials,
                    safety,
                    interactions,
                )
            )

        # Generate additional random compounds
        for i in range(len(sample_drugs), min(num_records, 40)):
            cmp_id = f"CMP-{i+1:03d}"
            name = f"Compound-{i+1}"
            formula = f"C{random.randint(10,30)}H{random.randint(15,50)}N{random.randint(1,8)}O{random.randint(2,12)}"
            mw = random.uniform(200.0, 2000.0)
            drug_class = random.choice(drug_classes)
            area = random.choice(therapeutic_areas)
            stage = random.choice(dev_stages)
            approval = random.choice(approval_statuses)
            mfg = random.choice(manufacturers)

            half_life = random.uniform(1.0, 200.0)
            bioavail = random.uniform(10.0, 100.0)
            preclin = random.randint(5, 100)
            trials = random.randint(0, 50)

            # Link to some single cell datasets
            linked_datasets = (
                random.sample(sc_dataset_ids, k=min(3, len(sc_dataset_ids)))
                if sc_dataset_ids
                else []
            )
            sc_datasets = ",".join(linked_datasets)

            approval_date = base_date + timedelta(days=random.randint(0, 12000))
            patent_date = (
                approval_date + timedelta(days=random.randint(3650, 7300))
                if stage == "Approved"
                else None
            )

            sample_data.append(
                (
                    cmp_id,
                    name,
                    formula,
                    mw,
                    None,
                    None,
                    drug_class,
                    f"Target modulation",
                    f"Target protein {i}",
                    area,
                    stage,
                    approval,
                    f"Indication {i}",
                    "Tablet",
                    "Oral",
                    half_life,
                    bioavail,
                    mfg,
                    patent_date,
                    approval_date,
                    sc_datasets,
                    preclin,
                    trials,
                    "Standard safety profile",
                    "Standard drug interactions",
                )
            )

        drug_df = spark.createDataFrame(sample_data, drug_schema)
    else:
        drug_df = spark.createDataFrame([], drug_schema)

    # Write table
    table_name = f"{catalog_name}.{schema_name}.drug_compounds"
    drug_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
        table_name
    )

    # Add table comment
    spark.sql(
        f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'comment' = 'Comprehensive database of pharmaceutical compounds and drugs with chemical properties, clinical information, and links to single cell datasets where drug effects have been studied. Includes approved drugs, investigational compounds, and their associated research data.'
        )
    """
    )

    # Add column comments
    column_comments = {
        "compound_id": "Unique identifier for each drug compound",
        "compound_name": "Generic or brand name of the drug compound",
        "chemical_formula": "Molecular formula of the compound",
        "molecular_weight": "Molecular weight in daltons (Da)",
        "smiles_string": "Simplified Molecular Input Line Entry System notation for chemical structure",
        "inchi_key": "International Chemical Identifier key for unique compound identification",
        "drug_class": "Pharmacological class or category of the drug",
        "mechanism_of_action": "Primary biological mechanism by which the drug exerts its effects",
        "target_protein": "Primary molecular target(s) of the drug",
        "therapeutic_area": "Primary medical specialty or disease area",
        "development_stage": "Current stage of drug development: Preclinical, Phase I-III, Approved, Discontinued",
        "approval_status": "Regulatory approval status: FDA Approved, EMA Approved, Investigational, etc.",
        "approved_indications": "Medical conditions for which the drug is approved",
        "dosage_form": "Physical form of the drug: Tablet, Injectable, Capsule, etc.",
        "route_of_administration": "How the drug is administered: Oral, Intravenous, Subcutaneous, etc.",
        "half_life_hours": "Elimination half-life in hours",
        "bioavailability_percent": "Percentage of drug that reaches systemic circulation",
        "manufacturer": "Pharmaceutical company that manufactures the drug",
        "patent_expiry_date": "Date when primary patents expire",
        "first_approval_date": "Date of first regulatory approval",
        "associated_single_cell_datasets": "Comma-separated list of single cell dataset IDs where drug effects were studied",
        "preclinical_studies_count": "Number of preclinical studies conducted",
        "clinical_trials_count": "Number of clinical trials conducted or ongoing",
        "safety_profile": "Summary of known safety issues and adverse effects",
        "drug_interactions": "Notable drug-drug interactions and contraindications",
    }

    for column, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{comment}'")

    logger.info(f"Created drug_compounds table with {drug_df.count()} records")
    return drug_df


drug_df = create_drug_compounds_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Clinical Trials

# COMMAND ----------


def create_clinical_trials_table():
    """Create clinical trials table as additional example"""

    # Define schema
    trials_schema = StructType(
        [
            StructField("trial_id", StringType(), False),
            StructField("nct_number", StringType(), True),
            StructField("trial_title", StringType(), False),
            StructField("sponsor", StringType(), False),
            StructField("phase", StringType(), False),
            StructField("study_type", StringType(), False),
            StructField("intervention_type", StringType(), False),
            StructField("primary_indication", StringType(), False),
            StructField("target_enrollment", IntegerType(), True),
            StructField("actual_enrollment", IntegerType(), True),
            StructField("primary_endpoint", StringType(), True),
            StructField("secondary_endpoints", StringType(), True),
            StructField("inclusion_criteria", StringType(), True),
            StructField("study_start_date", TimestampType(), True),
            StructField("primary_completion_date", TimestampType(), True),
            StructField("study_completion_date", TimestampType(), True),
            StructField("trial_status", StringType(), False),
            StructField("study_locations", StringType(), True),
            StructField("principal_investigator", StringType(), True),
            StructField("biomarker_strategy", StringType(), True),
            StructField("genomic_profiling", StringType(), True),
            StructField("associated_compounds", StringType(), True),
            StructField("data_monitoring_committee", StringType(), True),
            StructField("interim_analysis_planned", StringType(), True),
            StructField("regulatory_pathway", StringType(), True),
        ]
    )

    if include_sample_data:
        # Sample clinical trials
        sample_trials = [
            (
                "CT-001",
                "NCT04567890",
                "Phase II Study of Pembrolizumab in Advanced Melanoma",
                "Merck & Co",
                "Phase II",
                "Interventional",
                "Drug",
                "Advanced Melanoma",
                150,
                142,
                "Overall Response Rate",
                "Progression-free survival, Overall survival",
                "Advanced melanoma, ECOG 0-1, adequate organ function",
                "Genomic profiling for tumor mutational burden",
                "PD-L1 expression, microsatellite instability",
                "CMP-002",
            ),
            (
                "CT-002",
                "NCT05123456",
                "Phase III Trial of Osimertinib vs Chemotherapy in EGFR+ NSCLC",
                "AstraZeneca",
                "Phase III",
                "Interventional",
                "Drug",
                "Non-Small Cell Lung Cancer",
                500,
                487,
                "Progression-free Survival",
                "Overall survival, Quality of life",
                "EGFR mutation positive, Stage IV NSCLC",
                "Next-generation sequencing for EGFR mutations",
                "Comprehensive genomic profiling, circulating tumor DNA",
                "CMP-006",
            ),
            (
                "CT-003",
                "NCT04789012",
                "Biomarker-Driven Precision Medicine Study in Breast Cancer",
                "National Cancer Institute",
                "Phase II",
                "Interventional",
                "Drug",
                "Breast Cancer",
                300,
                289,
                "Objective Response Rate",
                "Duration of response, Toxicity profile",
                "Metastatic breast cancer, specific biomarker positive",
                "Tumor sequencing for actionable mutations",
                "Whole exome sequencing, RNA sequencing",
                "CMP-001,CMP-004",
            ),
            (
                "CT-004",
                "NCT05234567",
                "Single Cell Analysis of Immune Response in COVID-19",
                "Stanford University",
                "Phase II",
                "Observational",
                "Procedure",
                "COVID-19",
                200,
                195,
                "Immune Cell Dynamics",
                "Cytokine profiles, Clinical outcomes",
                "COVID-19 patients, various severity levels",
                "Single cell RNA sequencing of immune cells",
                "scRNA-seq, proteomics, metabolomics",
                "CMP-005",
            ),
            (
                "CT-005",
                "NCT04678901",
                "Phase I Dose Escalation Study of Novel CAR-T Therapy",
                "University of Pennsylvania",
                "Phase I",
                "Interventional",
                "Biological",
                "B-cell Lymphoma",
                24,
                18,
                "Safety and Tolerability",
                "Efficacy, CAR-T expansion",
                "Relapsed/refractory B-cell lymphoma",
                "Single cell tracking of CAR-T cells",
                "scRNA-seq of CAR-T cells, flow cytometry",
                "None",
            ),
            (
                "CT-006",
                "NCT05345678",
                "Alzheimer's Disease Prevention Trial with Donepezil",
                "Mayo Clinic",
                "Phase III",
                "Interventional",
                "Drug",
                "Alzheimer's Disease",
                800,
                756,
                "Time to Clinical Dementia",
                "Cognitive decline rate, Brain imaging changes",
                "Mild cognitive impairment, amyloid positive",
                "Brain imaging biomarkers",
                "CSF biomarkers, PET imaging, cognitive assessments",
                "CMP-008",
            ),
        ]

        sample_data = []
        phases = ["Phase I", "Phase II", "Phase III", "Phase IV"]
        study_types = ["Interventional", "Observational", "Expanded Access"]
        intervention_types = ["Drug", "Biological", "Procedure", "Device", "Behavioral"]
        statuses = [
            "Active",
            "Completed",
            "Recruiting",
            "Suspended",
            "Terminated",
            "Withdrawn",
        ]
        sponsors = [
            "Pfizer",
            "Novartis",
            "Roche",
            "Merck & Co",
            "AstraZeneca",
            "Bristol Myers Squibb",
            "National Cancer Institute",
            "Stanford University",
        ]
        indications = [
            "Cancer",
            "Alzheimer's Disease",
            "Diabetes",
            "Cardiovascular Disease",
            "Autoimmune Disease",
            "Infectious Disease",
            "Rare Disease",
        ]

        base_date = datetime(2020, 1, 1)

        for i, (
            trial_id,
            nct,
            title,
            sponsor,
            phase,
            study_type,
            interv_type,
            indication,
            target_enroll,
            actual_enroll,
            primary_ep,
            secondary_ep,
            inclusion,
            genomic,
            biomarker,
            compounds,
        ) in enumerate(sample_trials):
            start_date = base_date + timedelta(days=random.randint(0, 1000))
            primary_comp = start_date + timedelta(
                days=random.randint(365, 1095)
            )  # 1-3 years
            study_comp = primary_comp + timedelta(
                days=random.randint(30, 365)
            )  # Additional follow-up

            locations = "Multi-center (US, EU)"
            pi = f"Dr. Principal Investigator {i+1}"
            dmc = "Yes" if phase in ["Phase II", "Phase III"] else "No"
            interim = "Yes" if target_enroll and target_enroll > 100 else "No"
            regulatory = "FDA IND" if study_type == "Interventional" else "IRB approval"

            sample_data.append(
                (
                    trial_id,
                    nct,
                    title,
                    sponsor,
                    phase,
                    study_type,
                    interv_type,
                    indication,
                    target_enroll,
                    actual_enroll,
                    primary_ep,
                    secondary_ep,
                    inclusion,
                    start_date,
                    primary_comp,
                    study_comp,
                    random.choice(statuses),
                    locations,
                    pi,
                    biomarker,
                    genomic,
                    compounds,
                    dmc,
                    interim,
                    regulatory,
                )
            )

        # Generate additional random trials
        for i in range(len(sample_trials), min(num_records, 25)):
            trial_id = f"CT-{i+1:03d}"
            nct = f"NCT{random.randint(10000000, 99999999):08d}"
            title = f"Clinical Trial {i+1} for {random.choice(indications)}"
            sponsor = random.choice(sponsors)
            phase = random.choice(phases)
            study_type = random.choice(study_types)
            interv_type = random.choice(intervention_types)
            indication = random.choice(indications)

            target_enroll = random.randint(20, 1000)
            actual_enroll = int(target_enroll * random.uniform(0.7, 1.0))

            primary_ep = "Primary efficacy endpoint"
            secondary_ep = "Safety, biomarkers"
            inclusion = "Standard inclusion criteria"

            start_date = base_date + timedelta(days=random.randint(0, 1200))
            primary_comp = start_date + timedelta(days=random.randint(365, 1095))
            study_comp = primary_comp + timedelta(days=random.randint(30, 365))

            status = random.choice(statuses)
            locations = "Multi-center"
            pi = f"Dr. PI {i+1}"
            biomarker = "Biomarker analysis planned"
            genomic = "Genomic profiling included"
            compounds = "CMP-001" if random.random() > 0.5 else None
            dmc = "Yes" if target_enroll > 100 else "No"
            interim = "Yes" if target_enroll > 200 else "No"
            regulatory = "FDA IND"

            sample_data.append(
                (
                    trial_id,
                    nct,
                    title,
                    sponsor,
                    phase,
                    study_type,
                    interv_type,
                    indication,
                    target_enroll,
                    actual_enroll,
                    primary_ep,
                    secondary_ep,
                    inclusion,
                    start_date,
                    primary_comp,
                    study_comp,
                    status,
                    locations,
                    pi,
                    biomarker,
                    genomic,
                    compounds,
                    dmc,
                    interim,
                    regulatory,
                )
            )

        trials_df = spark.createDataFrame(sample_data, trials_schema)
    else:
        trials_df = spark.createDataFrame([], trials_schema)

    # Write table
    table_name = f"{catalog_name}.{schema_name}.clinical_trials"
    trials_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
        table_name
    )

    # Add table comment
    spark.sql(
        f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'comment' = 'Clinical trials database containing information about interventional and observational studies in life sciences. Includes trial design, enrollment data, endpoints, biomarker strategies, and links to associated drug compounds and genomic profiling efforts.'
        )
    """
    )

    # Add column comments
    column_comments = {
        "trial_id": "Unique internal identifier for each clinical trial",
        "nct_number": "ClinicalTrials.gov NCT number for public identification",
        "trial_title": "Official title of the clinical trial",
        "sponsor": "Organization sponsoring and conducting the trial",
        "phase": "Clinical development phase: Phase I, II, III, or IV",
        "study_type": "Type of study: Interventional, Observational, or Expanded Access",
        "intervention_type": "Category of intervention: Drug, Biological, Procedure, Device, Behavioral",
        "primary_indication": "Primary disease or condition being studied",
        "target_enrollment": "Planned number of participants to enroll",
        "actual_enrollment": "Actual number of participants enrolled",
        "primary_endpoint": "Primary outcome measure for the study",
        "secondary_endpoints": "Secondary outcome measures and exploratory endpoints",
        "inclusion_criteria": "Key criteria for participant eligibility",
        "study_start_date": "Date when the study began enrolling participants",
        "primary_completion_date": "Date when primary endpoint data collection is completed",
        "study_completion_date": "Date when all study activities are completed",
        "trial_status": "Current status: Active, Completed, Recruiting, Suspended, etc.",
        "study_locations": "Geographic locations where the trial is conducted",
        "principal_investigator": "Lead investigator responsible for the trial",
        "biomarker_strategy": "Planned biomarker analysis and patient stratification approach",
        "genomic_profiling": "Genomic sequencing and molecular profiling components",
        "associated_compounds": "Drug compounds being tested (references drug_compounds table)",
        "data_monitoring_committee": "Whether an independent data monitoring committee oversees the trial",
        "interim_analysis_planned": "Whether interim efficacy or safety analyses are planned",
        "regulatory_pathway": "Regulatory approval pathway and designation status",
    }

    for column, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{comment}'")

    logger.info(f"Created clinical_trials table with {trials_df.count()} records")
    return trials_df


trials_df = create_clinical_trials_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("LIFE SCIENCES DEMO DATA GENERATION SUMMARY")
print("=" * 80)
print(f"Environment: {environment}")
print(f"Schema: {catalog_name}.{schema_name}")
print(f"Sample data included: {include_sample_data}")
print()
print("Tables created:")
print(f"1. customer_accounts - {accounts_df.count()} records")
print(f"2. products_orders - {orders_df.count()} records")
print(f"3. bulk_rnaseq_experiments - {rnaseq_df.count()} records")
print(f"4. single_cell_datasets - {sc_df.count()} records")
print(f"5. drug_compounds - {drug_df.count()} records")
print(f"6. clinical_trials - {trials_df.count()} records")
print()
print("Data relationships:")
print("- products_orders.account_id → customer_accounts.account_id")
print(
    "- drug_compounds.associated_single_cell_datasets → single_cell_datasets.dataset_id"
)
print("- clinical_trials.associated_compounds → drug_compounds.compound_id")
print()
print("These tables provide realistic life sciences data for testing:")
print("- Customer relationship management (accounts, orders)")
print("- Genomics research (bulk RNA-seq, single cell)")
print("- Drug discovery and development")
print("- Clinical research and trials")
print("=" * 80)

logger.info("Life sciences demo data generation completed successfully!")
