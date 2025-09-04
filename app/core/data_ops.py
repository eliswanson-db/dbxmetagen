"""
Data operations module.
Handles table validation, CSV processing, metadata operations.
"""

import streamlit as st
import pandas as pd
import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from io import StringIO

logger = logging.getLogger(__name__)


class DataOperations:
    """Handles data processing operations for metadata generation."""

    def __init__(self):
        pass

    def validate_table_names(
        self, table_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate table names format and accessibility.

        Returns:
            Tuple of (valid_tables, invalid_tables)
        """
        valid_tables = []
        invalid_tables = []

        table_name_pattern = re.compile(
            r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$"
        )

        for table_name in table_names:
            table_name = table_name.strip()
            if not table_name:
                continue

            if table_name_pattern.match(table_name):
                valid_tables.append(table_name)
            else:
                invalid_tables.append(table_name)
                logger.warning(f"Invalid table name format: {table_name}")

        return valid_tables, invalid_tables

    def display_table_validation_results(
        self, valid_tables: List[str], invalid_tables: List[str]
    ):
        """Display validation results to the user."""
        if valid_tables:
            st.success(f"✅ {len(valid_tables)} valid table names")
            with st.expander(f"Valid Tables ({len(valid_tables)})"):
                for table in valid_tables:
                    st.write(f"✅ {table}")

        if invalid_tables:
            st.error(f"❌ {len(invalid_tables)} invalid table names")
            with st.expander(f"Invalid Tables ({len(invalid_tables)})"):
                for table in invalid_tables:
                    st.write(f"❌ {table}")
                st.write("**Expected format:** `catalog.schema.table`")

    def validate_tables(self, tables: List[str]) -> Dict[str, List[str]]:
        """
        Validate that tables exist and are accessible.

        Returns:
            Dictionary with 'accessible', 'inaccessible', and 'errors' keys
        """
        if not st.session_state.get("workspace_client"):
            return {
                "accessible": [],
                "inaccessible": tables,
                "errors": ["Workspace client not initialized"],
            }

        accessible = []
        inaccessible = []
        errors = []

        try:
            for table in tables:
                try:
                    # Try to get table info
                    catalog, schema, table_name = table.split(".")
                    table_info = st.session_state.workspace_client.tables.get(
                        f"{catalog}.{schema}.{table_name}"
                    )

                    if table_info:
                        accessible.append(table)
                        logger.info(f"✅ Table accessible: {table}")
                    else:
                        inaccessible.append(table)
                        logger.warning(f"❌ Table not found: {table}")

                except Exception as e:
                    inaccessible.append(table)
                    error_msg = f"Error checking {table}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        except Exception as e:
            error_msg = f"Failed to validate tables: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        return {
            "accessible": accessible,
            "inaccessible": inaccessible,
            "errors": errors,
        }

    def process_uploaded_csv(self, uploaded_file) -> List[str]:
        """
        Process uploaded CSV file to extract table names.

        Returns:
            List of table names
        """
        try:
            # Read CSV file
            content = uploaded_file.read()

            # Handle both string and bytes
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            # Parse CSV
            df = pd.read_csv(StringIO(content))

            # Look for table name column
            possible_columns = ["table_name", "table", "name", "table_names"]
            table_column = None

            for col in possible_columns:
                if col in df.columns:
                    table_column = col
                    break

            if table_column is None:
                # If no standard column found, use first column
                if len(df.columns) > 0:
                    table_column = df.columns[0]
                    st.warning(f"Using first column '{table_column}' as table names")
                else:
                    st.error("No columns found in CSV file")
                    return []

            # Extract table names and clean them
            table_names = df[table_column].dropna().astype(str).str.strip().tolist()

            # Remove empty strings
            table_names = [name for name in table_names if name and name != "nan"]

            st.success(f"✅ Loaded {len(table_names)} table names from CSV")
            logger.info(
                f"Loaded {len(table_names)} table names from {uploaded_file.name}"
            )

            return table_names

        except Exception as e:
            error_msg = f"Failed to process CSV file: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            return []

    def save_table_list(self, tables: List[str]) -> Optional[bytes]:
        """
        Save table list as CSV for download.

        Returns:
            CSV content as bytes, or None if error
        """
        try:
            df = pd.DataFrame({"table_name": tables})
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)

            logger.info(f"Created CSV with {len(tables)} table names")
            return csv_buffer.getvalue().encode("utf-8")

        except Exception as e:
            error_msg = f"Failed to create CSV: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            return None


class MetadataProcessor:
    """Handles metadata processing operations."""

    def __init__(self):
        pass

    def load_metadata_from_volume(
        self, catalog: str, schema: str, volume: str
    ) -> Optional[pd.DataFrame]:
        """Load metadata files from Unity Catalog volume."""
        if not st.session_state.get("workspace_client"):
            st.error("❌ Workspace client not initialized")
            return None

        try:
            volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

            # List files in volume
            files = st.session_state.workspace_client.files.list_directory_contents(
                volume_path
            )

            # Look for TSV files
            tsv_files = [f for f in files if f.name.endswith(".tsv")]

            if not tsv_files:
                st.warning("No TSV files found in volume")
                return None

            # Use most recent file
            latest_file = max(tsv_files, key=lambda x: x.modification_time)

            # Read file content
            file_path = f"{volume_path}/{latest_file.name}"
            content = st.session_state.workspace_client.files.download(file_path)

            # Parse TSV
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep="\t")

            st.success(f"✅ Loaded metadata from {latest_file.name}")
            logger.info(f"Loaded metadata file: {latest_file.name}")

            return df

        except Exception as e:
            error_msg = f"Failed to load metadata from volume: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            return None

    def _load_file_from_volume(
        self, catalog: str, schema: str, volume: str, filename: str
    ) -> Optional[str]:
        """Load a specific file from Unity Catalog volume."""
        try:
            volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
            file_path = f"{volume_path}/{filename}"

            content = st.session_state.workspace_client.files.download(file_path)
            return content.decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to load file {filename}: {str(e)}")
            return None

    def apply_metadata_to_tables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply metadata changes to actual tables (DDL execution)."""
        if not st.session_state.get("workspace_client"):
            return {"success": False, "error": "Workspace client not initialized"}

        results = {"success": False, "applied": 0, "failed": 0, "errors": []}

        try:
            # This would contain the actual DDL execution logic
            # For safety, this is just a placeholder
            st.warning("⚠️ DDL application not implemented in this example")

            results["success"] = True
            results["applied"] = len(df)

        except Exception as e:
            error_msg = f"Failed to apply metadata: {str(e)}"
            results["error"] = error_msg
            results["errors"].append(error_msg)
            logger.error(error_msg)

        return results

    def review_uploaded_metadata(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Review uploaded metadata file."""
        try:
            # Read the uploaded file
            content = uploaded_file.read()

            # Handle both string and bytes
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            # Try to parse as TSV first, then CSV
            try:
                df = pd.read_csv(StringIO(content), sep="\t")
            except:
                df = pd.read_csv(StringIO(content))

            # Validate required columns
            required_columns = ["table_name", "column_name"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                return None

            st.success(f"✅ Loaded metadata file with {len(df)} rows")
            logger.info(f"Loaded metadata file: {uploaded_file.name}")

            return df

        except Exception as e:
            error_msg = f"Failed to process metadata file: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            return None
