"""
Data operations module.
Handles table validation, CSV processing, metadata operations.
"""

from datetime import datetime
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
        """Load metadata files from Unity Catalog volume - simplified version."""
        if not st.session_state.get("workspace_client"):
            st.error("❌ Workspace client not initialized")
            return None

        try:
            # Build path
            current_user = st.session_state.workspace_client.current_user.me().user_name
            sanitized_user = (
                current_user.replace("@", "_").replace(".", "_").replace("-", "_")
            )
            current_date = str(datetime.now().strftime("%Y%m%d"))
            full_directory_path = f"/Volumes/{catalog}/{schema}/{volume}/{sanitized_user}/{current_date}/exportable_run_logs/"

            st.info(f"🔍 Looking for files in: {full_directory_path}")

            # List files
            files = list(
                st.session_state.workspace_client.files.list_directory_contents(
                    full_directory_path
                )
            )
            st.info(f"📂 Found {len(files)} files in directory")

            # Find TSV files
            tsv_files = [f for f in files if f.name.endswith(".tsv")]
            if not tsv_files:
                st.warning(f"No TSV files found in {full_directory_path}")
                return None

            # Use first TSV file (simplest approach)
            latest_file = tsv_files[0]
            st.info(f"📄 Loading file: {latest_file.name}")

            # Download and read file
            file_path = f"{full_directory_path}{latest_file.name}"
            raw_content = st.session_state.workspace_client.files.download(file_path)

            # Extract content - try different methods since DownloadResponse varies
            content = None
            if hasattr(raw_content, "read"):
                # If it has a read method, use it directly
                content_bytes = raw_content.read()
                content = (
                    content_bytes.decode("utf-8")
                    if isinstance(content_bytes, bytes)
                    else str(content_bytes)
                )
                st.info("✅ Used read() method")
            elif hasattr(raw_content, "contents"):
                # If it has contents attribute, extract from there
                actual_content = raw_content.contents
                if hasattr(actual_content, "read"):
                    content_bytes = actual_content.read()
                    content = (
                        content_bytes.decode("utf-8")
                        if isinstance(content_bytes, bytes)
                        else str(content_bytes)
                    )
                    st.info("✅ Used contents.read() method")
                else:
                    content = str(actual_content)
                    st.info("✅ Used str(contents)")
            else:
                # Last resort - try context manager or convert to string
                try:
                    with raw_content as stream:
                        content = stream.read().decode("utf-8")
                    st.info("✅ Used context manager")
                except Exception:
                    content = str(raw_content)
                    st.info("⚠️ Fallback to string conversion")

            if not content:
                raise Exception("Failed to extract content from DownloadResponse")

            st.info(f"✅ Successfully read {len(content)} characters")

            # Parse TSV
            df = pd.read_csv(StringIO(content), sep="\t")
            st.info(
                f"🔍 Loaded DataFrame: {df.shape} shape, columns: {list(df.columns)}"
            )

            if len(df) == 0:
                st.warning("⚠️ DataFrame is empty!")
                return None

            st.success(f"✅ Loaded {len(df)} records from {latest_file.name}")
            return df

        except Exception as e:
            st.error(f"❌ Error loading metadata: {str(e)}")
            logger.error(f"Error in load_metadata_from_volume: {str(e)}")

            # Show debug info for directories that don't exist
            try:
                # Try parent directories to help debug
                path_parts = full_directory_path.rstrip("/").split("/")
                for i in range(len(path_parts) - 1, 2, -1):
                    parent_dir = "/".join(path_parts[: i + 1])
                    try:
                        parent_files = list(
                            st.session_state.workspace_client.files.list_directory_contents(
                                parent_dir
                            )
                        )
                        st.info(
                            f"✅ Found parent directory: {parent_dir} with {len(parent_files)} items"
                        )
                        break
                    except:
                        continue
            except:
                pass

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

    def apply_metadata_to_tables(
        self, df: pd.DataFrame, job_manager=None
    ) -> Dict[str, Any]:
        """Generate DDL from comments and trigger Databricks job to execute it."""
        if not st.session_state.get("workspace_client"):
            return {"success": False, "error": "Workspace client not initialized"}

        if not job_manager:
            return {"success": False, "error": "Job manager not provided"}

        results = {"success": False, "applied": 0, "failed": 0, "errors": []}

        try:
            # Debug: Show DataFrame structure
            st.info(f"🔍 DataFrame columns: {list(df.columns)}")
            st.info(f"🔍 DataFrame shape: {df.shape}")

            # Generate DDL from edited comments first
            st.info("🔄 Generating DDL from edited comments...")
            updated_df = self._generate_ddl_from_comments(df)

            # Save the updated DataFrame to Unity Catalog volume first
            saved_filename = self._save_updated_metadata_for_job(updated_df)
            if not saved_filename:
                results["error"] = "Failed to save metadata file for job execution"
                return results

            # Trigger Databricks job to execute DDL using sync_reviewed_ddl.py notebook
            st.info("🚀 Triggering Databricks job to execute DDL statements...")
            job_result = self._trigger_ddl_sync_job(saved_filename, job_manager)

            if job_result and job_result.get("success"):
                results["success"] = True
                results["applied"] = len(
                    [
                        row
                        for _, row in updated_df.iterrows()
                        if row.get("ddl")
                        and pd.notna(row.get("ddl"))
                        and str(row.get("ddl")).strip()
                    ]
                )

                # Update session state with the updated DataFrame
                st.session_state.review_metadata = updated_df

                st.success("🎉 Successfully triggered DDL execution job!")
                st.info(f"📋 Job ID: {job_result.get('job_id')}")
                if job_result.get("run_id"):
                    st.info(f"🔄 Run ID: {job_result.get('run_id')}")
            else:
                error_msg = job_result.get("error", "Unknown error triggering job")
                results["error"] = error_msg
                results["errors"].append(error_msg)
                st.error(f"❌ Failed to trigger DDL execution job: {error_msg}")

        except Exception as e:
            error_msg = f"Failed to apply metadata: {str(e)}"
            results["error"] = error_msg
            results["errors"].append(error_msg)
            logger.error(error_msg)

        return results

    def _generate_ddl_from_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DDL statements from Description and PII Classification fields."""
        updated_df = df.copy()

        # Determine column names (handle different possible column name variations)
        table_col = "table_name" if "table_name" in df.columns else "table"
        column_col = "column_name" if "column_name" in df.columns else "column"
        desc_cols = [
            col
            for col in df.columns
            if col.lower() in ["description", "column_content"]
        ]
        pii_cols = [col for col in df.columns if "pii" in col.lower()]

        st.info(
            f"🔍 Generating DDL from comments - table: {table_col}, column: {column_col}, description: {desc_cols}, pii: {pii_cols}"
        )

        for index, row in updated_df.iterrows():
            table_name = row[table_col]
            column_name = row[column_col]

            # Check if this is a table-level comment (column_name is null/nan)
            is_table_comment = pd.isna(column_name) or str(
                column_name
            ).lower().strip() in ["nan", "null", ""]

            # Build comment content from available editable columns
            comment_parts = []

            # Apply Description if it exists and is not empty
            for desc_col in desc_cols:
                if (
                    desc_col in row
                    and pd.notna(row[desc_col])
                    and str(row[desc_col]).strip()
                ):
                    comment_parts.append(str(row[desc_col]).strip())

            # Add PII classification if available and not empty
            for pii_col in pii_cols:
                if (
                    pii_col in row
                    and pd.notna(row[pii_col])
                    and str(row[pii_col]).strip()
                ):
                    pii_value = str(row[pii_col]).strip()
                    comment_parts.append(f"PII: {pii_value}")

            # Create the combined comment and generate DDL
            if comment_parts:
                combined_comment = " | ".join(comment_parts)
                escaped_comment = combined_comment.replace(
                    "'", "''"
                )  # Escape single quotes

                if is_table_comment:
                    # Table-level comment DDL
                    ddl_statement = (
                        f"ALTER TABLE {table_name} COMMENT '{escaped_comment}'"
                    )
                    st.info(f"🔍 Generated TABLE DDL for {table_name}: {ddl_statement}")
                else:
                    # Column-level comment DDL
                    ddl_statement = f"ALTER TABLE {table_name} ALTER COLUMN `{column_name}` COMMENT '{escaped_comment}'"
                    st.info(
                        f"🔍 Generated COLUMN DDL for {table_name}.{column_name}: {ddl_statement}"
                    )

                updated_df.at[index, "ddl"] = ddl_statement
            else:
                target = (
                    table_name if is_table_comment else f"{table_name}.{column_name}"
                )
                st.info(f"⚠️ No comment content for {target} - skipping DDL generation")

        return updated_df

    def _save_updated_metadata_for_job(self, df: pd.DataFrame) -> str:
        """Save updated metadata to Unity Catalog volume for job execution."""
        try:
            if not st.session_state.get("review_metadata_original_path"):
                st.error(
                    "❌ Original file path not found. Cannot save for job execution."
                )
                return None

            path_info = st.session_state.review_metadata_original_path

            # Get current user and date for path construction
            current_user = st.session_state.workspace_client.current_user.me().user_name
            sanitized_user = (
                current_user.replace("@", "_").replace(".", "_").replace("-", "_")
            )
            current_date = datetime.now().strftime("%Y%m%d")

            # Construct the output path
            volume_path = f"/Volumes/{path_info['catalog']}/{path_info['schema']}/{path_info['volume']}"
            output_dir = (
                f"{volume_path}/{sanitized_user}/{current_date}/exportable_run_logs/"
            )

            # Generate unique filename for job execution
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reviewed_metadata_for_job_{timestamp}.tsv"
            output_path = f"{output_dir}{filename}"

            # Save as TSV format
            st.info(f"💾 Saving metadata for job execution: {output_path}")
            tsv_content = df.to_csv(sep="\t", index=False)

            # Use WorkspaceClient to save the file
            try:
                tsv_bytes = tsv_content.encode("utf-8")
                st.session_state.workspace_client.files.upload(
                    output_path, tsv_bytes, overwrite=True
                )
                st.success(f"✅ Saved metadata file: {filename}")
                return filename
            except Exception as upload_error:
                st.error(f"❌ Failed to save file: {upload_error}")
                return None

        except Exception as e:
            st.error(f"❌ Error saving metadata for job: {str(e)}")
            logger.error(f"Error in _save_updated_metadata_for_job: {str(e)}")
            return None

    def _trigger_ddl_sync_job(self, filename: str, job_manager) -> Dict[str, Any]:
        """Trigger a Databricks job to execute DDL using sync_reviewed_ddl.py notebook."""
        try:
            # Prepare job parameters for the sync_reviewed_ddl.py notebook
            job_params = {
                "reviewed_file_name": filename,
                "mode": "comment",  # Default mode, could be made configurable
            }

            st.info(f"🔧 Job parameters: {job_params}")

            # Create and run the DDL sync job
            try:
                job_id, run_id = job_manager.create_and_run_sync_job(
                    filename=filename, mode="comment"
                )

                return {
                    "success": True,
                    "job_id": job_id,
                    "run_id": run_id,
                    "message": f"DDL sync job triggered successfully",
                }

            except AttributeError:
                # Fallback: use generic job creation if sync-specific method doesn't exist
                return {
                    "success": False,
                    "error": "DDL sync job creation not implemented in job manager",
                }

        except Exception as e:
            error_msg = f"Failed to trigger DDL sync job: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

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
