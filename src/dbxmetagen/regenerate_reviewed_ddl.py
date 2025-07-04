import pandas as pd
import re
from typing import Optional
from config import MetadataConfig

def load_metadata_file(file_path: str, config: MetadataConfig, file_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load metadata from a TSV or Excel file based on the file_type.

    Args:
        file_path (str): Path to the input file.
        file_type (Optional[str]): Type of the file, either 'tsv' or 'excel'. If None, uses config.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if file_type is None:
        file_type = config.review_input_file_type

    file_path = f"/Volumes/{config.catalog_name}.{config.schema_name}.generated_metadata/reviewed_outputs/"
    if not os.path.exists(file_path):
        os.mkdirs(file_path)       
    if file_type == 'tsv':
        return pd.read_csv(file_path, sep='\t', dtype=str)
    elif file_type == 'excel':
        return pd.read_excel(file_path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def replace_comment_in_ddl(ddl: str, new_comment: str) -> str:
    """
    Replace the comment string in a DDL statement with a new comment.

    Supports:
    - COMMENT ON TABLE ... IS "..."
    - ALTER TABLE ... ALTER COLUMN ... COMMENT "..."

    Args:
        ddl (str): The original DDL statement.
        new_comment (str): The new comment to insert.

    Returns:
        str: The DDL statement with the updated comment.
    """
    # Replace COMMENT ON TABLE ... IS "..."
    ddl = re.sub(
        r'(COMMENT ON TABLE [^"\']+ IS\s+)(["\'])(.*?)(["\'])',
        lambda m: f"{m.group(1)}{m.group(2)}{new_comment}{m.group(4)}",
        ddl
    )
    # Replace ALTER TABLE ... ALTER COLUMN ... COMMENT "..."
    ddl = re.sub(
        r'(ALTER TABLE [^"\']+ COMMENT ON COLUMN [^ ]+ IS\s+)(["\'])(.*?)(["\'])',
        lambda m: f"{m.group(1)}{m.group(2)}{new_comment}{m.group(4)}",
        ddl
    )
    return ddl

def replace_pii_tags_in_ddl(ddl: str, classification: str, subclassification: str) -> str:
    """
    Replace the PII tagging strings in a DDL statement with new classification and subclassification.

    Supports:
    - ALTER TABLE ... SET TAGS ('data_classification' = '...', 'data_subclassification' = '...');
    - ALTER TABLE ... ALTER COLUMN ... SET TAGS ('data_classification' = '...', 'data_subclassification' = '...');

    Args:
        ddl (str): The original DDL statement.
        classification (str): The new data classification.
        subclassification (str): The new data subclassification.

    Returns:
        str: The DDL statement with the updated tags.
    """
    # Replace ALTER TABLE ... SET TAGS ...
    ddl = re.sub(
        r"(ALTER TABLE [^ ]+ SET TAGS \('data_classification' = ')[^']+(', 'data_subclassification' = ')[^']+('\);)",
        lambda m: f"{m.group(1)}{classification}{m.group(2)}{subclassification}{m.group(4)}",
        ddl
    )
    # Replace ALTER TABLE ... ALTER COLUMN ... SET TAGS ...
    ddl = re.sub(
        r"(ALTER TABLE [^ ]+ ALTER COLUMN [^ ]+ SET TAGS \('data_classification' = ')[^']+(', 'data_subclassification' = ')[^']+('\);)",
        lambda m: f"{m.group(1)}{classification}{m.group(2)}{subclassification}{m.group(4)}",
        ddl
    )
    return ddl

def process_metadata_file(
    file_path: str,
    config: MetadataConfig, 
    updated_file_path: Optional[str] = None,
    sql_output_path: str = "output.sql",
    file_type: Optional[str] = None
) -> None:
    """
    Load a metadata file (TSV or Excel), update DDL comments and PII tags using the relevant columns,
    and export a .sql file with updated DDL statements.

    Args:
        file_path (str): Path to the input file.
        updated_file_path (Optional[str]): If provided, save the updated file here.
        sql_output_path (str): Path to the output .sql file.
        file_type (Optional[str]): Type of the file, either 'tsv' or 'excel'. If None, uses config.
    """
    df = load_metadata_file(file_path, config, file_type)

    def update_ddl(row):
        if pd.isna(row['ddl']):
            return row['ddl']
        # Update PII tagging DDLs if classification and type are present
        if 'classification' in row and 'type' in row and pd.notna(row['classification']) and pd.notna(row['type']):
            return replace_pii_tags_in_ddl(row['ddl'], row['classification'], row['type'])
        # Otherwise, update comment DDLs if column_content is present
        if 'column_content' in row and pd.notna(row['column_content']):
            return replace_comment_in_ddl(row['ddl'], row['column_content'])
        return row['ddl']

    df['ddl'] = df.apply(update_ddl, axis=1)

    # Optionally save the updated file in the correct format
    if updated_file_path:
        if file_type == 'excel' or (file_type is None and config.review_input_file_type == 'excel'):
            df.to_excel(updated_file_path, index=False)
        else:
            df.to_csv(updated_file_path, sep='\t', index=False)

    # Export only the DDL statements to a .sql file
    with open(sql_output_path, 'w', encoding='utf-8') as f:
        for ddl in df['ddl'].dropna():
            f.write(ddl.strip())
            if not ddl.strip().endswith(';'):
                f.write(';')
            f.write('\n')
