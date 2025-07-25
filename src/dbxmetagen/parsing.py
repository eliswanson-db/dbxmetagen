"""Parsing and cleansing utilities shared among modules.
"""

import re
from typing import Optional

def sanitize_email(email: str) -> str:
    """
    Replaces '@' and '.' in an email address with '_'.

    Args:
        email (str): The email address to sanitize.

    Returns:
        str: The sanitized email address.
    """
    return email.replace('@', '_').replace('.', '_')


def cleanse_sql_comment(comment: str) -> str:
    """
    Cleanse a SQL comment string to make it compatible with DB SQL.

    - Replaces double double-quotes ("") and single double-quotes (") with a single quote (').
    - Escapes single quotes (') by doubling them ('')
    - Leaves standard double quotes as-is (unless you need to escape them for your SQL dialect)
    
    Args:
        comment (str): The original comment string.
    Returns:
        str: The cleansed comment string.
    """
    if comment is None:
        return comment

    #comment = re.sub(r"(?<!')'(?!')", "''", comment)
    comment = comment.replace('""', "'")
    comment = comment.replace('"', "'")

    return comment
