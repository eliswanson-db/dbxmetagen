"""Parsing utilities shared among modules.
"""

def sanitize_email(email: str) -> str:
    """
    Replaces '@' and '.' in an email address with '_'.

    Args:
        email (str): The email address to sanitize.

    Returns:
        str: The sanitized email address.
    """
    return email.replace('@', '_').replace('.', '_')