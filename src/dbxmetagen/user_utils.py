"""
User utility functions for dbxmetagen.

This module contains utility functions for handling user identifiers,
separated to avoid circular imports.
"""


def sanitize_user_identifier(identifier: str) -> str:
    """
    Sanitizes user identifier - handles both emails and service principal names.

    For emails (contains @): removes domain and replaces special chars with underscores
    For service principals (alphanumeric + hyphens): replaces hyphens with underscores

    Args:
        identifier (str): Email address or service principal name to sanitize

    Returns:
        str: Sanitized identifier safe for file/table names

    Examples:
        sanitize_user_identifier("user@databricks.com") -> "user"
        sanitize_user_identifier("034f50f1-0a51-4d3d-9137-ca312e31fc23") -> "034f50f1_0a51_4d3d_9137_ca312e31fc23"
    """
    if "@" in identifier:
        # Email format - extract username part and sanitize
        username = identifier.split("@")[0]
        return username.replace(".", "_").replace("-", "_")
    else:
        # Service principal format - replace hyphens with underscores
        return identifier.replace("-", "_")


def sanitize_email(email: str) -> str:
    """
    DEPRECATED: Use sanitize_user_identifier instead.
    Backward compatibility wrapper for sanitize_user_identifier.
    """
    return sanitize_user_identifier(email)
