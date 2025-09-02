"""Test the sanitize_user_identifier function

This test can be run with:
- pytest tests/unit_tests/test_user_sanitizer.py -v (if pytest is installed)
- python -m pytest tests/unit_tests/test_user_sanitizer.py -v

Or individually by importing and calling test functions directly.
"""

from src.dbxmetagen.user_utils import sanitize_user_identifier


def test_sanitize_user_identifier_emails():
    """Test sanitize_user_identifier with email formats"""
    test_cases = [
        ("eli.swanson@databricks.com", "eli_swanson"),
        ("user@company.com", "user"),
        ("user.name@example.org", "user_name"),
        ("test-user@domain.co.uk", "test_user"),
    ]

    for input_val, expected in test_cases:
        result = sanitize_user_identifier(input_val)
        assert (
            result == expected
        ), f"Failed for '{input_val}': got '{result}', expected '{expected}'"


def test_sanitize_user_identifier_service_principals():
    """Test sanitize_user_identifier with service principal formats"""
    test_cases = [
        (
            "034f50f1-0a51-4d3d-9137-ca312e31fc23",
            "034f50f1_0a51_4d3d_9137_ca312e31fc23",
        ),
        ("test-sp-123", "test_sp_123"),
        ("abcd-1234-5678", "abcd_1234_5678"),
        ("simple-name", "simple_name"),
    ]

    for input_val, expected in test_cases:
        result = sanitize_user_identifier(input_val)
        assert (
            result == expected
        ), f"Failed for '{input_val}': got '{result}', expected '{expected}'"


def test_sanitize_user_identifier_specific_service_principal():
    """Test the specific service principal that was causing SQL issues"""
    sp_guid = "034f50f1-0a51-4d3d-9137-ca312e31fc23"
    result = sanitize_user_identifier(sp_guid)
    expected = "034f50f1_0a51_4d3d_9137_ca312e31fc23"

    assert result == expected
    # Ensure the problematic "9137" is still there but as part of underscores
    assert "9137" in result
    # Ensure no hyphens remain (those cause SQL issues)
    assert "-" not in result
    # Ensure it's a valid SQL identifier (alphanumeric + underscores)
    assert result.replace("_", "").isalnum()


def test_sanitize_user_identifier_edge_cases():
    """Test edge cases for sanitize_user_identifier"""
    test_cases = [
        ("user", "user"),  # No special characters
        ("123-456", "123_456"),  # Numbers with hyphens
        ("a@b", "a"),  # Minimal email
        (
            "user.with.many.dots@domain.com",
            "user_with_many_dots",
        ),  # Multiple dots in email
    ]

    for input_val, expected in test_cases:
        result = sanitize_user_identifier(input_val)
        assert (
            result == expected
        ), f"Failed for '{input_val}': got '{result}', expected '{expected}'"


def test_backward_compatibility_sanitize_email():
    """Test that the old sanitize_email function still works via the new function"""
    from src.dbxmetagen.processing import sanitize_email

    # Test that sanitize_email now uses the new logic
    assert sanitize_email("user@domain.com") == "user"
    assert (
        sanitize_email("034f50f1-0a51-4d3d-9137-ca312e31fc23")
        == "034f50f1_0a51_4d3d_9137_ca312e31fc23"
    )
