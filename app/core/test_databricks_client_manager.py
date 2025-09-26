# """
# Unit tests for DatabricksClientManager.

# These tests demonstrate the improved testability of the refactored methods.
# Run with: python -m pytest test_databricks_client_manager.py -v
# """

# import pytest
# import os
# from unittest.mock import patch, MagicMock, Mock
# from config import DatabricksClientManager


# class TestDatabricksClientManager:
#     """Unit tests for DatabricksClientManager methods."""

#     def test_is_valid_host_url_valid_cases(self):
#         """Test URL validation with valid URLs."""
#         valid_urls = [
#             "https://example.databricks.com",
#             "http://localhost.databricks.com",
#             "https://workspace-123.cloud.databricks.com",
#             "https://my-workspace.databricks.com/path",
#         ]

#         for url in valid_urls:
#             assert DatabricksClientManager._is_valid_host_url(
#                 url
#             ), f"Should be valid: {url}"

#     def test_is_valid_host_url_invalid_cases(self):
#         """Test URL validation with invalid URLs."""
#         invalid_urls = [
#             "invalid",
#             "workspace.com",  # No protocol
#             "https://",  # Too short
#             "https://a",  # Too short, no dot
#             "",  # Empty
#             "ftp://example.com",  # Wrong protocol
#         ]

#         for url in invalid_urls:
#             assert not DatabricksClientManager._is_valid_host_url(
#                 url
#             ), f"Should be invalid: {url}"

#     @patch("os.environ.get")
#     def test_validate_environment_missing_host(self, mock_env_get):
#         """Test environment validation when DATABRICKS_HOST is missing."""
#         mock_env_get.return_value = None

#         with patch("streamlit.error") as mock_error:
#             result = DatabricksClientManager._validate_environment()

#             assert result is None
#             mock_error.assert_called_once()
#             assert "DATABRICKS_HOST" in mock_error.call_args[0][0]

#     @patch("os.environ.get")
#     def test_validate_environment_invalid_host(self, mock_env_get):
#         """Test environment validation with invalid host URL."""
#         mock_env_get.return_value = "invalid-url"

#         with patch("streamlit.error") as mock_error:
#             result = DatabricksClientManager._validate_environment()

#             assert result is None
#             mock_error.assert_called_once()
#             assert "Invalid host URL" in mock_error.call_args[0][0]

#     @patch("os.environ.get")
#     def test_validate_environment_success(self, mock_env_get):
#         """Test successful environment validation."""
#         test_host = "https://example.databricks.com"
#         mock_env_get.return_value = test_host

#         with patch("streamlit.error") as mock_error:
#             result = DatabricksClientManager._validate_environment()

#             assert result == test_host
#             mock_error.assert_not_called()

#     def test_try_environment_token_found(self):
#         """Test token extraction from environment variables."""
#         test_token = "test-token-123"

#         with patch.dict(os.environ, {"DATABRICKS_TOKEN": test_token}):
#             result = DatabricksClientManager._try_environment_token()
#             assert result == test_token

#     def test_try_environment_token_alternative(self):
#         """Test token extraction from alternative environment variable."""
#         test_token = "test-access-token-456"

#         with patch.dict(os.environ, {"DATABRICKS_ACCESS_TOKEN": test_token}):
#             result = DatabricksClientManager._try_environment_token()
#             assert result == test_token

#     def test_try_environment_token_not_found(self):
#         """Test token extraction when no environment token exists."""
#         # Clear environment variables
#         with patch.dict(os.environ, {}, clear=True):
#             result = DatabricksClientManager._try_environment_token()
#             assert result is None

#     @patch("streamlit.query_params")
#     def test_try_query_params_token_found(self, mock_query_params):
#         """Test token extraction from query parameters."""
#         test_token = "query-token-789"
#         mock_query_params.get.side_effect = lambda x: (
#             test_token if x == "token" else None
#         )

#         result = DatabricksClientManager._try_query_params_token()
#         assert result == test_token

#     @patch("streamlit.query_params")
#     def test_try_query_params_token_alternative(self, mock_query_params):
#         """Test token extraction from alternative query parameter."""
#         test_token = "access-token-101"
#         mock_query_params.get.side_effect = lambda x: (
#             test_token if x == "access_token" else None
#         )

#         result = DatabricksClientManager._try_query_params_token()
#         assert result == test_token

#     @patch("streamlit.query_params")
#     def test_try_query_params_token_exception(self, mock_query_params):
#         """Test token extraction when query params cause exception."""
#         mock_query_params.get.side_effect = Exception("Query params error")

#         result = DatabricksClientManager._try_query_params_token()
#         assert result is None

#     def test_try_streamlit_context_token_no_context(self):
#         """Test token extraction when no Streamlit context available."""
#         with patch(
#             "streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=None
#         ):
#             result = DatabricksClientManager._try_streamlit_context_token()
#             assert result is None

#     def test_try_streamlit_context_token_with_headers(self):
#         """Test token extraction from Streamlit context headers."""
#         test_token = "header-token-202"

#         # Mock the context chain
#         mock_headers = {"x-forwarded-access-token": test_token}
#         mock_session_info = Mock()
#         mock_session_info.headers = mock_headers
#         mock_ctx = Mock()
#         mock_ctx.session_info = mock_session_info

#         with patch(
#             "streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock_ctx
#         ):
#             result = DatabricksClientManager._try_streamlit_context_token()
#             assert result == test_token

#     def test_try_streamlit_context_token_with_bearer(self):
#         """Test token extraction from Authorization header."""
#         test_token = "bearer-token-303"

#         mock_headers = {"authorization": f"Bearer {test_token}"}
#         mock_session_info = Mock()
#         mock_session_info.headers = mock_headers
#         mock_ctx = Mock()
#         mock_ctx.session_info = mock_session_info

#         with patch(
#             "streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock_ctx
#         ):
#             result = DatabricksClientManager._try_streamlit_context_token()
#             assert result == test_token

#     @patch.object(DatabricksClientManager, "_try_streamlit_context_token")
#     @patch.object(DatabricksClientManager, "_try_query_params_token")
#     @patch.object(DatabricksClientManager, "_try_environment_token")
#     def test_extract_user_token_priority(self, mock_env, mock_query, mock_streamlit):
#         """Test token extraction follows correct priority order."""
#         # Setup mocks - Streamlit should be tried first
#         mock_streamlit.return_value = "streamlit-token"
#         mock_query.return_value = "query-token"
#         mock_env.return_value = "env-token"

#         result = DatabricksClientManager._extract_user_token()

#         # Should return the first successful token (Streamlit)
#         assert result == "streamlit-token"

#         # Streamlit should be called, but others might not be due to early return
#         mock_streamlit.assert_called_once()

#     @patch.object(DatabricksClientManager, "_try_streamlit_context_token")
#     @patch.object(DatabricksClientManager, "_try_query_params_token")
#     @patch.object(DatabricksClientManager, "_try_environment_token")
#     def test_extract_user_token_fallback(self, mock_env, mock_query, mock_streamlit):
#         """Test token extraction fallback when earlier methods fail."""
#         # Setup mocks - only environment token available
#         mock_streamlit.return_value = None
#         mock_query.return_value = None
#         mock_env.return_value = "env-token-only"

#         result = DatabricksClientManager._extract_user_token()

#         # Should return the environment token
#         assert result == "env-token-only"

#         # All methods should have been tried
#         mock_streamlit.assert_called_once()
#         mock_query.assert_called_once()
#         mock_env.assert_called_once()

#     @patch.object(DatabricksClientManager, "_try_streamlit_context_token")
#     @patch.object(DatabricksClientManager, "_try_query_params_token")
#     @patch.object(DatabricksClientManager, "_try_environment_token")
#     def test_extract_user_token_none_found(self, mock_env, mock_query, mock_streamlit):
#         """Test token extraction when no tokens are found."""
#         mock_streamlit.return_value = None
#         mock_query.return_value = None
#         mock_env.return_value = None

#         result = DatabricksClientManager._extract_user_token()

#         assert result is None

#     def test_create_authenticated_client_with_token(self):
#         """Test client creation with user token."""
#         test_host = "https://example.databricks.com"
#         test_token = "test-token"

#         with patch.object(
#             DatabricksClientManager, "_extract_user_token", return_value=test_token
#         ):
#             with patch("config.WorkspaceClient") as mock_client_class:
#                 mock_client_instance = Mock()
#                 mock_client_class.return_value = mock_client_instance

#                 result = DatabricksClientManager._create_authenticated_client(test_host)

#                 assert result == mock_client_instance
#                 mock_client_class.assert_called_once_with(
#                     host=test_host, token=test_token
#                 )

#     def test_create_authenticated_client_without_token(self):
#         """Test client creation without user token (default auth)."""
#         test_host = "https://example.databricks.com"

#         with patch.object(
#             DatabricksClientManager, "_extract_user_token", return_value=None
#         ):
#             with patch("config.WorkspaceClient") as mock_client_class:
#                 mock_client_instance = Mock()
#                 mock_client_class.return_value = mock_client_instance

#                 result = DatabricksClientManager._create_authenticated_client(test_host)

#                 assert result == mock_client_instance
#                 mock_client_class.assert_called_once_with(host=test_host)

#     def test_create_authenticated_client_exception(self):
#         """Test client creation handles exceptions gracefully."""
#         test_host = "https://example.databricks.com"

#         with patch.object(
#             DatabricksClientManager, "_extract_user_token", return_value=None
#         ):
#             with patch(
#                 "config.WorkspaceClient", side_effect=Exception("Connection failed")
#             ):
#                 result = DatabricksClientManager._create_authenticated_client(test_host)

#                 assert result is None

#     def test_test_client_connection_success(self):
#         """Test successful client connection testing."""
#         mock_client = Mock()
#         mock_user = Mock()
#         mock_user.user_name = "test@example.com"
#         mock_user.display_name = "Test User"
#         mock_user.id = "user-123"

#         mock_client.current_user.me.return_value = mock_user

#         result = DatabricksClientManager._test_client_connection(mock_client)

#         expected_user_info = {
#             "user_name": "test@example.com",
#             "display_name": "Test User",
#             "user_id": "user-123",
#         }

#         assert result == expected_user_info
#         mock_client.current_user.me.assert_called_once()

#     def test_test_client_connection_failure(self):
#         """Test client connection testing handles failures."""
#         mock_client = Mock()
#         mock_client.current_user.me.side_effect = Exception("Auth failed")

#         with patch("streamlit.error") as mock_error:
#             result = DatabricksClientManager._test_client_connection(mock_client)

#             assert result is None
#             mock_error.assert_called_once()

#     @patch("streamlit.session_state", {})
#     def test_store_client_in_session(self):
#         """Test storing client and user info in session state."""
#         mock_client = Mock()
#         user_info = {
#             "user_name": "test@example.com",
#             "display_name": "Test User",
#             "user_id": "user-123",
#         }

#         with patch("streamlit.success") as mock_success:
#             DatabricksClientManager._store_client_in_session(mock_client, user_info)

#             # Import streamlit to access session_state
#             import streamlit as st

#             assert st.session_state.workspace_client == mock_client
#             assert st.session_state.databricks_user_info == user_info
#             mock_success.assert_called_once()

#     def test_handle_setup_error(self):
#         """Test error handling provides useful feedback."""
#         test_error = Exception("Test error message")

#         with patch("streamlit.error") as mock_error:
#             with patch("streamlit.expander") as mock_expander:
#                 mock_expander_context = Mock()
#                 mock_expander.return_value.__enter__ = Mock(
#                     return_value=mock_expander_context
#                 )
#                 mock_expander.return_value.__exit__ = Mock(return_value=None)

#                 DatabricksClientManager._handle_setup_error(test_error)

#                 mock_error.assert_called_once()
#                 mock_expander.assert_called_once_with("🔧 Troubleshooting Tips")


# if __name__ == "__main__":
#     # Run tests if executed directly
#     pytest.main([__file__, "-v"])
