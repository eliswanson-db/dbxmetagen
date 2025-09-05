"""
DBXMetaGen Streamlit Application.

"""

import streamlit as st
import logging
import sys
from core.config import ConfigManager, DatabricksClientManager
from ui.components import UIComponents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

logger.info("🚀 DBX MetaGen App starting up...")

st.set_page_config(
    page_title="DBX MetaGen",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

class DBXMetaGenApp:
    """Main application class - now much cleaner and focused."""

    def __init__(self):
        """Initialize the application.""" # TODO: Leave off review here
        self.config_manager = ConfigManager()
        self.ui_components = UIComponents()

        client_ready = DatabricksClientManager.setup_client()

        if client_ready:
            logger.info(
                "✅ DBX MetaGen App initialized successfully with Databricks connection"
            )
        else:
            logger.warning(
                "⚠️ DBX MetaGen App initialized but Databricks connection failed"
            )

        self.client_ready = client_ready

    def run(self):
        """Main app execution - clean and organized."""
        # Header
        st.title("🏷️ DBX MetaGen")
        st.markdown("### AI-Powered Metadata Generation for Databricks Tables")

        # Sidebar configuration
        self.ui_components.render_sidebar_config()

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Tables & Jobs", "📊 Results", "✏️ Review Metadata", "❓ Help"]
        )

        with tab1:
            self.ui_components.render_unified_table_management()
            st.markdown("---")
            self.ui_components.render_job_status_section()

        with tab2:
            self.ui_components.render_results_viewer()

        with tab3:
            self.ui_components.render_metadata_review()

        with tab4:
            self.ui_components.render_help()

    def render_debug_info(self):
        """Render debug information in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Debug Information")

        if st.sidebar.button("Show Debug Info"):
            st.sidebar.write("**System Status:**")

            # Check workspace client
            if st.session_state.get("workspace_client"):
                st.sidebar.write("- Workspace Client: ✅ Connected")
            else:
                st.sidebar.write("- Workspace Client: ❌ Not initialized")

            # Check config
            if st.session_state.get("config"):
                st.sidebar.write(f"- Config: ✅ {len(st.session_state.config)} keys")
            else:
                st.sidebar.write("- Config: ❌ Not loaded")

            # Log viewing instructions
            st.sidebar.markdown("---")
            st.sidebar.write("**📋 View Full Logs:**")
            st.sidebar.write("Add `/logz` to your app URL")

            logger.info("🔍 Debug info requested by user")


# Run the app
if __name__ == "__main__":
    try:
        # Initialize and run the app
        app = DBXMetaGenApp()
        app.run()

        # Add debug info to sidebar
        app.render_debug_info()

    except Exception as e:
        logger.error("❌ Application failed to start: %s", str(e))
        st.error(f"❌ Application Error: {str(e)}")

        # Show error details in expander
        with st.expander("Error Details"):
            st.exception(e)
