-- DBX MetaGen App Permissions Setup
-- This SQL script grants necessary permissions to the app service principal

-- Set parameters (these will be replaced by the job parameters)
SET VAR catalog_name = '${catalog_name}';
SET VAR app_service_principal = '${app_service_principal}';

-- Grant USE CATALOG permission
GRANT USE CATALOG ON CATALOG ${catalog_name} TO `${app_service_principal}`;

-- Grant CREATE SCHEMA permission to allow metadata schema creation
GRANT CREATE SCHEMA ON CATALOG ${catalog_name} TO `${app_service_principal}`;

-- Grant USE SCHEMA permission on existing schemas that might contain target tables
-- This is broad but necessary since we don't know all source schemas in advance
GRANT USE SCHEMA ON SCHEMA ${catalog_name}.* TO `${app_service_principal}`;

-- Grant SELECT permission on all tables in the catalog
-- This allows the app to read table metadata and sample data
GRANT SELECT ON SCHEMA ${catalog_name}.* TO `${app_service_principal}`;

-- Grant permissions on the metadata results schema specifically
-- These will be created by the app
GRANT ALL PRIVILEGES ON SCHEMA ${catalog_name}.metadata_results TO `${app_service_principal}`;

-- Grant permission to create and manage volumes for file storage
GRANT CREATE VOLUME ON SCHEMA ${catalog_name}.metadata_results TO `${app_service_principal}`;

-- Grant permission to execute on the catalog (for functions and procedures)
GRANT EXECUTE ON CATALOG ${catalog_name} TO `${app_service_principal}`;

-- Output completion message
SELECT 'Permissions granted successfully to ' || '${app_service_principal}' || ' on catalog ' || '${catalog_name}' AS status; 