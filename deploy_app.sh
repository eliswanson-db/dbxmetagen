#!/bin/bash

# DBX MetaGen App Deployment Script
# This script helps deploy the DBX MetaGen Streamlit app using Databricks Asset Bundle

set -e

echo "🚀 DBX MetaGen App Deployment Script"
echo "====================================="

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo "❌ Databricks CLI not found. Please install it first:"
    echo "   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh"
    echo "   For more details: https://docs.databricks.com/en/dev-tools/cli/install.html"
    exit 1
fi

# Check CLI version compatibility
CLI_VERSION=$(databricks version 2>/dev/null | head -1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' || echo "0.0.0")
echo "📋 Databricks CLI version: $CLI_VERSION"

# Convert version to comparable number (major.minor.patch -> major*10000 + minor*100 + patch)
version_to_number() {
    echo "$1" | awk -F. '{printf "%d", $1*10000 + $2*100 + $3}'
}

MIN_VERSION="0.234.0"
CLI_VERSION_NUM=$(version_to_number "$CLI_VERSION")
MIN_VERSION_NUM=$(version_to_number "$MIN_VERSION")

if [ "$CLI_VERSION_NUM" -lt "$MIN_VERSION_NUM" ]; then
    echo "⚠️  Warning: Databricks CLI version $CLI_VERSION detected."
    echo "   This script requires version $MIN_VERSION or higher for proper functionality."
    echo "   Please update your CLI: https://docs.databricks.com/en/dev-tools/cli/install.html"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if we're in the right directory
if [ ! -f "databricks.yml" ]; then
    echo "❌ databricks.yml not found. Please run this script from the dbxmetagen directory."
    exit 1
fi

# Get current directory name
PROJECT_DIR=$(basename "$PWD")
echo "📂 Project directory: $PROJECT_DIR"

# Check if user is authenticated
echo "🔐 Checking Databricks authentication..."
if ! databricks current-user me &> /dev/null; then
    echo "❌ Not authenticated with Databricks. Please run:"
    echo "   databricks configure"
    exit 1
fi

# Get current user
CURRENT_USER=$(databricks current-user me | grep -o '"userName":"[^"]*"' | sed 's/"userName":"\([^"]*\)"/\1/')
echo "👤 Authenticated as: $CURRENT_USER"

# Function to create secret scope
create_secret_scope() {
    local scope_name="dbxmetagen"
    
    echo "🔑 Setting up secret scope: $scope_name"
    
    # Check if scope exists
    if databricks secrets list-scopes | grep -q "$scope_name"; then
        echo "✅ Secret scope '$scope_name' already exists"
    else
        echo "📝 Creating secret scope: $scope_name"
        databricks secrets create-scope "$scope_name"
        echo "✅ Secret scope created"
    fi
    
    # Check if token secret exists
    if databricks secrets list-secrets "$scope_name" 2>/dev/null | grep -q "databricks_token"; then
        echo "✅ Token secret already configured"
    else
        echo "🔐 Please enter your Databricks token for the app:"
        echo "   (You can generate one in User Settings > Access Tokens)"
        read -s -p "Token: " token
        echo
        
        if [ -n "$token" ]; then
            # Use the new CLI format with JSON input
            databricks secrets put-secret --json "{
                \"scope\": \"$scope_name\",
                \"key\": \"databricks_token\", 
                \"string_value\": \"$token\"
            }"
            echo "✅ Token secret configured"
        else
            echo "⚠️  No token provided. You'll need to configure this manually later:"
            echo "   databricks secrets put-secret --json '{\"scope\": \"$scope_name\", \"key\": \"databricks_token\", \"string_value\": \"YOUR_TOKEN\"}'"
        fi
    fi
}

# Function to validate bundle
validate_bundle() {
    echo "✅ Validating bundle configuration..."
    if databricks bundle validate; then
        echo "✅ Bundle validation passed"
    else
        echo "❌ Bundle validation failed. Please check your configuration."
        exit 1
    fi
}

# Function to deploy bundle
deploy_bundle() {
    echo "🚀 Deploying bundle..."
    
    # Ask for target environment
    echo "📋 Available targets:"
    echo "   1. dev (development)"
    echo "   2. prod (production)"
    read -p "Select target environment (1 or 2, default: 1): " target_choice
    
    case $target_choice in
        2)
            TARGET="prod"
            ;;
        *)
            TARGET="dev"
            ;;
    esac
    
    echo "🎯 Deploying to target: $TARGET"
    
    if databricks bundle deploy --target "$TARGET"; then
        echo "✅ Bundle deployed successfully to $TARGET"
    else
        echo "❌ Bundle deployment failed"
        exit 1
    fi
}

# Function to show app URL
show_app_info() {
    echo "📱 App Information"
    echo "=================="
    echo "✅ DBX MetaGen app has been deployed successfully!"
    echo ""
    echo "🔗 To access your app:"
    echo "   1. Go to your Databricks workspace"
    echo "   2. Navigate to Apps in the left sidebar"
    echo "   3. Find 'dbxmetagen-app' in your apps list"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Configure your catalog and host settings in the app"
    echo "   2. Upload a table_names.csv or manually enter table names"
    echo "   3. Create and run metadata generation jobs"
    echo "   4. Review generated metadata and download results"
    echo ""
    echo "📖 For detailed usage instructions, see app/README.md"
}

# Function to show troubleshooting info
show_troubleshooting() {
    echo ""
    echo "🔧 Troubleshooting"
    echo "=================="
    echo "If you encounter issues:"
    echo ""
    echo "1. **Authentication errors:**"
    echo "   databricks configure"
    echo ""
    echo "2. **Secret scope issues:**"
    echo "   databricks secrets create-scope dbxmetagen"
    echo "   databricks secrets put-secret --json '{\"scope\": \"dbxmetagen\", \"key\": \"databricks_token\", \"string_value\": \"YOUR_TOKEN\"}'"
    echo ""
    echo "3. **Bundle validation errors:**"
    echo "   databricks bundle validate"
    echo "   Check databricks.yml and resource files"
    echo ""
    echo "4. **App access issues:**"
    echo "   Verify permissions in databricks.yml"
    echo "   Check app status in Databricks workspace"
    echo ""
}

# Main execution
main() {
    echo "Starting deployment process..."
    echo ""
    
    # Step 1: Create secret scope and configure token
    create_secret_scope
    echo ""
    
    # Step 2: Copy configuration to app folder
    echo "📋 Copying configuration files to app folder..."
    if [ -f "variables.yml" ]; then
        cp variables.yml app/
        echo "✅ variables.yml copied to app folder"
    else
        echo "⚠️  variables.yml not found, app will use default configuration"
    fi
    echo ""
    
    # Step 3: Validate bundle
    validate_bundle
    echo ""
    
    # Step 4: Deploy bundle
    deploy_bundle
    echo ""
    
    # Step 5: Show success information
    show_app_info
    
    # Step 6: Show troubleshooting info
    show_troubleshooting
    
    echo ""
    echo "🎉 Deployment complete! Happy metadata generating!"
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "DBX MetaGen App Deployment Script"
    echo ""
    echo "Usage: ./deploy_app.sh"
    echo ""
    echo "This script will:"
    echo "  1. Check Databricks CLI installation and authentication"
    echo "  2. Create secret scope and configure app token"
    echo "  3. Validate the bundle configuration"
    echo "  4. Deploy the app to your chosen environment"
    echo ""
    echo "Prerequisites:"
    echo "  - Databricks CLI installed and configured"
    echo "  - Proper permissions in your Databricks workspace"
    echo "  - Valid databricks.yml configuration"
    echo ""
    exit 0
fi

# Run main function
main 