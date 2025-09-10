#!/bin/bash

set -e

if ! command -v databricks &> /dev/null; then
    echo "Error: Databricks CLI not found. Please install it first."
    exit 1
fi

if [ ! -f "databricks.yml" ]; then
    echo "Error: databricks.yml not found. Run from the dbxmetagen directory."
    exit 1
fi

if ! databricks current-user me &> /dev/null; then
    echo "Error: Not authenticated with Databricks. Run: databricks configure"
    exit 1
fi

create_secret_scope() {
    local scope_name="dbxmetagen"
    
    if ! databricks secrets list-scopes | grep -q "$scope_name"; then
        databricks secrets create-scope "$scope_name"
        echo "Created secret scope: $scope_name"
    fi
    
    if ! databricks secrets list-secrets "$scope_name" 2>/dev/null | grep -q "databricks_token"; then
        echo "Enter your Databricks token (generate in User Settings > Access Tokens):"
        read -s -p "Token: " token
        echo
        
        if [ -n "$token" ]; then
            databricks secrets put-secret --json "{
                \"scope\": \"$scope_name\",
                \"key\": \"databricks_token\", 
                \"string_value\": \"$token\"
            }"
            echo "Token configured"
        fi
    fi
}

validate_bundle() {
    echo "Validating bundle..."
    if ! databricks bundle validate; then
        echo "Error: Bundle validation failed"
        exit 1
    fi
}

deploy_bundle() {
    TARGET="${TARGET:-dev}"
    echo "Deploying to $TARGET..."
    
    DEPLOY_VARS=""
    if [ "$DEBUG_MODE" = true ]; then
        DEPLOY_VARS="$DEPLOY_VARS --var=debug_mode=true"
    fi
    if [ "$CREATE_TEST_DATA" = true ]; then
        DEPLOY_VARS="$DEPLOY_VARS --var=create_test_data=true"
    fi
    
    if ! databricks bundle deploy --target "$TARGET" $DEPLOY_VARS; then
        echo "Error: Bundle deployment failed"
            exit 1
        fi
            
            export DEPLOY_TARGET="$TARGET"
    echo "Bundle deployed successfully"
}

run_permissions_setup() {
    echo "Setting up permissions..."
    
    local catalog_name="dbxmetagen"
    if [ -f "variables.yml" ]; then
        catalog_name=$(awk '/catalog_name:/{flag=1; next} flag && /default:/{print $2; exit}' variables.yml | xargs)
        catalog_name=${catalog_name:-dbxmetagen}
    fi
    
    local job_id
    job_id=$(databricks jobs list --output json | grep -B5 -A5 "dbxmetagen_permissions_setup" | grep '"job_id"' | head -1 | sed 's/.*"job_id": *\([0-9]*\).*/\1/' || echo "")
    
    if [ -n "$job_id" ]; then
        echo "Running permissions setup job..."
        databricks jobs run-now --json "{\"job_id\": $job_id, \"job_parameters\": {\"catalog_name\": \"$catalog_name\"}}"
        echo "Permissions job started (ID: $job_id)"
    else
        echo "Warning: Could not find permissions setup job"
    fi
}

start_app() {
    if databricks apps list --output json 2>/dev/null | grep -q "dbxmetagen-app"; then
        echo "Starting app..."
        databricks apps start dbxmetagen-app 2>/dev/null || true
        databricks apps deploy dbxmetagen-app 2>/dev/null || true
        echo "App deployment complete"
    else
        echo "Warning: App not found in workspace"
    fi
}

# Parse arguments
RUN_PERMISSIONS=false
DEBUG_MODE=false
CREATE_TEST_DATA=false
TARGET="dev"

while [[ $# -gt 0 ]]; do
    case $1 in
        --permissions)
            RUN_PERMISSIONS=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --create-test-data)
            CREATE_TEST_DATA=true
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --permissions     Run permissions setup job"
            echo "  --debug          Enable debug mode"
            echo "  --create-test-data Generate test data"
            echo "  --target TARGET  Deploy to specific target (dev/prod)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
echo "DBX MetaGen Deployment"
echo "Target: $TARGET"

# Copy variables to app folder if it exists
if [ -f "variables.yml" ]; then
    cp variables.yml app/ 2>/dev/null || true
fi

# Deploy everything
create_secret_scope
validate_bundle
deploy_bundle
start_app

# Run permissions if requested
if [ "$RUN_PERMISSIONS" = true ]; then
        run_permissions_setup
fi

echo "Deployment complete!"
echo "Access your app in Databricks workspace > Apps > dbxmetagen-app"