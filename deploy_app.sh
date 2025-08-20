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
    
    # Pass debug mode variable to bundle deployment
    if [ "$DEBUG_MODE" = true ]; then
        echo "🐛 Debug mode: Deploying with debug_mode=true"
        if databricks bundle deploy --target "$TARGET" --var="debug_mode=true"; then
            echo "✅ Bundle deployed successfully to $TARGET with debug mode"
            
            # Store target for later use
            export DEPLOY_TARGET="$TARGET"
        else
            echo "❌ Bundle deployment failed"
            exit 1
        fi
    else
        if databricks bundle deploy --target "$TARGET" --var="debug_mode=false"; then
            echo "✅ Bundle deployed successfully to $TARGET"
            
            # Store target for later use
            export DEPLOY_TARGET="$TARGET"
        else
            echo "❌ Bundle deployment failed"
            exit 1
        fi
    fi
}

# Function to run permissions setup job
run_permissions_setup() {
    echo "🔐 Setting up permissions..."
    
    # Get catalog name from variables.yml
    local catalog_name="dbxmetagen"
    if [ -f "variables.yml" ]; then
        catalog_name=$(awk '/catalog_name:/{flag=1; next} flag && /default:/{print $2; exit}' variables.yml || echo "dbxmetagen")
        # Trim any whitespace and use fallback if empty
        catalog_name=$(echo "$catalog_name" | xargs)
        if [ -z "$catalog_name" ]; then
            catalog_name="dbxmetagen"
        fi
        echo "📊 Using catalog: $catalog_name"
    else
        echo "📊 Using catalog: $catalog_name (variables.yml not found)"
    fi
    
    # Get the app service principal name using 'databricks apps get'
    echo "🔍 Getting app service principal name..."
    local app_name="dbxmetagen-app"
    local app_details
    app_details=$(databricks apps get "$app_name" --output json 2>/dev/null || echo "")
    
    local service_principal=""
    if [ -n "$app_details" ]; then
        # Extract service principal name from the app details JSON
        service_principal=$(echo "$app_details" | grep -o '"service_principal_name": *"[^"]*"' | sed 's/"service_principal_name": *"\([^"]*\)"/\1/' || echo "")
    fi
    
    # If we still don't have the service principal, prompt for it
    if [ -z "$service_principal" ]; then
        echo "⚠️  Could not determine service principal automatically from app details."
        echo "📋 Available apps:"
        databricks apps list --output table 2>/dev/null || echo "Could not list apps"
        echo ""
        read -p "Enter the app service principal name: " service_principal
    fi
    
    if [ -n "$service_principal" ]; then
        echo "🎯 Using service principal: $service_principal"
        
        # Get the service principal ID from the name using Databricks CLI
        echo "🔍 Looking up service principal ID..."
        local service_principal_id=""
        
        if command -v jq &> /dev/null; then
            # Use jq for precise JSON parsing - note: field is "displayName" and we need "applicationId"
            service_principal_id=$(databricks service-principals list --output json | jq -r --arg name "$service_principal" '.[] | select(.displayName == $name) | .applicationId' || echo "")
        else
            # Fallback to grep/sed if jq is not available
            local sp_list
            sp_list=$(databricks service-principals list --output json 2>/dev/null || echo "[]")
            if [ -n "$sp_list" ]; then
                # Extract ID for matching service principal name using grep and sed
                service_principal_id=$(echo "$sp_list" | grep -B3 -A3 "\"$service_principal\"" | grep '"applicationId"' | head -1 | sed 's/.*"applicationId": *"\([^"]*\)".*/\1/' || echo "")
            fi
        fi
        
        if [ -n "$service_principal_id" ]; then
            echo "✅ Found service principal ID: $service_principal_id"
        else
            echo "⚠️  Could not find service principal ID for name: $service_principal"
            echo "📋 Available service principals:"
            databricks service-principals list --output table 2>/dev/null || echo "Could not list service principals"
            echo ""
            echo "💡 You can run the permissions setup manually later using:"
            echo "   1. Get service principal ID: databricks service-principals list"
            echo "   2. Run job: databricks jobs run-now --job-id JOB_ID --json '{\"job_parameters\": {\"catalog_name\": \"$catalog_name\", \"app_service_principal\": \"SERVICE_PRINCIPAL_ID\"}}'"
            service_principal_id=""
        fi
        
        # Get the job ID for permissions setup (accounting for DAB naming: "dev user@example.com dbxmetagen_permissions_setup")
        local job_id
        local job_count=0
        local matching_jobs=""
        
        if command -v jq &> /dev/null; then
            # Use jq for precise JSON parsing - jobs list returns an array at root level
            matching_jobs=$(databricks jobs list --output json | jq -r '.[] | select(.settings.name | contains("dbxmetagen_permissions_setup")) | "\(.job_id):\(.settings.name)"' || echo "")
        else
            # Fallback to grep/sed if jq is not available
            matching_jobs=$(databricks jobs list --output json | grep -B5 -A5 "dbxmetagen_permissions_setup" | grep -E '"job_id"|"name"' | paste - - | sed 's/.*"job_id": *\([0-9]*\).*"name": *"\([^"]*\)".*/\1:\2/' || echo "")
        fi
        
        if [ -n "$matching_jobs" ]; then
            # Count number of matching jobs
            job_count=$(echo "$matching_jobs" | wc -l | tr -d ' ')
            
            if [ "$job_count" -eq 1 ]; then
                # Exactly one job found - safe to proceed
                job_id=$(echo "$matching_jobs" | cut -d':' -f1)
                local job_name=$(echo "$matching_jobs" | cut -d':' -f2-)
                echo "✅ Found permissions setup job: $job_name (ID: $job_id)"
            elif [ "$job_count" -gt 1 ]; then
                # Multiple jobs found - ask user to choose
                echo "⚠️  Found $job_count jobs matching 'dbxmetagen_permissions_setup':"
                echo ""
                local counter=1
                while IFS=':' read -r id name; do
                    echo "  $counter. $name (ID: $id)"
                    counter=$((counter + 1))
                done <<< "$matching_jobs"
                echo ""
                
                read -p "Select job number to run (1-$job_count), or 0 to skip: " job_choice
                
                if [ "$job_choice" -ge 1 ] && [ "$job_choice" -le "$job_count" ]; then
                    job_id=$(echo "$matching_jobs" | sed -n "${job_choice}p" | cut -d':' -f1)
                    local selected_job_name=$(echo "$matching_jobs" | sed -n "${job_choice}p" | cut -d':' -f2-)
                    echo "🎯 Selected: $selected_job_name (ID: $job_id)"
                elif [ "$job_choice" = "0" ]; then
                    echo "⏭️  Skipping permissions setup. You can run it manually later."
                    job_id=""
                else
                    echo "❌ Invalid selection. Skipping permissions setup."
                    job_id=""
                fi
            fi
        else
            # No matching jobs found
            job_id=""
        fi
        
        if [ -n "$job_id" ] && [ -n "$service_principal_id" ]; then
            echo "🚀 Running permissions setup job (ID: $job_id)..."
            
            # Debug: Show the exact command we're about to run
            echo "🔍 Debug: Running job with catalog_name='$catalog_name' and app_service_principal='$service_principal_id'"
            
            # Run the job with parameters - use job_parameters for jobs with parameter definitions
            local run_result
            run_result=$(databricks jobs run-now \
                --json "{\"job_id\": $job_id, \"job_parameters\": {\"catalog_name\": \"$catalog_name\", \"app_service_principal\": \"$service_principal_id\"}}" \
                --output json 2>&1 || echo "ERROR_OCCURRED")
            
            # Debug: Show the raw result
            echo "🔍 Debug: Job run result: $run_result"
            
            if [ -n "$run_result" ] && [ "$run_result" != "ERROR_OCCURRED" ]; then
                local run_id
                run_id=$(echo "$run_result" | grep -o '"run_id": *[0-9]*' | sed 's/"run_id": *\([0-9]*\)/\1/' || echo "")
                
                if [ -n "$run_id" ]; then
                    echo "✅ Permissions setup job started (Run ID: $run_id)"
                    
                    # Get workspace URL for job link
                    local workspace_url
                    workspace_url=$(databricks current-user me | grep -o '"userName":"[^"]*"' | sed 's/"userName":"\([^"]*\)"/\1/' | sed 's/@.*//' || echo "")
                    if [ -n "$workspace_url" ]; then
                        echo "🔗 Monitor progress at: https://$workspace_url#job/$job_id/run/$run_id"
                    fi
                    
                    # Wait a bit and check status
                    echo "⏳ Waiting for job to complete..."
                    sleep 15
                    
                    local status
                    status=$(databricks jobs get-run "$run_id" --output json | grep -o '"life_cycle_state": *"[^"]*"' | sed 's/"life_cycle_state": *"\([^"]*\)"/\1/' || echo "UNKNOWN")
                    
                    case $status in
                        "TERMINATED")
                            # Check if it was successful
                            local result_state
                            result_state=$(databricks jobs get-run "$run_id" --output json | grep -o '"result_state": *"[^"]*"' | sed 's/"result_state": *"\([^"]*\)"/\1/' || echo "UNKNOWN")
                            if [ "$result_state" = "SUCCESS" ]; then
                                echo "✅ Permissions setup completed successfully!"
                            else
                                echo "⚠️  Permissions setup job terminated with status: $result_state"
                                echo "   Please check the job logs in Databricks for details."
                            fi
                            ;;
                        "RUNNING"|"PENDING")
                            echo "🔄 Permissions setup is still running. Check the Databricks workspace for completion."
                            ;;
                        "FAILED")
                            echo "❌ Permissions setup job failed. Please check the job logs in Databricks."
                            ;;
                        *)
                            echo "⚠️  Permissions setup status: $status. Please check the job logs in Databricks."
                            ;;
                    esac
                else
                    echo "⚠️  Could not extract run ID from job result."
                    echo "   Raw result: $run_result"
                fi
            else
                echo "⚠️  Failed to start permissions setup job."
                if [ "$run_result" = "ERROR_OCCURRED" ]; then
                    echo "   Check the debug output above for error details."
                else
                    echo "   No result returned from job run command."
                fi
            fi
        elif [ -n "$job_id" ] && [ -z "$service_principal_id" ]; then
            echo "⚠️  Found permissions setup job but could not determine service principal ID."
            echo "   Job ID: $job_id"
            echo "   You'll need to run the job manually with the correct service principal ID."
            echo ""
            echo "💡 To run manually:"
            echo "   1. Get the service principal ID: databricks service-principals list"
            echo "   2. Go to Databricks workspace → Workflows → Jobs"
            echo "   3. Find job ID $job_id"
            echo "   4. Run with parameters: catalog_name=$catalog_name, app_service_principal=SERVICE_PRINCIPAL_ID"
        else
            echo "⚠️  Could not find or execute permissions setup job."
            echo "📋 Available jobs containing 'permission':"
            databricks jobs list --output text | grep -i permission || echo "No permission-related jobs found"
            echo ""
            echo "💡 To run manually:"
            echo "   1. Get the service principal ID: databricks service-principals list"
            echo "   2. Go to Databricks workspace → Workflows → Jobs"
            echo "   3. Find a job containing 'dbxmetagen_permissions_setup'"
            echo "   4. Run with parameters: catalog_name=$catalog_name, app_service_principal=SERVICE_PRINCIPAL_ID"
        fi
    else
        echo "⚠️  Service principal not found. Please run the permissions setup job manually."
    fi
    
    echo ""
}

# Function to start and check the app
start_and_check_app() {
    echo "🚀 Starting and checking Streamlit app..."
    
    echo "📋 Checking if app was deployed by bundle..."
    
    # Check if app exists
    if databricks apps list --output json 2>/dev/null | grep -q "dbxmetagen-app"; then
        echo "✅ App 'dbxmetagen-app' found in workspace"
    else
        echo "❌ App 'dbxmetagen-app' not found"
        echo "   The app should have been created during bundle deployment"
        echo "   Check the bundle deployment logs above for errors"
        return 1
    fi
    
    # Try to start the app
    echo "🚀 Starting the app..."
    if databricks apps start dbxmetagen-app 2>&1; then
        echo "✅ App start command executed successfully"
    else
        echo "⚠️ App start command may have failed, but this is often normal if the app is already running"
    fi

    echo "🚀 Deploying the app..."
    if databricks apps deploy dbxmetagen-app 2>&1; then
        echo "✅ App deployment command executed successfully"
    else
        echo "⚠️ App deployment command failed... Please check the app status in the Databricks workspace"
    fi
    
    # Wait a moment for the app to initialize
    echo "⏳ Waiting for app to initialize..."
    sleep 15
    
    # Check app status
    echo "📊 Checking app status..."
    local app_status
    app_status=$(databricks apps get dbxmetagen-app --output json 2>/dev/null | grep -o '"compute_status":"[^"]*"' | sed 's/"compute_status":"\([^"]*\)"/\1/' || echo "UNKNOWN")
    
    echo "📊 App compute status: $app_status"
    
    # Also check the app state
    local app_state
    app_state=$(databricks apps get dbxmetagen-app --output json 2>/dev/null | grep -o '"state":"[^"]*"' | sed 's/"state":"\([^"]*\)"/\1/' || echo "UNKNOWN")
    
    echo "📊 App state: $app_state"
    
    if [ "$app_status" = "ACTIVE" ] && [ "$app_state" = "RUNNING" ]; then
        echo "✅ App is running and ready to use!"
    elif [ "$app_status" = "ACTIVE" ] || [ "$app_state" = "RUNNING" ]; then
        echo "🔄 App is starting up or partially ready..."
        echo "   Check the app status in a few minutes"
    elif [ "$app_status" = "STARTING" ] || [ "$app_state" = "STARTING" ]; then
        echo "🔄 App is starting up..."
        echo "   This may take a few minutes for the first deployment"
    else
        echo "⚠️ App status: $app_status, state: $app_state"
        echo "   Check the Databricks workspace for details"
        echo "   The app may still be initializing"
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
    echo "   1. ✅ Permissions have been set up automatically (if successful)"
    echo "   2. Configure your catalog and host settings in the app"  
    echo "   3. Upload a table_names.csv or manually enter table names"
    echo "   4. Create and run metadata generation jobs"
    echo "   5. Review generated metadata and download results"
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
    echo "4. **App deployment issues:**"
    echo "   Check app source code path in resources/apps/dbxmetagen_app.yml"
    echo "   Verify app files are in ./app/ directory"
    echo "   Check app status: databricks apps get dbxmetagen-app"
    echo "   View app logs in Databricks workspace"
    echo ""
    echo "5. **App access issues:**"
    echo "   Verify permissions in databricks.yml"
    echo "   Check app status in Databricks workspace"
    echo "   Try restarting: databricks apps restart dbxmetagen-app"
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
    
    # Debug mode will be passed via bundle variables
    echo ""
    
    # Step 3: Validate bundle
    validate_bundle
    echo ""
    
    # Step 4: Deploy bundle
    deploy_bundle
    echo ""
    
    # Step 5: Start and check the app
    start_and_check_app
    echo ""
    
    # Step 6: Run permissions setup job (unless skipped)
    if [ "$SKIP_PERMISSIONS" = false ]; then
        run_permissions_setup
    else
        echo "⏭️  Skipping permissions setup as requested"
        echo ""
        echo "💡 To set up permissions manually later, run:"
        echo "   ./deploy_app.sh (without --skip-permissions flag)"
        echo ""
    fi
    
    # Step 7: Show success information
    show_app_info
    
    # Step 8: Show troubleshooting info
    show_troubleshooting
    
    echo ""
    echo "🎉 Deployment complete! Happy metadata generating!"
}

# Parse command line arguments
SKIP_PERMISSIONS=false
DEBUG_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "DBX MetaGen App Deployment Script"
            echo ""
            echo "Usage: ./deploy_app.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-permissions    Skip the permissions setup job"
            echo "  --debug              Enable debug mode for detailed logging"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "This script will:"
            echo "  1. Check Databricks CLI installation and authentication"
            echo "  2. Create secret scope and configure app token"
            echo "  3. Validate the bundle configuration"
            echo "  4. Deploy the bundle resources"
            echo "  5. Start the Streamlit app via CLI command"
            echo "  6. Run the permissions setup job (unless --skip-permissions is used)"
            echo ""
            echo "Prerequisites:"
            echo "  - Databricks CLI installed and configured"
            echo "  - Proper permissions in your Databricks workspace"
            echo "  - Valid databricks.yml configuration"
            echo ""
            exit 0
            ;;
        --skip-permissions)
            SKIP_PERMISSIONS=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main 