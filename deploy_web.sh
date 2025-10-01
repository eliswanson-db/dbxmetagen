#!/bin/bash

# DEPRECATED: This script is deprecated. Please use ./deploy.sh instead.
# This legacy script is kept for backward compatibility but will be removed in a future version.

echo "⚠️  DEPRECATED: This deployment script is deprecated."
echo "   Please use './deploy.sh' instead for the new unified deployment experience."
echo "   Run './deploy.sh --help' to see available options."
echo ""
echo "Falling back to legacy deployment method..."
echo ""

echo "$0"

echo "$1"

curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_working_directory>"
    exit 1
fi

CWD=$1
cd "$CWD"

databricks bundle validate
if [ $? -eq 0 ]; then
    databricks bundle deploy
else
    echo "Validation failed. Deployment aborted."
fi