#!/bin/bash

echo "$0"

echo "$1"

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