#!/bin/bash

# Install the package
pip install .

# Start the API
uvicorn api:app --host 0.0.0.0 --port 8000

# Start container in interactive shell
# /bin/bash "$@"