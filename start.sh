#!/bin/bash

# Ensure required storage directories exist
mkdir -p models static

# Make script executable
chmod +x start.sh

# Launch 24/7 continuous internet learner in the background
python continuous_learner.py &

# Launch FastAPI web application server
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
