#!/bin/bash
# Start script for Streamlit app

echo "Starting Streamlit application..."

# Run Streamlit
streamlit run dashboard.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

