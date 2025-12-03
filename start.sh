#!/bin/bash
# Start FastAPI backend on port 8000
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to be ready
sleep 3

# Start Streamlit frontend on port 8080 (App Runner's default port)
streamlit run frontend/dashboard.py --server.port 8080 --server.address 0.0.0.0 --server.headless true

# Wait for all background processes
wait

