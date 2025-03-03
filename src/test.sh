#!/bin/bash

# uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload &

curl localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{ "prompt": ["def fibonnaci(n): return 0"] }'
