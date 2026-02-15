#!/usr/bin/env bash

echo "🚀 Starting Stablecoin RAG Backend..."

# --------------------------------------------------
# Step 1: Download embeddings + build vector DB
# --------------------------------------------------

if [ ! -d "vector_db" ]; then
    echo "📦 Vector DB not found. Building index..."

    python scripts/build_index.py

    echo "✅ Index built successfully."
else
    echo "✅ Vector DB already exists. Skipping build."
fi

# --------------------------------------------------
# Step 2: Start API Server
# --------------------------------------------------

echo "🌐 Starting FastAPI server..."

uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}