#!/usr/bin/env bash
# Featurama Quick Start Script

set -e

echo "================================================================================"
echo "🚀 Featurama - Quick Start"
echo "================================================================================"
echo ""

# Check if ScyllaDB is running
echo "🔍 Checking if ScyllaDB is running..."
if ! docker ps | grep -q featurama-scylla; then
    echo "⚠️  ScyllaDB is not running. Starting it now..."
    docker-compose up -d
    echo "⏳ Waiting for ScyllaDB to be ready (30 seconds)..."
    sleep 30
else
    echo "✅ ScyllaDB is already running"
fi
echo ""

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source .venv/bin/activate
echo ""

# Run setup steps
echo "================================================================================"
echo "Step 1: Initialize ScyllaDB Schema"
echo "================================================================================"
python examples/01_setup_scylla.py
echo ""

echo "================================================================================"
echo "Step 2: Generate Synthetic Data"
echo "================================================================================"
python examples/02_generate_data.py
echo ""

echo "================================================================================"
echo "Step 3: Ingest Features"
echo "================================================================================"
python examples/03_feature_ingestion.py
echo ""

echo "================================================================================"
echo "Step 4: Benchmark Feature Retrieval"
echo "================================================================================"
python examples/04_feature_retrieval.py
echo ""

echo "================================================================================"
echo "Step 5: Train ML Model"
echo "================================================================================"
python examples/05_train_model.py
echo ""

echo "================================================================================"
echo "✨ Featurama Quick Start Complete!"
echo "================================================================================"
echo ""
echo "🎉 Your feature store is now ready!"
echo ""
echo "Next steps:"
echo "  1. Start the inference server:"
echo "     python -m featurama.ml.inference"
echo ""
echo "  2. Run the inference demo (in another terminal):"
echo "     python examples/06_inference.py"
echo ""
echo "  3. Try Ray Data integration:"
echo "     python examples/07_ray_integration.py"
echo ""
echo "  4. Explore the API docs at:"
echo "     http://localhost:8000/docs"
echo ""

