# Makefile for Featurama

.PHONY: help install setup start-db stop-db clean test quickstart inference

help:
	@echo "Featurama - Available Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make start-db     - Start ScyllaDB with Docker Compose"
	@echo "  make stop-db      - Stop ScyllaDB"
	@echo "  make setup        - Initialize ScyllaDB schema"
	@echo "  make quickstart   - Run complete setup (all examples)"
	@echo "  make inference    - Start inference server"
	@echo "  make clean        - Clean generated data and models"
	@echo "  make test         - Run syntax checks"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

start-db:
	@echo "🚀 Starting ScyllaDB..."
	docker-compose up -d
	@echo "⏳ Waiting for ScyllaDB to be ready..."
	@sleep 10
	@echo "✅ ScyllaDB started!"

stop-db:
	@echo "🛑 Stopping ScyllaDB..."
	docker-compose down

setup:
	@echo "🏗️  Initializing schema..."
	python examples/01_setup_scylla.py

quickstart:
	@./quickstart.sh

inference:
	@echo "🚀 Starting inference server..."
	@echo "API docs will be available at: http://localhost:8000/docs"
	python -m featurama.ml.inference

clean:
	@echo "🧹 Cleaning up..."
	rm -rf data/*.csv
	rm -rf models/*.pkl
	rm -rf data/ray_features
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

test:
	@echo "🧪 Running syntax checks..."
	python -m py_compile featurama/**/*.py
	python -m py_compile examples/*.py
	@echo "✅ All files valid!"

