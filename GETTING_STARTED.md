# Getting Started with Featurama

Welcome to Featurama! This guide will help you get up and running with the feature store in minutes.

## Prerequisites

- Python 3.13+ (3.11+ should work)
- Docker & Docker Compose
- 4GB+ RAM available
- macOS, Linux, or WSL2 on Windows

## Installation

### 1. Clone and Setup

```bash
cd /path/to/featurama
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start ScyllaDB

```bash
# Start ScyllaDB with Docker Compose
docker-compose up -d

# Check if it's running
docker-compose ps

# Wait for ScyllaDB to be ready (30-60 seconds)
# You can check logs:
docker-compose logs -f scylla
```

### 3. Quick Start (Automated)

The easiest way to get started:

```bash
./quickstart.sh
```

This will:
1. ✅ Check ScyllaDB status
2. ✅ Initialize the schema
3. ✅ Generate synthetic data
4. ✅ Ingest features into the store
5. ✅ Run benchmarks
6. ✅ Train an ML model

**Time**: ~5-10 minutes depending on your machine

### 4. Manual Setup (Step by Step)

If you prefer to run each step manually:

#### Step 1: Initialize Schema

```bash
python examples/01_setup_scylla.py
```

Creates the Featurama keyspace and tables in ScyllaDB.

#### Step 2: Generate Data

```bash
python examples/02_generate_data.py
```

Generates:
- 1,000 characters
- 100 planets
- 10,000 deliveries
- 30 days of historical features
- Saves to `data/entities.csv` and `data/features.csv`

#### Step 3: Ingest Features

```bash
python examples/03_feature_ingestion.py
```

Ingests features into ScyllaDB:
- Registers entities
- Registers feature definitions
- Writes feature values in batches

#### Step 4: Benchmark Performance

```bash
python examples/04_feature_retrieval.py
```

Runs benchmarks for:
- Online feature retrieval
- Historical (point-in-time) queries
- Time-series history queries

#### Step 5: Train ML Model

```bash
python examples/05_train_model.py
```

Trains a delivery time prediction model:
- Uses XGBoost
- Features: distance, weight, hazard level, etc.
- Target: actual delivery duration
- Saves model to `models/delivery_predictor.pkl`

#### Step 6: Start Inference Server

```bash
python -m featurama.ml.inference
```

Starts FastAPI server on http://localhost:8000

API Documentation: http://localhost:8000/docs

#### Step 7: Test Inference (In Another Terminal)

```bash
python examples/06_inference.py
```

Demonstrates:
- Single predictions
- Batch predictions
- Various delivery scenarios

#### Step 8 (Bonus): Ray Data Integration

```bash
python examples/07_ray_integration.py
```

Demonstrates distributed data processing with Ray Data.

## Using the Feature Store

### Python API Examples

#### Register a Feature

```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

fs.register_feature(
    feature_name="customer_lifetime_value",
    feature_type="float",
    description="Total customer value",
    version=1,
    tags={"team": "marketing", "owner": "fry"}
)
```

#### Write Features

```python
import pandas as pd
from datetime import datetime

features_df = pd.DataFrame({
    'entity_id': ['fry_001', 'bender_002', 'leela_003'],
    'feature_name': ['delivery_count', 'delivery_count', 'delivery_count'],
    'value': [42, 99, 67],
    'timestamp': [datetime.now()] * 3
})

fs.write_features(features_df)
```

#### Get Online Features (Latest Values)

```python
features = fs.get_online_features(
    entity_ids=['fry_001', 'bender_002'],
    feature_names=['delivery_count', 'success_rate']
)

print(features)
```

#### Get Historical Features (Point-in-Time)

```python
from datetime import datetime

historical = fs.get_historical_features(
    entity_ids=['fry_001'],
    feature_names=['delivery_count'],
    timestamp=datetime(2026, 1, 1)
)

print(historical)
```

#### Get Time-Series History

```python
history = fs.get_feature_history(
    entity_id='fry_001',
    feature_name='delivery_count',
    limit=100
)

print(history)
```

### REST API Examples

#### Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "distance": 5000,
      "package_weight": 150,
      "hazard_level": 0.3,
      "estimated_duration": 10.5,
      "traffic_level": 0.7,
      "weather_index": 0.8
    }
  }'
```

#### Get Entity Features

```bash
curl "http://localhost:8000/features/fry_001?feature_names=delivery_count,success_rate"
```

#### Health Check

```bash
curl "http://localhost:8000/health"
```

## Using the Makefile

Convenient commands:

```bash
# See all available commands
make help

# Install dependencies
make install

# Start ScyllaDB
make start-db

# Stop ScyllaDB
make stop-db

# Initialize schema
make setup

# Run complete setup
make quickstart

# Start inference server
make inference

# Clean generated data
make clean

# Run syntax checks
make test
```

## Project Structure

```
featurama/
├── featurama/              # Main package
│   ├── core/              # Feature store core
│   │   └── feature_store.py
│   ├── scylla/            # ScyllaDB integration
│   │   ├── client.py
│   │   └── schema.py
│   ├── data_generation/   # Synthetic data
│   │   └── synthetic_data.py
│   └── ml/                # ML pipeline
│       ├── training.py
│       └── inference.py
├── examples/              # Example scripts
│   ├── 01_setup_scylla.py
│   ├── 02_generate_data.py
│   ├── 03_feature_ingestion.py
│   ├── 04_feature_retrieval.py
│   ├── 05_train_model.py
│   ├── 06_inference.py
│   └── 07_ray_integration.py
├── docker-compose.yml     # ScyllaDB setup
├── requirements.txt       # Python dependencies
├── README.md             # Project overview
├── ARCHITECTURE.md       # Technical architecture
├── Makefile              # Convenience commands
└── quickstart.sh         # Automated setup
```

## Troubleshooting

### ScyllaDB Won't Start

```bash
# Check if port 9042 is already in use
lsof -i :9042

# Remove existing containers and volumes
docker-compose down -v

# Start fresh
docker-compose up -d
```

### Connection Errors

```bash
# Check ScyllaDB health
docker-compose exec scylla nodetool status

# Check logs
docker-compose logs scylla

# Wait longer (ScyllaDB takes 30-60s to be ready)
sleep 30
```

### Import Errors

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+
```

### Out of Memory

ScyllaDB is configured for 2GB RAM. To adjust:

Edit `docker-compose.yml`:
```yaml
command: --smp 1 --memory 4G --overprovisioned 1 --api-address 0.0.0.0
```

## Next Steps

1. **Explore the Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Customize Data Generation**: Modify `featurama/data_generation/synthetic_data.py`
3. **Add New Features**: Extend the feature definitions
4. **Train Custom Models**: Create your own ML pipelines
5. **Scale Up**: Add more ScyllaDB nodes in `docker-compose.yml`
6. **Deploy to Production**: Use ScyllaDB Cloud or Kubernetes

## Resources

- **ScyllaDB**: https://docs.scylladb.com/
- **Ray Data**: https://docs.ray.io/en/latest/data/data.html
- **FastAPI**: https://fastapi.tiangolo.com/
- **XGBoost**: https://xgboost.readthedocs.io/

## Support

Questions or issues?
- Check the examples in `examples/`
- Review the architecture in `ARCHITECTURE.md`
- Examine the code (it's well-documented!)

## License

MIT License - See LICENSE file

---

*"Good news, everyone! You're now ready to build amazing ML features!"* - Professor Farnsworth

