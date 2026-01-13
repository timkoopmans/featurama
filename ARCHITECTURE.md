# Featurama Architecture

## Overview

Featurama is a high-performance feature store built on ScyllaDB, designed to demonstrate efficient storage, retrieval, and serving of millions of high-cardinality features for machine learning pipelines.

## System Components

### 1. ScyllaDB Backend

**Purpose**: High-performance, distributed NoSQL database for feature storage

**Key Tables**:

- `feature_metadata`: Feature definitions and versioning
  - Partition key: `feature_name`
  - Clustering key: `version` (DESC)
  - Stores: type, description, tags, timestamps

- `feature_values`: Time-series feature data
  - Partition key: `entity_id`
  - Clustering keys: `feature_name`, `timestamp` (DESC)
  - Stores: typed value columns, version
  - Compaction: TimeWindowCompactionStrategy

- `feature_values_by_name`: Inverted index for batch queries
  - Partition key: `feature_name`
  - Clustering keys: `entity_id`, `timestamp` (DESC)

- `entity_registry`: Entity catalog
  - Partition key: `entity_id`
  - Stores: type, name, metadata

- `entity_by_type`: Entity type index
  - Partition key: `entity_type`
  - Clustering key: `entity_id`

**Design Decisions**:

- **Partition Strategy**: Entity-based partitioning for online serving (single-entity lookups)
- **Clustering**: Feature name + timestamp for efficient time-series queries
- **Dual Writes**: Write to both entity-partitioned and feature-partitioned tables
- **Type Columns**: Separate columns for different value types (double, int, text, bool)
- **Compaction**: Time-window compaction for time-series data optimization

### 2. Feature Store Core

**File**: `featurama/core/feature_store.py`

**Key Methods**:

- `register_feature()`: Register feature definitions with versioning
- `register_entity()`: Add entities to the catalog
- `write_features()`: Batch write feature values
- `get_online_features()`: Latest values for online serving (low latency)
- `get_historical_features()`: Point-in-time correctness
- `get_feature_history()`: Time-series data retrieval

**Capabilities**:

- Pandas DataFrame integration
- Batch writes with configurable batch sizes
- Point-in-time queries for training data
- Online serving for inference
- Feature metadata management

### 3. Data Generation

**File**: `featurama/data_generation/synthetic_data.py`

**Purpose**: Generate realistic high-cardinality synthetic data

**Features**:

- Futurama-themed entities (characters, planets, deliveries)
- Time-series feature generation with realistic patterns
- Configurable cardinality and history depth
- Correlated feature generation
- Seasonal and noise patterns

**Entity Types**:

- **Characters**: Delivery crew with behavioral metrics
- **Planets**: Destinations with traffic and environmental data
- **Deliveries**: Routes with performance metrics

### 4. ML Pipeline

**Training** (`featurama/ml/training.py`):

- XGBoost regression model for delivery time prediction
- Feature engineering (interactions, polynomials)
- Model versioning and persistence
- Performance metrics (MAE, RMSE, R²)

**Inference** (`featurama/ml/inference.py`):

- FastAPI REST API
- Real-time predictions using feature store
- Batch prediction support
- Health checks and monitoring
- Swagger/OpenAPI documentation

**Endpoints**:

- `POST /predict`: Single prediction
- `POST /features/batch`: Batch predictions
- `GET /features/{entity_id}`: Feature retrieval
- `GET /health`: Health check

### 5. Ray Data Integration

**File**: `examples/07_ray_integration.py`

**Purpose**: Distributed data processing at scale

**Capabilities**:

- Parallel feature generation across workers
- Lazy evaluation for memory efficiency
- Multiple format support (Parquet, CSV, Arrow)
- Seamless pandas integration
- Scalable to multi-node clusters

## Data Flow

### Write Path

```
Data Generation → Pandas DataFrame → Feature Store → ScyllaDB
                                          ↓
                                    Batch Writes
                                          ↓
                              [feature_values table]
                              [feature_values_by_name table]
```

### Online Serving Path

```
Inference Request → Feature Store → ScyllaDB Query → Latest Values
                                         ↓
                                   ML Model → Prediction
```

### Training Path

```
Historical Data Request → Feature Store → Point-in-Time Query
                              ↓
                        Training Dataset → Model Training
```

## Performance Characteristics

### Read Performance

- **Online Serving**: Sub-millisecond latency for single-entity lookups
- **Batch Queries**: Parallel execution across partitions
- **Time-Series**: Efficient clustering key scans

### Write Performance

- **Batch Writes**: Configurable batch sizes (500-1000)
- **Throughput**: Thousands of features per second
- **Dual Writes**: Maintained atomicity with batch statements

### Scalability

- **Horizontal**: ScyllaDB scales linearly with nodes
- **Data Volume**: Handles billions of feature values
- **Cardinality**: Supports millions of unique features

## Schema Design Patterns

### 1. Denormalization

- Dual writes to entity-partitioned and feature-partitioned tables
- Trade-off: Write amplification for read optimization

### 2. Time-Series Optimization

- Clustering by timestamp (DESC) for latest-first retrieval
- Time-window compaction for automatic data lifecycle
- Efficient range queries

### 3. Type Flexibility

- Multiple value columns for different data types
- Allows mixed-type features without complex serialization
- Efficient storage and querying

### 4. Versioning

- Feature version tracking in metadata
- Version field in feature values
- Enables feature evolution and A/B testing

## Deployment Architecture

### Development

```
Docker Compose (Single Node)
    ↓
ScyllaDB (1 node, 2GB RAM)
    ↓
Feature Store API
    ↓
ML Inference Server
```

### Production (Recommended)

```
Load Balancer
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Inference 1 │ Inference 2 │ Inference N │
└─────────────┴─────────────┴─────────────┘
         ↓             ↓             ↓
    ┌────────────────────────────────────┐
    │      Feature Store API Layer       │
    └────────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │   ScyllaDB Cluster (3+ nodes)       │
    │   - Replication Factor: 3           │
    │   - Consistency: QUORUM             │
    └─────────────────────────────────────┘
```

## Technology Stack

- **Database**: ScyllaDB 5.4
- **Language**: Python 3.13
- **Data Processing**: Pandas, Ray Data
- **ML Framework**: Scikit-learn, XGBoost
- **API Framework**: FastAPI, Uvicorn
- **Orchestration**: Docker Compose
- **Data Format**: Parquet, CSV

## Future Enhancements

1. **Streaming Ingestion**: Kafka/Kinesis integration
2. **Feature Serving Cache**: Redis/Memcached layer
3. **Feature Lineage**: DAG tracking and visualization
4. **Feature Store UI**: Web interface for exploration
5. **Multi-Tenancy**: Namespace isolation
6. **Feature Monitoring**: Data quality checks and alerting
7. **AutoML Integration**: Feature selection and engineering
8. **Cross-DC Replication**: Multi-region deployment

## Performance Tuning

### ScyllaDB

- Adjust `--smp` for CPU cores
- Tune `--memory` based on available RAM
- Configure compaction strategies per use case
- Monitor with Prometheus/Grafana

### Feature Store

- Batch size tuning for write throughput
- Connection pooling for concurrent queries
- Async I/O for improved concurrency
- Caching layer for hot features

### ML Inference

- Model quantization for faster predictions
- Feature pre-computation for common queries
- Horizontal scaling with load balancing
- GPU acceleration for complex models

## Monitoring & Observability

**Key Metrics**:

- Read/write latency (p50, p95, p99)
- Throughput (features/second)
- Error rates
- Cache hit rates
- Model prediction latency
- Feature freshness

**Tools**:

- ScyllaDB metrics (via Prometheus API)
- Application logs (structured JSON)
- Distributed tracing (OpenTelemetry)
- Custom dashboards (Grafana)

## Security Considerations

1. **Authentication**: ScyllaDB user/password
2. **Encryption**: TLS for client-server communication
3. **Authorization**: Role-based access control
4. **API Security**: JWT tokens, rate limiting
5. **Network**: VPC isolation, firewall rules
6. **Audit Logging**: Feature access tracking

## References

- [ScyllaDB Documentation](https://docs.scylladb.com/)
- [Ray Data Guide](https://docs.ray.io/en/latest/data/data.html)
- [Feature Store Concepts](https://www.featurestore.org/)

