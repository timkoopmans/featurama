"""
Example 4: Feature Retrieval Benchmarks

Benchmark feature retrieval performance from ScyllaDB.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List

from featurama.core.feature_store import FeatureStore

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def benchmark_online_features(
    fs: FeatureStore,
    entity_ids: List[str],
    feature_names: List[str],
    num_queries: int = 100
):
    """Benchmark online feature retrieval."""
    print("🚀 Benchmarking online feature retrieval...")
    print(f"   Queries: {num_queries}")
    print(f"   Features per query: {len(feature_names)}")
    print()

    latencies = []

    for i in range(num_queries):
        # Sample random entities
        sample_entities = np.random.choice(entity_ids, size=min(5, len(entity_ids)), replace=False)

        start = time.time()
        result = fs.get_online_features(
            entity_ids=sample_entities.tolist(),
            feature_names=feature_names
        )
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i + 1}/{num_queries}", end='\r')

    print()

    latencies = np.array(latencies)

    print(f"   ✅ Completed {num_queries} queries")
    print(f"   Average latency: {latencies.mean():.2f} ms")
    print(f"   Median latency: {np.median(latencies):.2f} ms")
    print(f"   P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"   P99 latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"   Min latency: {latencies.min():.2f} ms")
    print(f"   Max latency: {latencies.max():.2f} ms")
    print()

    return latencies


def benchmark_historical_features(
    fs: FeatureStore,
    entity_ids: List[str],
    feature_names: List[str],
    num_queries: int = 50
):
    """Benchmark historical (point-in-time) feature retrieval."""
    print("⏰ Benchmarking historical feature retrieval...")
    print(f"   Queries: {num_queries}")
    print()

    latencies = []

    for i in range(num_queries):
        # Sample random entities and timestamp
        sample_entities = np.random.choice(entity_ids, size=min(3, len(entity_ids)), replace=False)
        pit_timestamp = datetime.now() - timedelta(days=np.random.randint(1, 30))

        start = time.time()
        result = fs.get_historical_features(
            entity_ids=sample_entities.tolist(),
            feature_names=feature_names,
            timestamp=pit_timestamp
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{num_queries}", end='\r')

    print()

    latencies = np.array(latencies)

    print(f"   ✅ Completed {num_queries} queries")
    print(f"   Average latency: {latencies.mean():.2f} ms")
    print(f"   Median latency: {np.median(latencies):.2f} ms")
    print(f"   P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print()

    return latencies


def benchmark_feature_history(
    fs: FeatureStore,
    entity_ids: List[str],
    feature_names: List[str],
    num_queries: int = 30
):
    """Benchmark time-series feature history retrieval."""
    print("📈 Benchmarking time-series history retrieval...")
    print(f"   Queries: {num_queries}")
    print()

    latencies = []
    row_counts = []

    for i in range(num_queries):
        # Sample random entity and feature
        entity = np.random.choice(entity_ids)
        feature = np.random.choice(feature_names)

        start = time.time()
        result = fs.get_feature_history(
            entity_id=entity,
            feature_name=feature,
            limit=100
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        row_counts.append(len(result))

        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{num_queries}", end='\r')

    print()

    latencies = np.array(latencies)
    row_counts = np.array(row_counts)

    print(f"   ✅ Completed {num_queries} queries")
    print(f"   Average latency: {latencies.mean():.2f} ms")
    print(f"   Average rows returned: {row_counts.mean():.1f}")
    print(f"   P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print()

    return latencies


def main():
    """Run feature retrieval benchmarks."""
    print("=" * 80)
    print("⚡ Featurama - Feature Retrieval Benchmarks")
    print("=" * 80)
    print()

    print('"I\'m going to build my own feature store! With blackjack!"')
    print("                                           - Bender")
    print()

    # Load entity data
    print("📂 Loading entity data...")
    try:
        entities_df = pd.read_csv('data/entities.csv')
        features_df = pd.read_csv('data/features.csv')
    except FileNotFoundError:
        print("❌ Data files not found!")
        print("Run the previous examples first.")
        return 1

    print(f"  ✅ Loaded {len(entities_df):,} entities")
    print()

    # Connect to feature store
    print("🔌 Connecting to feature store...")
    fs = FeatureStore()
    fs.connect()
    print("  ✅ Connected!")
    print()

    # Get sample entities by type
    character_ids = entities_df[entities_df['entity_type'] == 'character']['entity_id'].tolist()[:100]
    planet_ids = entities_df[entities_df['entity_type'] == 'planet']['entity_id'].tolist()[:50]
    delivery_ids = entities_df[entities_df['entity_type'] == 'delivery']['entity_id'].tolist()[:100]

    # Common features
    character_features = ['delivery_count', 'success_rate', 'efficiency_score']
    planet_features = ['incoming_deliveries', 'traffic_level', 'weather_index']
    delivery_features = ['distance', 'estimated_duration', 'hazard_level']

    print("=" * 80)
    print("🎯 Benchmark 1: Character Online Features")
    print("=" * 80)
    print()

    char_latencies = benchmark_online_features(
        fs, character_ids, character_features, num_queries=100
    )

    print("=" * 80)
    print("🌍 Benchmark 2: Planet Online Features")
    print("=" * 80)
    print()

    planet_latencies = benchmark_online_features(
        fs, planet_ids, planet_features, num_queries=100
    )

    print("=" * 80)
    print("📦 Benchmark 3: Delivery Features")
    print("=" * 80)
    print()

    delivery_latencies = benchmark_online_features(
        fs, delivery_ids, delivery_features, num_queries=100
    )

    print("=" * 80)
    print("⏰ Benchmark 4: Historical Features (Point-in-Time)")
    print("=" * 80)
    print()

    hist_latencies = benchmark_historical_features(
        fs, character_ids, character_features, num_queries=50
    )

    print("=" * 80)
    print("📈 Benchmark 5: Time-Series History")
    print("=" * 80)
    print()

    ts_latencies = benchmark_feature_history(
        fs, character_ids, character_features, num_queries=30
    )

    # Overall summary
    print("=" * 80)
    print("📊 Overall Performance Summary")
    print("=" * 80)
    print()

    all_latencies = np.concatenate([
        char_latencies, planet_latencies, delivery_latencies
    ])

    print(f"Online Feature Retrieval:")
    print(f"  • Total queries: {len(all_latencies)}")
    print(f"  • Average latency: {all_latencies.mean():.2f} ms")
    print(f"  • Median latency: {np.median(all_latencies):.2f} ms")
    print(f"  • P95 latency: {np.percentile(all_latencies, 95):.2f} ms")
    print(f"  • P99 latency: {np.percentile(all_latencies, 99):.2f} ms")
    print()

    print(f"Historical Feature Retrieval:")
    print(f"  • Average latency: {hist_latencies.mean():.2f} ms")
    print(f"  • P95 latency: {np.percentile(hist_latencies, 95):.2f} ms")
    print()

    print(f"Time-Series History:")
    print(f"  • Average latency: {ts_latencies.mean():.2f} ms")
    print(f"  • P95 latency: {np.percentile(ts_latencies, 95):.2f} ms")
    print()

    # List registered features
    print("📋 Registered Features:")
    feature_metadata = fs.list_features()
    print(f"  • Total: {len(feature_metadata)}")
    print()
    print(feature_metadata[['feature_name', 'feature_type', 'version']].to_string(index=False))
    print()

    fs.disconnect()

    print("=" * 80)
    print("✨ Benchmarks complete!")
    print("=" * 80)
    print()
    print("Next step:")
    print("  Run: python examples/05_train_model.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())

