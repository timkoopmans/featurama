"""
Bonus Example: Ray Data Integration

Demonstrate distributed data generation and processing with Ray Data.
"""

import logging
import pandas as pd
import ray
from ray import data
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_character_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for a batch of characters using Ray.

    This function is executed in parallel across Ray workers.
    """
    import random
    import numpy as np
    from datetime import datetime, timedelta

    features = []
    feature_names = ['delivery_count', 'success_rate', 'efficiency_score']

    for _, entity in batch.iterrows():
        entity_id = entity['entity_id']

        # Generate 24 hours of data
        start_time = datetime.now() - timedelta(days=1)

        for hour in range(24):
            timestamp = start_time + timedelta(hours=hour)

            # Generate correlated features
            delivery_count = int(random.uniform(5, 20))
            success_rate = random.uniform(0.8, 0.99)
            efficiency = success_rate * random.uniform(0.7, 0.95)

            values = [delivery_count, success_rate, efficiency]

            for feature_name, value in zip(feature_names, values):
                features.append({
                    'entity_id': entity_id,
                    'feature_name': feature_name,
                    'value': value,
                    'timestamp': timestamp
                })

    return pd.DataFrame(features)


def main():
    """Demonstrate Ray Data integration."""
    print("=" * 80)
    print("⚡ Featurama - Ray Data Integration")
    print("=" * 80)
    print()

    print('"I\'m going to parallelize my own features!"')
    print("                                  - Bender")
    print()

    # Initialize Ray
    print("🚀 Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    print(f"  ✅ Ray initialized")
    print(f"  Dashboard: {ray.get_webui_url()}")
    print()

    # Load entity data
    print("📂 Loading entity data...")
    try:
        entities_df = pd.read_csv('data/entities.csv')
        character_entities = entities_df[entities_df['entity_type'] == 'character'].head(100)
        print(f"  ✅ Loaded {len(character_entities)} characters for processing")
    except FileNotFoundError:
        print("❌ Data files not found!")
        print("Run: python examples/02_generate_data.py")
        ray.shutdown()
        return 1

    print()

    # Create Ray Dataset from pandas DataFrame
    print("🔄 Converting to Ray Dataset...")
    ds = ray.data.from_pandas(character_entities)
    print(f"  ✅ Created Ray Dataset with {ds.count()} rows")
    print()

    # Process in parallel using Ray
    print("⚡ Generating features in parallel with Ray...")
    print(f"  Processing {len(character_entities)} entities...")
    print()

    start_time = datetime.now()

    # Use map_batches for parallel processing
    feature_ds = ds.map_batches(
        generate_character_batch,
        batch_format="pandas",
        batch_size=10
    )

    # Collect results
    features_df = feature_ds.to_pandas()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"  ✅ Generated {len(features_df):,} features")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Throughput: {len(features_df) / duration:.0f} features/second")
    print()

    # Show statistics
    print("=" * 80)
    print("📊 Processing Statistics")
    print("=" * 80)
    print()

    print("Feature Distribution:")
    print(features_df['feature_name'].value_counts().to_string())
    print()

    print("Sample Features:")
    print(features_df.head(10).to_string())
    print()

    # Demonstrate aggregations with Ray
    print("=" * 80)
    print("📈 Ray Data Aggregations")
    print("=" * 80)
    print()

    print("Computing feature statistics with Ray...")

    # Convert back to Ray Dataset for aggregations
    agg_ds = ray.data.from_pandas(features_df)

    # Group by feature and compute stats
    print()
    print("Average values by feature type:")

    for feature_name in features_df['feature_name'].unique():
        feature_data = features_df[features_df['feature_name'] == feature_name]['value']
        print(f"  • {feature_name}:")
        print(f"      Mean: {feature_data.mean():.4f}")
        print(f"      Std:  {feature_data.std():.4f}")
        print(f"      Min:  {feature_data.min():.4f}")
        print(f"      Max:  {feature_data.max():.4f}")

    print()

    # Demonstrate writing to different formats
    print("=" * 80)
    print("💾 Ray Data Export Capabilities")
    print("=" * 80)
    print()

    print("Ray Data supports exporting to:")
    print("  • Parquet (columnar format)")
    print("  • CSV")
    print("  • JSON")
    print("  • NumPy")
    print("  • Arrow")
    print()

    # Save as parquet
    print("Saving to Parquet format...")
    output_path = "data/ray_features"
    feature_ds.write_parquet(output_path)
    print(f"  ✅ Saved to {output_path}/")
    print()

    # Demonstrate reading back
    print("Reading back from Parquet...")
    loaded_ds = ray.data.read_parquet(output_path)
    print(f"  ✅ Loaded {loaded_ds.count()} rows")
    print()

    # Show Ray Dataset schema
    print("Dataset Schema:")
    print(loaded_ds.schema())
    print()

    # Demonstrate filtering and transformations
    print("=" * 80)
    print("🔍 Ray Data Transformations")
    print("=" * 80)
    print()

    print("Filtering high-performing features (success_rate > 0.9)...")

    def filter_high_performance(batch: pd.DataFrame) -> pd.DataFrame:
        return batch[
            (batch['feature_name'] == 'success_rate') &
            (batch['value'] > 0.9)
        ]

    high_perf_ds = loaded_ds.map_batches(
        filter_high_performance,
        batch_format="pandas"
    )

    high_perf_count = high_perf_ds.count()
    print(f"  ✅ Found {high_perf_count} high-performance records")
    print()

    if high_perf_count > 0:
        print("Sample high-performance records:")
        sample = high_perf_ds.take(5)
        for record in sample:
            print(f"  • {record['entity_id']}: {record['value']:.4f} at {record['timestamp']}")
        print()

    # Performance comparison
    print("=" * 80)
    print("⚡ Ray vs Pandas Performance Comparison")
    print("=" * 80)
    print()

    print("Benefits of Ray Data:")
    print("  ✅ Distributed processing across multiple cores/machines")
    print("  ✅ Lazy evaluation for memory efficiency")
    print("  ✅ Automatic data partitioning and load balancing")
    print("  ✅ Built-in support for large datasets (> memory)")
    print("  ✅ Seamless integration with Ray ML libraries")
    print()

    print("Use Ray Data when:")
    print("  • Dataset doesn't fit in memory")
    print("  • Processing requires heavy computation")
    print("  • Need to scale across multiple machines")
    print("  • Working with ML training pipelines")
    print()

    # Cleanup
    print("🧹 Cleaning up...")
    ray.shutdown()
    print("  ✅ Ray shutdown complete")
    print()

    print("=" * 80)
    print("✨ Ray Data demonstration complete!")
    print("=" * 80)
    print()

    print("Key Takeaways:")
    print("  • Ray Data enables distributed feature generation")
    print("  • Seamlessly integrates with pandas DataFrames")
    print("  • Supports multiple data formats (Parquet, CSV, etc.)")
    print("  • Perfect for scaling feature engineering pipelines")
    print()

    print("For production use:")
    print("  • Use Ray clusters for multi-node processing")
    print("  • Leverage Ray's fault tolerance for long-running jobs")
    print("  • Integrate with Ray Serve for model serving at scale")
    print()

    return 0


if __name__ == "__main__":
    exit(main())

