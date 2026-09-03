"""
Example 3: Feature Ingestion

Ingest generated features into ScyllaDB feature store.
"""

import logging
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from featurama.core.feature_store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Ingest features into the feature store."""
    print("=" * 80)
    print("📥 Featurama - Feature Ingestion")
    print("=" * 80)
    print()

    print('"Shut up and take my features!" - Bender')
    print()

    # Load generated data
    print("📂 Loading generated data...")
    try:
        entities_df = pd.read_csv('data/entities.csv')
        features_df = pd.read_csv('data/features.csv')
        print(f"  ✅ Loaded {len(entities_df):,} entities")
        print(f"  ✅ Loaded {len(features_df):,} features")
    except FileNotFoundError:
        print("❌ Data files not found!")
        print("Run: python examples/02_generate_data.py")
        return 1

    print()

    # Connect to feature store
    print("🔌 Connecting to Featurama feature store...")
    fs = FeatureStore()
    fs.connect()
    print("  ✅ Connected!")
    print()

    # Register entities
    print("📝 Registering entities...")
    entity_types = entities_df['entity_type'].unique()

    for entity_type in entity_types:
        type_entities = entities_df[entities_df['entity_type'] == entity_type]

        print(f"  Registering {len(type_entities):,} {entity_type} entities...")
        for _, entity in tqdm(type_entities.iterrows(), total=len(type_entities), desc=f"  {entity_type}"):
            try:
                metadata = eval(entity['metadata']) if isinstance(entity['metadata'], str) else entity['metadata']
                fs.register_entity(
                    entity_id=entity['entity_id'],
                    entity_type=entity['entity_type'],
                    name=entity['name'],
                    metadata=metadata
                )
            except Exception as e:
                logger.debug(f"Entity registration error: {e}")

    print()

    # Register feature definitions
    print("📋 Registering feature definitions...")
    unique_features = features_df['feature_name'].unique()

    feature_types = {
        'delivery_count': 'int',
        'success_rate': 'float',
        'distance_traveled': 'float',
        'efficiency_score': 'float',
        'energy_level': 'float',
        'customer_rating': 'float',
        'incoming_deliveries': 'int',
        'outgoing_deliveries': 'int',
        'traffic_level': 'float',
        'weather_index': 'float',
        'population_density': 'float',
        'estimated_duration': 'float',
        'actual_duration': 'float',
        'distance': 'float',
        'fuel_consumption': 'float',
        'delay_minutes': 'float',
        'hazard_level': 'float',
        'package_weight': 'float'
    }

    for feature_name in unique_features:
        feature_type = feature_types.get(feature_name, 'float')
        try:
            fs.register_feature(
                feature_name=feature_name,
                feature_type=feature_type,
                description=f"Auto-generated feature: {feature_name}",
                version=1,
                tags={'source': 'synthetic', 'generator': 'featurama'}
            )
            print(f"  ✅ {feature_name} ({feature_type})")
        except Exception as e:
            logger.debug(f"Feature registration error: {e}")

    print()

    # Ingest features in batches
    print("💉 Ingesting feature values...")
    print(f"  Total features to ingest: {len(features_df):,}")
    print()

    # Convert timestamp strings to datetime if needed
    if 'timestamp' in features_df.columns and features_df['timestamp'].dtype == 'object':
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    # Batch write
    batch_size = 1000
    num_batches = (len(features_df) + batch_size - 1) // batch_size

    start_time = datetime.now()

    for i in tqdm(range(0, len(features_df), batch_size), desc="  Batches", total=num_batches):
        batch = features_df.iloc[i:i + batch_size]
        fs.write_features(batch, version=1, batch_size=500)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print("📊 Ingestion Summary")
    print("=" * 80)
    print()
    print(f"  • Entities registered: {len(entities_df):,}")
    print(f"  • Feature types registered: {len(unique_features)}")
    print(f"  • Feature values ingested: {len(features_df):,}")
    print(f"  • Duration: {duration:.2f} seconds")
    print(f"  • Throughput: {len(features_df) / duration:.0f} features/second")
    print()

    # Verify ingestion
    print("🔍 Verifying ingestion...")
    # Sample both from the same row, so the entity actually owns the feature
    sample_row = features_df.iloc[0]
    sample_entity = sample_row['entity_id']
    sample_feature = sample_row['feature_name']

    result = fs.get_online_features(
        entity_ids=[sample_entity],
        feature_names=[sample_feature]
    )

    if not result.empty:
        print(f"  ✅ Successfully retrieved feature '{sample_feature}' for entity '{sample_entity}'")
        print(f"     Value: {result.iloc[0]['value']}")
    else:
        print("  ⚠️  Could not verify - might need more time for consistency")

    print()

    fs.disconnect()

    print("=" * 80)
    print("✨ Feature ingestion complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run: python examples/04_feature_retrieval.py")
    print("  2. Run: python examples/05_train_model.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())

