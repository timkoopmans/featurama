"""
Example 2: Generate Synthetic Data

Generate high-cardinality synthetic data for the Featurama universe.
"""

import logging
from featurama.data_generation.synthetic_data import FuturamaDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Generate synthetic data."""
    print("=" * 80)
    print("🎲 Featurama - Synthetic Data Generation")
    print("=" * 80)
    print()

    print('"Good news, everyone! Time to generate some features!"')
    print("                                    - Professor Farnsworth")
    print()

    # Initialize generator
    generator = FuturamaDataGenerator(seed=42)

    # Generate dataset
    print("🏗️  Generating high-cardinality dataset...")
    print()
    print("Configuration:")
    print("  • Characters: 1,000")
    print("  • Planets: 100")
    print("  • Deliveries: 10,000")
    print("  • History: 30 days")
    print("  • Character readings/day: 24")
    print("  • Planet readings/day: 4")
    print("  • Delivery readings: 10")
    print()

    entities_df, features_df = generator.generate_all_features(
        num_characters=1000,
        num_planets=100,
        num_deliveries=10000,
        days_of_history=30,
        character_readings_per_day=24,
        planet_readings_per_day=4,
        delivery_readings=10
    )

    print()
    print("=" * 80)
    print("📊 Generation Summary")
    print("=" * 80)
    print()

    # Entity summary
    print("Entities Generated:")
    entity_counts = entities_df['entity_type'].value_counts()
    for entity_type, count in entity_counts.items():
        print(f"  • {entity_type.capitalize()}: {count:,}")
    print(f"  • Total: {len(entities_df):,}")
    print()

    # Feature summary
    print("Features Generated:")
    feature_counts = features_df['feature_name'].value_counts()
    print(f"  • Unique feature types: {len(feature_counts)}")
    print(f"  • Total feature values: {len(features_df):,}")
    print()

    print("Top 10 Feature Types:")
    for feature_name, count in feature_counts.head(10).items():
        print(f"  • {feature_name}: {count:,}")
    print()

    # Save to CSV for inspection
    print("💾 Saving to CSV files...")
    entities_df.to_csv('data/entities.csv', index=False)
    features_df.to_csv('data/features.csv', index=False)
    print("  ✅ data/entities.csv")
    print("  ✅ data/features.csv")
    print()

    # Sample data preview
    print("🔍 Sample Entity:")
    print(entities_df[entities_df['entity_type'] == 'character'].head(1).to_string())
    print()

    print("🔍 Sample Features:")
    print(features_df.head(3).to_string())
    print()

    print("=" * 80)
    print("✨ Data generation complete!")
    print("=" * 80)
    print()
    print("Next step:")
    print("  Run: python examples/03_feature_ingestion.py")
    print()

    return 0


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    exit(main())

