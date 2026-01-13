"""
COPY-PASTE READY EXAMPLE
========================

Run this code directly to query features from Featurama!
"""

from featurama.core.feature_store import FeatureStore

# Initialize and connect
fs = FeatureStore()
fs.connect()

# =============================================================================
# EXAMPLE 1: Simple Query
# =============================================================================
print("Example 1: Query 3 deliveries, 3 features each")
print("-" * 60)

result = fs.get_online_features(
    entity_ids=['delivery_00000000', 'delivery_00000001', 'delivery_00000002'],
    feature_names=['distance', 'fuel_consumption', 'package_weight']
)

print(result)
print(f"\n✅ Retrieved {len(result)} values\n")

# =============================================================================
# EXAMPLE 2: Batch Query (10 deliveries)
# =============================================================================
print("Example 2: Batch query 10 deliveries")
print("-" * 60)

entity_ids = [f'delivery_{i:08d}' for i in range(10)]
feature_names = ['distance', 'fuel_consumption']

result = fs.get_online_features(entity_ids, feature_names)

# Reshape to matrix format
pivot = result.pivot(index='entity_id', columns='feature_name', values='value')
print(pivot)
print(f"\n✅ Retrieved {len(result)} values for {len(entity_ids)} entities\n")

# =============================================================================
# EXAMPLE 3: Single Entity, All Features
# =============================================================================
print("Example 3: All delivery features for one entity")
print("-" * 60)

result = fs.get_online_features(
    entity_ids=['delivery_00000000'],
    feature_names=[
        'distance',
        'fuel_consumption',
        'package_weight',
        'estimated_duration',
        'actual_duration',
        'delay_minutes',
        'hazard_level'
    ]
)

for _, row in result.iterrows():
    print(f"  {row['feature_name']:<25} = {row['value']:>15.2f}")

print(f"\n✅ Retrieved {len(result)} features\n")

# Clean up
fs.disconnect()

print("=" * 60)
print("✨ Done! Modify the entity_ids and feature_names above")
print("   to query different data.")
print("=" * 60)

