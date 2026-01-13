# Featurama Query Examples - Quick Reference

## 🎯 Basic Query Pattern

```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

# Query pattern:
result = fs.get_online_features(
    entity_ids=['entity_1', 'entity_2'],
    feature_names=['feature_a', 'feature_b']
)

print(result)
fs.disconnect()
```

## 📊 Working Examples with Real Data

### Example 1: Query Delivery Features ✅

```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

# Delivery entity IDs follow format: delivery_########
entity_ids = [
    'delivery_00000000',
    'delivery_00000001', 
    'delivery_00000002'
]

# Available delivery features:
feature_names = [
    'distance',                  # Distance of delivery
    'fuel_consumption',          # Fuel used
    'package_weight',            # Weight of package
    'estimated_duration',        # Estimated time
    'actual_duration',           # Actual time taken
    'delay_minutes',             # Delay if any
    'hazard_level'              # Risk level
]

result = fs.get_online_features(entity_ids, feature_names)
print(result)

fs.disconnect()
```

**Output:**
```
        entity_id     feature_name         value               timestamp
delivery_00000000         distance  31164.899129 2025-11-06 13:28:48.761
delivery_00000000 fuel_consumption 203669.796716 2025-11-06 13:28:48.761
delivery_00000000   package_weight    964.547060 2025-11-06 13:28:48.761
...
```

### Example 2: Query Character Features ✅

```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

# Character entity IDs follow format: char_<name>_<id>
entity_ids = [
    'char_lisa19_757',
    'char_aaronfinley_902',
    'char_aaronhayes_112'
]

# Available character features:
feature_names = [
    'delivery_count',       # Number of deliveries made
    'success_rate',         # Success percentage
    'distance_traveled',    # Total distance
    'efficiency_score',     # Efficiency metric
    'energy_level',         # Current energy
    'customer_rating'       # Average rating
]

result = fs.get_online_features(entity_ids, feature_names)
print(result)

fs.disconnect()
```

### Example 3: Query Planet Features ✅

```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

# Planet entity IDs follow format: planet_<name>_<id>
entity_ids = [
    'planet_agreement_75',
    'planet_all_71',
    'planet_alone_99'
]

# Available planet features:
feature_names = [
    'incoming_deliveries',  # Deliveries arriving
    'outgoing_deliveries',  # Deliveries departing
    'traffic_level',        # Traffic density
    'weather_index',        # Weather conditions
    'population_density'    # Population metric
]

result = fs.get_online_features(entity_ids, feature_names)
print(result)

fs.disconnect()
```

### Example 4: Single Entity, Multiple Features

```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

# Query all features for one delivery
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

# Display as a feature vector
for _, row in result.iterrows():
    print(f"{row['feature_name']:<25} = {row['value']}")

fs.disconnect()
```

### Example 5: Batch Query - Multiple Entities

```python
from featurama.core.feature_store import FeatureStore
import pandas as pd

fs = FeatureStore()
fs.connect()

# Query 10 deliveries
entity_ids = [f'delivery_{i:08d}' for i in range(10)]
feature_names = ['distance', 'fuel_consumption', 'package_weight']

result = fs.get_online_features(entity_ids, feature_names)

# Pivot for better display (rows=entities, columns=features)
pivot = result.pivot(index='entity_id', columns='feature_name', values='value')
print(pivot)

fs.disconnect()
```

**Output:**
```
feature_name           distance  fuel_consumption  package_weight
entity_id                                                        
delivery_00000000  31164.899129     203669.796716      964.547060
delivery_00000001  46067.569482     105847.468941      323.070091
delivery_00000002  43194.605769     285622.106626      746.942878
...
```

## 📋 All Available Features

### Character Features (6 total)
- `delivery_count` - Number of deliveries completed
- `success_rate` - Percentage of successful deliveries
- `distance_traveled` - Total distance traveled
- `efficiency_score` - Overall efficiency metric
- `energy_level` - Current energy level
- `customer_rating` - Average customer rating

### Planet Features (5 total)
- `incoming_deliveries` - Number of incoming deliveries
- `outgoing_deliveries` - Number of outgoing deliveries
- `traffic_level` - Traffic density metric
- `weather_index` - Weather conditions
- `population_density` - Population density

### Delivery Features (7+ total)
- `distance` - Delivery distance
- `fuel_consumption` - Fuel consumed
- `package_weight` - Package weight
- `estimated_duration` - Estimated delivery time
- `actual_duration` - Actual delivery time
- `delay_minutes` - Delay amount
- `hazard_level` - Risk/hazard level

## 🔍 Finding Entity IDs

### List entities by type:
```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

# Get all characters
characters = fs.list_entities(entity_type='character')
print(characters.head(10))

# Get all planets
planets = fs.list_entities(entity_type='planet')
print(planets.head(10))

# Get all deliveries
deliveries = fs.list_entities(entity_type='delivery')
print(deliveries.head(10))

fs.disconnect()
```

### List all available features:
```python
from featurama.core.feature_store import FeatureStore

fs = FeatureStore()
fs.connect()

features = fs.list_features()
print(features[['feature_name', 'feature_type', 'description']])

fs.disconnect()
```

## 📈 Advanced Queries

### Historical (Point-in-Time) Query:
```python
from featurama.core.feature_store import FeatureStore
from datetime import datetime

fs = FeatureStore()
fs.connect()

# Get features as they were at a specific time
result = fs.get_historical_features(
    entity_ids=['delivery_00000000'],
    feature_names=['distance', 'fuel_consumption'],
    timestamp=datetime(2025, 11, 6, 12, 0, 0)
)

print(result)
fs.disconnect()
```

### Time-Series History:
```python
from featurama.core.feature_store import FeatureStore
from datetime import datetime, timedelta

fs = FeatureStore()
fs.connect()

# Get feature history over time
end_time = datetime.now()
start_time = end_time - timedelta(days=30)

history = fs.get_feature_history(
    entity_id='delivery_00000000',
    feature_name='distance',
    start_time=start_time,
    end_time=end_time,
    limit=100
)

print(history)
fs.disconnect()
```

## 🎯 Quick Test Command

Run this to test a simple query:

```bash
cd /Users/timkoopmans/Git/featurama
source .venv/bin/activate
python examples/example_queries.py
```

## 💡 Tips

1. **Entity ID Format:**
   - Characters: `char_<name>_<id>`
   - Planets: `planet_<name>_<id>`
   - Deliveries: `delivery_########`

2. **Return Format:**
   - DataFrame with columns: `entity_id`, `feature_name`, `value`, `timestamp`
   - Use `.pivot()` to reshape into entity × feature matrix

3. **Performance:**
   - ~3ms latency per query
   - ~29,000 features/second ingestion rate
   - Batch queries are efficient

4. **Feature Types:**
   - `int` - Integer values (counts, IDs)
   - `float` - Decimal values (distances, scores)
   - `bool` - Boolean flags
   - `string` - Text values

