"""
ScyllaDB schema definitions for Featurama.

Optimized schema design for high-cardinality feature storage:
- feature_metadata: Feature definitions and versions
- feature_values: Time-series feature data partitioned by entity_id
- entity_registry: Entity catalog and metadata
"""

KEYSPACE_NAME = "featurama"

# Feature metadata table - stores feature definitions
CREATE_FEATURE_METADATA_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.feature_metadata (
        feature_name TEXT,
        version INT,
        feature_type TEXT,
        description TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        tags MAP<TEXT, TEXT>,
        PRIMARY KEY (feature_name, version)
    ) WITH CLUSTERING ORDER BY (version DESC)
"""

# Feature values table - optimized for time-series queries
# Partition by entity_id for efficient single-entity lookups
# Cluster by feature_name and timestamp for range queries
CREATE_FEATURE_VALUES_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.feature_values (
        entity_id TEXT,
        feature_name TEXT,
        timestamp TIMESTAMP,
        value_double DOUBLE,
        value_text TEXT,
        value_int BIGINT,
        value_bool BOOLEAN,
        version INT,
        PRIMARY KEY ((entity_id), feature_name, timestamp)
    ) WITH CLUSTERING ORDER BY (feature_name ASC, timestamp DESC)
    AND compaction = {{'class': 'TimeWindowCompactionStrategy', 'compaction_window_size': '2', 'compaction_window_unit': 'DAYS'}}
    AND default_time_to_live = 2592000
"""

# Feature values by feature name - for batch queries across entities
CREATE_FEATURE_VALUES_BY_NAME_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.feature_values_by_name (
        feature_name TEXT,
        entity_id TEXT,
        timestamp TIMESTAMP,
        value_double DOUBLE,
        value_text TEXT,
        value_int BIGINT,
        value_bool BOOLEAN,
        version INT,
        PRIMARY KEY ((feature_name), entity_id, timestamp)
    ) WITH CLUSTERING ORDER BY (entity_id ASC, timestamp DESC)
    AND compaction = {{'class': 'TimeWindowCompactionStrategy', 'compaction_window_size': '2', 'compaction_window_unit': 'DAYS'}}
    AND default_time_to_live = 2592000
"""

# Entity registry - catalog of all entities
CREATE_ENTITY_REGISTRY_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.entity_registry (
        entity_id TEXT,
        entity_type TEXT,
        name TEXT,
        metadata MAP<TEXT, TEXT>,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        PRIMARY KEY (entity_id)
    )
"""

# Entity types index - for querying entities by type
CREATE_ENTITY_BY_TYPE_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}.entity_by_type (
        entity_type TEXT,
        entity_id TEXT,
        name TEXT,
        created_at TIMESTAMP,
        PRIMARY KEY ((entity_type), entity_id)
    ) WITH CLUSTERING ORDER BY (entity_id ASC)
"""

ALL_TABLES = [
    CREATE_FEATURE_METADATA_TABLE,
    CREATE_FEATURE_VALUES_TABLE,
    CREATE_FEATURE_VALUES_BY_NAME_TABLE,
    CREATE_ENTITY_REGISTRY_TABLE,
    CREATE_ENTITY_BY_TYPE_TABLE,
]


def get_schema_statements(replication_factor):
    """Get all schema creation statements."""
    CREATE_KEYSPACE = f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE_NAME}
        WITH replication = {{
            'class': 'NetworkTopologyStrategy',
            'replication_factor': {replication_factor}
        }}
    """
    return [CREATE_KEYSPACE] + ALL_TABLES
