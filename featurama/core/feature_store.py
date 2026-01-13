"""
Core Feature Store implementation.

Provides high-level API for feature management, registration,
writing, and retrieval from ScyllaDB.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging

from featurama.scylla.client import ScyllaClient
from featurama.scylla.schema import KEYSPACE_NAME

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    High-performance feature store backed by ScyllaDB.

    Supports:
    - Feature registration with versioning
    - Batch and streaming feature writes
    - Point-in-time feature retrieval
    - Online feature serving
    - Pandas and Ray Data integration
    """

    def __init__(
        self,
        contact_points: List[str] = None,
        port: int = 9042,
        keyspace: str = KEYSPACE_NAME
    ):
        """
        Initialize Feature Store.

        Args:
            contact_points: ScyllaDB node addresses
            port: CQL port
            keyspace: Keyspace name
        """
        self.client = ScyllaClient(contact_points, port, keyspace)
        self.keyspace = keyspace
        self._connected = False

    def connect(self):
        """Connect to ScyllaDB."""
        if not self._connected:
            self.client.connect()
            self.client.session.set_keyspace(self.keyspace)
            self._connected = True
            logger.info("Feature Store connected")

    def disconnect(self):
        """Disconnect from ScyllaDB."""
        if self._connected:
            self.client.disconnect()
            self._connected = False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def register_feature(
        self,
        feature_name: str,
        feature_type: str,
        description: str = "",
        version: int = 1,
        tags: Dict[str, str] = None
    ):
        """
        Register a new feature or feature version.

        Args:
            feature_name: Unique feature identifier
            feature_type: Data type (int, float, string, bool)
            description: Feature description
            version: Feature version
            tags: Additional metadata tags
        """
        self.connect()

        now = datetime.now()
        tags = tags or {}

        query_str = f"""
            INSERT INTO {self.keyspace}.feature_metadata
            (feature_name, version, feature_type, description, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        prepared = self.client.session.prepare(query_str)
        self.client.session.execute(prepared, (feature_name, version, feature_type, description, now, now, tags))

        logger.info(f"Registered feature: {feature_name} v{version} ({feature_type})")

    def register_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        metadata: Dict[str, str] = None
    ):
        """
        Register an entity in the catalog.

        Args:
            entity_id: Unique entity identifier
            entity_type: Entity type (e.g., 'character', 'planet', 'delivery')
            name: Human-readable name
            metadata: Additional metadata
        """
        self.connect()

        now = datetime.now()
        metadata = metadata or {}

        # Insert into entity registry
        query1_str = f"""
            INSERT INTO {self.keyspace}.entity_registry
            (entity_id, entity_type, name, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        prepared1 = self.client.session.prepare(query1_str)
        self.client.session.execute(prepared1, (entity_id, entity_type, name, metadata, now, now))

        # Insert into entity_by_type index
        query2_str = f"""
            INSERT INTO {self.keyspace}.entity_by_type
            (entity_type, entity_id, name, created_at)
            VALUES (?, ?, ?, ?)
        """
        prepared2 = self.client.session.prepare(query2_str)
        self.client.session.execute(prepared2, (entity_type, entity_id, name, now))

        logger.debug(f"Registered entity: {entity_id} ({entity_type})")

    def write_features(
        self,
        features_df: pd.DataFrame,
        version: int = 1,
        batch_size: int = 1000
    ):
        """
        Write features to the feature store.

        Expected DataFrame columns:
        - entity_id: Entity identifier
        - feature_name: Feature name
        - value: Feature value
        - timestamp: Feature timestamp (optional, defaults to now)

        Args:
            features_df: DataFrame with features
            version: Feature version
            batch_size: Batch size for writes
        """
        self.connect()

        if 'timestamp' not in features_df.columns:
            features_df['timestamp'] = datetime.now()

        logger.info(f"Writing {len(features_df)} feature values...")

        # Prepare batched inserts
        from cassandra.query import BatchStatement

        insert_query_str = f"""
            INSERT INTO {self.keyspace}.feature_values
            (entity_id, feature_name, timestamp, value_double, value_text, value_int, value_bool, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        insert_query_by_name_str = f"""
            INSERT INTO {self.keyspace}.feature_values_by_name
            (feature_name, entity_id, timestamp, value_double, value_text, value_int, value_bool, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Prepare statements once
        insert_stmt = self.client.session.prepare(insert_query_str)
        insert_stmt_by_name = self.client.session.prepare(insert_query_by_name_str)

        batch = BatchStatement()
        count = 0

        for _, row in features_df.iterrows():
            entity_id = row['entity_id']
            feature_name = row['feature_name']
            timestamp = row['timestamp']
            value = row['value']

            # Type-based value columns - check bool first since bool is subclass of int
            value_double = None
            value_text = None
            value_int = None
            value_bool = None

            if isinstance(value, bool):
                value_bool = value
            elif isinstance(value, int):
                value_int = value
            elif isinstance(value, float):
                value_double = value
            elif isinstance(value, str):
                value_text = value
            else:
                value_text = str(value)

            params = (entity_id, feature_name, timestamp, value_double, value_text, value_int, value_bool, version)
            params_by_name = (feature_name, entity_id, timestamp, value_double, value_text, value_int, value_bool, version)

            batch.add(insert_stmt, params)
            batch.add(insert_stmt_by_name, params_by_name)

            count += 1

            if count % batch_size == 0:
                self.client.session.execute(batch)
                batch = BatchStatement()
                logger.info(f"Written {count} features...")

        # Execute remaining batch
        if count % batch_size != 0:
            self.client.session.execute(batch)

        logger.info(f"Successfully wrote {count} feature values")

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get latest feature values for entities (online serving).

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names

        Returns:
            DataFrame with latest feature values
        """
        self.connect()

        results = []

        query_str = f"""
            SELECT entity_id, feature_name, timestamp, 
                   value_double, value_text, value_int, value_bool
            FROM {self.keyspace}.feature_values
            WHERE entity_id = ? AND feature_name = ?
            LIMIT 1
        """

        # Prepare the statement
        prepared = self.client.session.prepare(query_str)

        for entity_id in entity_ids:
            for feature_name in feature_names:
                result = self.client.session.execute(prepared, (entity_id, feature_name))
                row = result.one()

                if row:
                    value = self._extract_value(row)
                    results.append({
                        'entity_id': entity_id,
                        'feature_name': feature_name,
                        'value': value,
                        'timestamp': row['timestamp']
                    })

        return pd.DataFrame(results)

    def get_historical_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Get feature values at a specific point in time.

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            timestamp: Point-in-time timestamp

        Returns:
            DataFrame with historical feature values
        """
        self.connect()

        results = []

        query_str = f"""
            SELECT entity_id, feature_name, timestamp,
                   value_double, value_text, value_int, value_bool
            FROM {self.keyspace}.feature_values
            WHERE entity_id = ? AND feature_name = ? AND timestamp <= ?
            LIMIT 1
        """

        prepared = self.client.session.prepare(query_str)

        for entity_id in entity_ids:
            for feature_name in feature_names:
                result = self.client.session.execute(prepared, (entity_id, feature_name, timestamp))
                row = result.one()

                if row:
                    value = self._extract_value(row)
                    results.append({
                        'entity_id': entity_id,
                        'feature_name': feature_name,
                        'value': value,
                        'timestamp': row['timestamp']
                    })

        return pd.DataFrame(results)

    def get_feature_history(
        self,
        entity_id: str,
        feature_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get time-series history for a feature.

        Args:
            entity_id: Entity identifier
            feature_name: Feature name
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of records

        Returns:
            DataFrame with feature history
        """
        self.connect()

        if start_time and end_time:
            query_str = f"""
                SELECT timestamp, value_double, value_text, value_int, value_bool
                FROM {self.keyspace}.feature_values
                WHERE entity_id = ? AND feature_name = ?
                  AND timestamp >= ? AND timestamp <= ?
                LIMIT ?
            """
            prepared = self.client.session.prepare(query_str)
            result = self.client.session.execute(prepared, (entity_id, feature_name, start_time, end_time, limit))
        else:
            query_str = f"""
                SELECT timestamp, value_double, value_text, value_int, value_bool
                FROM {self.keyspace}.feature_values
                WHERE entity_id = ? AND feature_name = ?
                LIMIT ?
            """
            prepared = self.client.session.prepare(query_str)
            result = self.client.session.execute(prepared, (entity_id, feature_name, limit))

        data = []
        for row in result:
            value = self._extract_value(row)
            data.append({
                'timestamp': row['timestamp'],
                'value': value
            })

        return pd.DataFrame(data)

    def list_features(self) -> pd.DataFrame:
        """
        List all registered features.

        Returns:
            DataFrame with feature metadata
        """
        self.connect()

        query = f"SELECT * FROM {self.keyspace}.feature_metadata"
        result = self.client.execute(query)

        return pd.DataFrame(list(result))

    def list_entities(self, entity_type: str = None) -> pd.DataFrame:
        """
        List entities, optionally filtered by type.

        Args:
            entity_type: Optional entity type filter

        Returns:
            DataFrame with entity information
        """
        self.connect()

        if entity_type:
            query_str = f"""
                SELECT entity_type, entity_id, name, created_at
                FROM {self.keyspace}.entity_by_type
                WHERE entity_type = ?
            """
            prepared = self.client.session.prepare(query_str)
            result = self.client.session.execute(prepared, (entity_type,))
        else:
            query = f"SELECT * FROM {self.keyspace}.entity_registry"
            result = self.client.execute(query)

        return pd.DataFrame(list(result))

    def _extract_value(self, row: Dict) -> Any:
        """Extract the non-null value from a row."""
        if row['value_double'] is not None:
            return row['value_double']
        elif row['value_int'] is not None:
            return row['value_int']
        elif row['value_text'] is not None:
            return row['value_text']
        elif row['value_bool'] is not None:
            return row['value_bool']
        return None

