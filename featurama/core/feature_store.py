"""
Core Feature Store implementation.

Provides high-level API for feature management, registration,
writing, and retrieval from ScyllaDB.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from cassandra.concurrent import execute_concurrent

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
        keyspace: str = KEYSPACE_NAME,
        username: str = None,
        password: str = None,
        datacenter: str = None,
    ):
        """
        Initialize Feature Store.

        Args:
            contact_points: ScyllaDB node addresses
            port: CQL port
            keyspace: Keyspace name
        """
        self.client = ScyllaClient(
            contact_points, port, keyspace, username, password, datacenter
        )
        self.keyspace = keyspace
        self._connected = False
        self.prepared = {}

    def _prepare_statements(self):
        """Centralize all CQL preparations here."""

        # Register New Feature
        register_cql = f"""
            INSERT INTO {self.keyspace}.feature_metadata
            (feature_name, version, feature_type, description, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.prepared["register_feature_query"] = self.client.session.prepare(
            register_cql
        )

        # Register new Entitiy
        register_new_entity = f"""
            INSERT INTO {self.keyspace}.entity_registry
            (entity_id, entity_type, name, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.prepared["register_entity_query"] = self.client.session.prepare(
            register_new_entity
        )

        # Register new Entity Type Index
        register_new_entity_index = f"""
            INSERT INTO {self.keyspace}.entity_by_type
            (entity_type, entity_id, name, created_at)
            VALUES (?, ?, ?, ?)
        """
        self.prepared["register_entity_index_query"] = self.client.session.prepare(
            register_new_entity_index
        )

        insert_features_query_str = f"""
            INSERT INTO {self.keyspace}.feature_values
            (entity_id, feature_name, timestamp, value_double, value_text, value_int, value_bool, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.prepared["insert_features_query"] = self.client.session.prepare(
            insert_features_query_str
        )

        insert_features_query_by_name_str = f"""
            INSERT INTO {self.keyspace}.feature_values_by_name
            (feature_name, entity_id, timestamp, value_double, value_text, value_int, value_bool, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.prepared["insert_features_query_by_name"] = self.client.session.prepare(
            insert_features_query_by_name_str
        )

        get_online_features_query_str = f"""
            SELECT entity_id, feature_name, timestamp,
                   value_double, value_text, value_int, value_bool
            FROM {self.keyspace}.feature_values
            WHERE entity_id = ? AND feature_name = ?
            LIMIT 1
        """
        self.prepared["get_online_features_query"] = self.client.session.prepare(
            get_online_features_query_str
        )

        get_historical_features_query_str = f"""
            SELECT entity_id, feature_name, timestamp,
                   value_double, value_text, value_int, value_bool
            FROM {self.keyspace}.feature_values
            WHERE entity_id = ? AND feature_name = ? AND timestamp <= ?
            LIMIT 1
        """
        self.prepared["get_historical_features_query"] = self.client.session.prepare(
            get_historical_features_query_str
        )

        get_feature_history_timeframe_query_str = f"""
            SELECT timestamp, value_double, value_text, value_int, value_bool
            FROM {self.keyspace}.feature_values
            WHERE entity_id = ? AND feature_name = ?
              AND timestamp >= ? AND timestamp <= ?
            LIMIT ?
        """
        self.prepared["get_feature_history_timeframe_query"] = (
            self.client.session.prepare(get_feature_history_timeframe_query_str)
        )

        get_feature_history_all_query_str = f"""
            SELECT timestamp, value_double, value_text, value_int, value_bool
            FROM {self.keyspace}.feature_values
            WHERE entity_id = ? AND feature_name = ?
            LIMIT ?
        """
        self.prepared["get_feature_history_all_query"] = self.client.session.prepare(
            get_feature_history_all_query_str
        )

        list_features_query_str = f"SELECT * FROM {self.keyspace}.feature_metadata"
        self.prepared["list_features_query"] = self.client.session.prepare(
            list_features_query_str
        )

        list_entries_by_entity_type_query_str = f"""
                        SELECT entity_type, entity_id, name, created_at
                        FROM {self.keyspace}.entity_by_type
                        WHERE entity_type = ?
                    """
        self.prepared["list_entries_by_entity_type_query"] = (
            self.client.session.prepare(list_entries_by_entity_type_query_str)
        )

        list_entries_all_query_str = f"SELECT * FROM {self.keyspace}.entity_registry"
        self.prepared["list_entries_all_query"] = self.client.session.prepare(
            list_entries_all_query_str
        )

    def connect(self):
        """Connect to ScyllaDB."""
        if not self._connected:
            self.client.connect()
            self.client.session.set_keyspace(self.keyspace)
            self._connected = True
            logger.info("Feature Store connected")
            self._prepare_statements()

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
        tags: Dict[str, str] = None,
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

        self.client.session.execute(
            self.prepared["register_feature_query"],
            (feature_name, version, feature_type, description, now, now, tags),
        )

        logger.info(f"Registered feature: {feature_name} v{version} ({feature_type})")

    def register_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        metadata: Dict[str, str] = None,
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
        self.client.session.execute(
            self.prepared["register_entity_query"],
            (entity_id, entity_type, name, metadata, now, now),
        )

        # Insert into entity_by_type index
        self.client.session.execute(
            self.prepared["register_entity_index_query"],
            (entity_type, entity_id, name, now),
        )

        logger.debug(f"Registered entity: {entity_id} ({entity_type})")

    def write_features(
        self, features_df: pd.DataFrame, version: int = 1, batch_size: int = 1000
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

        if "timestamp" not in features_df.columns:
            features_df["timestamp"] = datetime.now()

        logger.info(f"Writing {len(features_df)} feature values...")

        # Prepare batched inserts
        from cassandra.query import BatchStatement

        features_batch = BatchStatement()
        features_by_name_batch = BatchStatement()

        for _, row in features_df.iterrows():
            entity_id = row["entity_id"]
            feature_name = row["feature_name"]
            timestamp = row["timestamp"]
            value = row["value"]

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

            params = (
                entity_id,
                feature_name,
                timestamp,
                value_double,
                value_text,
                value_int,
                value_bool,
                version,
            )
            params_by_name = (
                feature_name,
                entity_id,
                timestamp,
                value_double,
                value_text,
                value_int,
                value_bool,
                version,
            )

            features_batch.add(self.prepared["insert_features_query"], params)
            features_by_name_batch.add(
                self.prepared["insert_features_query_by_name"], params_by_name
            )

            if (
                len(features_batch) % batch_size == 0
                and len(features_by_name_batch) % batch_size == 0
            ):
                execute_concurrent(
                    self.client.session,
                    [(features_batch, None), (features_by_name_batch, None)],
                )
                logger.info(
                    f"Written {len(features_batch) + len(features_by_name_batch)} features..."
                )
                features_batch = BatchStatement()
                features_by_name_batch = BatchStatement()
            elif len(features_batch) % batch_size == 0:
                self.client.session.execute(features_batch)
                logger.info(f"Written {len(features_batch)} features...")
                features_batch = BatchStatement()
            elif len(features_by_name_batch) % batch_size == 0:
                self.client.session.execute(features_by_name_batch)
                logger.info(f"Written {len(features_by_name_batch)} features...")
                features_by_name_batch = BatchStatement()

        # Execute remaining batch
        if len(features_batch) % batch_size != 0:
            self.client.session.execute(features_batch)

        if len(features_by_name_batch) % batch_size != 0:
            self.client.session.execute(features_by_name_batch)

        logger.info(f"Successfully wrote {len(features_df)} feature values")

    def get_online_features(
        self, entity_ids: List[str], feature_names: List[str]
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

        for entity_id in entity_ids:
            for feature_name in feature_names:
                result = self.client.session.execute(
                    self.prepared["get_online_features_query"],
                    (entity_id, feature_name),
                )
                row = result.one()

                if row:
                    value = self._extract_value(row)
                    results.append(
                        {
                            "entity_id": entity_id,
                            "feature_name": feature_name,
                            "value": value,
                            "timestamp": row["timestamp"],
                        }
                    )

        return pd.DataFrame(results)

    def get_historical_features(
        self, entity_ids: List[str], feature_names: List[str], timestamp: datetime
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

        for entity_id in entity_ids:
            for feature_name in feature_names:
                result = self.client.session.execute(
                    self.prepared["get_historical_features_query"],
                    (entity_id, feature_name, timestamp),
                )
                row = result.one()

                if row:
                    value = self._extract_value(row)
                    results.append(
                        {
                            "entity_id": entity_id,
                            "feature_name": feature_name,
                            "value": value,
                            "timestamp": row["timestamp"],
                        }
                    )

        return pd.DataFrame(results)

    def get_feature_history(
        self,
        entity_id: str,
        feature_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000,
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
            result = self.client.session.execute(
                self.prepared["get_feature_history_timeframe_query"],
                (entity_id, feature_name, start_time, end_time, limit),
            )
        else:
            result = self.client.session.execute(
                self.prepared["get_feature_history_all_query"],
                (entity_id, feature_name, limit),
            )

        data = []
        for row in result:
            value = self._extract_value(row)
            data.append({"timestamp": row["timestamp"], "value": value})

        return pd.DataFrame(data)

    def list_features(self) -> pd.DataFrame:
        """
        List all registered features.

        Returns:
            DataFrame with feature metadata
        """
        self.connect()

        result = self.client.execute(self.prepared["list_features_query"])

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
            result = self.client.session.execute(
                self.prepared["list_entries_by_entity_type_query"], (entity_type,)
            )
        else:
            result = self.client.execute(self.prepared["list_entries_all_query"])

        return pd.DataFrame(list(result))

    def _extract_value(self, row: Dict) -> Any:
        """Extract the non-null value from a row."""
        if row["value_double"] is not None:
            return row["value_double"]
        elif row["value_int"] is not None:
            return row["value_int"]
        elif row["value_text"] is not None:
            return row["value_text"]
        elif row["value_bool"] is not None:
            return row["value_bool"]
        return None
