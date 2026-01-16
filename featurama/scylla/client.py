"""
ScyllaDB client for Featurama.

Handles connection management and query execution.
"""

import logging
from typing import Any, Dict, List, Optional

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import EXEC_PROFILE_DEFAULT, Cluster, ExecutionProfile
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from cassandra.query import dict_factory

from featurama.scylla.schema import KEYSPACE_NAME, get_schema_statements

logger = logging.getLogger(__name__)


class ScyllaClient:
    """Client for ScyllaDB operations."""

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
        Initialize ScyllaDB client.

        Args:
            contact_points: List of ScyllaDB node addresses
            port: CQL port (default 9042)
            keyspace: Keyspace name
        """
        self.contact_points = contact_points or ["127.0.0.1"]
        self.port = port
        self.keyspace = keyspace
        self.username = username
        self.password = password
        self.datacenter = datacenter
        self.cluster = None
        self.session = None

    def connect(self):
        """Establish connection to ScyllaDB."""
        if self.session:
            logger.info("Already connected to ScyllaDB")
            return

        logger.info(f"Connecting to ScyllaDB at {self.contact_points}:{self.port}")

        # Create execution profile for better performance
        profile = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(
                DCAwareRoundRobinPolicy(local_dc=self.datacenter)
            ),
            row_factory=dict_factory,
        )

        # Setup authentication if credentials are provided
        auth_provider = None
        if self.username and self.password:
            auth_provider = PlainTextAuthProvider(
                username=self.username, password=self.password
            )

        self.cluster = Cluster(
            contact_points=self.contact_points,
            port=self.port,
            execution_profiles={EXEC_PROFILE_DEFAULT: profile},
            protocol_version=4,
            auth_provider=auth_provider,
        )

        self.session = self.cluster.connect()
        logger.info("Successfully connected to ScyllaDB")

    def disconnect(self):
        """Close connection to ScyllaDB."""
        if self.session:
            self.session.shutdown()
            logger.info("Session closed")
        if self.cluster:
            self.cluster.shutdown()
            logger.info("Cluster connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def execute(self, query: str, parameters: tuple = None) -> Any:
        """
        Execute a CQL query.

        Args:
            query: CQL query string
            parameters: Query parameters

        Returns:
            Query result
        """
        if not self.session:
            self.connect()

        try:
            if parameters:
                return self.session.execute(query, parameters)
            return self.session.execute(query)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_batch(self, queries: List[tuple]):
        """
        Execute multiple queries in batch.

        Args:
            queries: List of (query, parameters) tuples
        """
        if not self.session:
            self.connect()

        from cassandra.query import BatchStatement

        batch = BatchStatement()
        for query, params in queries:
            batch.add(query, params)

        try:
            self.session.execute(batch)
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise

    def initialize_schema(self, replication_factor):
        """Create keyspace and tables."""
        logger.info("Initializing Featurama schema...")

        statements = get_schema_statements(replication_factor)
        for statement in statements:
            logger.info(f"Executing: {statement[:100]}...")
            self.execute(statement)

        # Set default keyspace
        self.session.set_keyspace(self.keyspace)
        logger.info(f"Schema initialized. Using keyspace: {self.keyspace}")

    def truncate_all_tables(self):
        """Truncate all feature store tables (use with caution!)."""
        tables = [
            "feature_metadata",
            "feature_values",
            "feature_values_by_name",
            "entity_registry",
            "entity_by_type",
        ]

        for table in tables:
            try:
                self.execute(f"TRUNCATE {self.keyspace}.{table}")
                logger.info(f"Truncated {table}")
            except Exception as e:
                logger.warning(f"Failed to truncate {table}: {e}")

    def get_table_count(self, table_name: str) -> int:
        """
        Get approximate row count for a table.

        Note: This is an expensive operation on large tables.
        Use with caution.
        """
        try:
            result = self.execute(f"SELECT COUNT(*) FROM {self.keyspace}.{table_name}")
            return result.one()["count"]
        except Exception as e:
            logger.error(f"Failed to count rows in {table_name}: {e}")
            return -1
