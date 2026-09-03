"""
ScyllaDB client for Featurama.

Handles connection management and query execution.
"""

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from cassandra.query import dict_factory
from typing import List, Dict, Any, Optional
import logging

from featurama.scylla.schema import KEYSPACE_NAME, get_schema_statements

logger = logging.getLogger(__name__)


class ScyllaClient:
    """Client for ScyllaDB operations."""

    def __init__(
        self,
        contact_points: List[str] = None,
        port: int = None,
        keyspace: str = None,
        username: str = None,
        password: str = None,
        local_dc: str = None,
        replication_factor: int = None,
        ssl: bool = None,
        config: "ScyllaConfig" = None
    ):
        """
        Initialize ScyllaDB client.

        Any argument left as None falls back to the environment-derived
        config (see featurama.config), so a local docker cluster needs no
        arguments and Scylla Cloud only needs a .env file.

        Args:
            contact_points: List of ScyllaDB node addresses
            port: CQL port (default 9042)
            keyspace: Keyspace name
            username: CQL username (required by Scylla Cloud)
            password: CQL password
            local_dc: Datacenter name for DC-aware routing
            replication_factor: Replication factor used when creating the keyspace
            ssl: Enable TLS for the CQL connection
            config: Pre-built ScyllaConfig; defaults to ScyllaConfig.from_env()
        """
        from featurama.config import ScyllaConfig

        cfg = config or ScyllaConfig.from_env()

        self.contact_points = contact_points or cfg.contact_points
        self.port = port if port is not None else cfg.port
        self.keyspace = keyspace or cfg.keyspace
        self.username = username if username is not None else cfg.username
        self.password = password if password is not None else cfg.password
        self.local_dc = local_dc if local_dc is not None else cfg.local_dc
        self.replication_factor = (
            replication_factor if replication_factor is not None
            else cfg.replication_factor
        )
        self.ssl = ssl if ssl is not None else cfg.ssl
        self.cluster = None
        self.session = None

    def connect(self):
        """Establish connection to ScyllaDB."""
        if self.session:
            logger.info("Already connected to ScyllaDB")
            return

        logger.info(f"Connecting to ScyllaDB at {self.contact_points}:{self.port}")

        # Pin routing to the local DC when known, so requests stay in-region
        dc_policy = (
            DCAwareRoundRobinPolicy(local_dc=self.local_dc)
            if self.local_dc else DCAwareRoundRobinPolicy()
        )

        # Create execution profile for better performance
        profile = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(dc_policy),
            row_factory=dict_factory
        )

        auth_provider = None
        if self.username:
            auth_provider = PlainTextAuthProvider(
                username=self.username,
                password=self.password
            )

        ssl_context = None
        if self.ssl:
            import ssl as ssl_module

            ssl_context = ssl_module.create_default_context()

        self.cluster = Cluster(
            contact_points=self.contact_points,
            port=self.port,
            auth_provider=auth_provider,
            ssl_context=ssl_context,
            execution_profiles={EXEC_PROFILE_DEFAULT: profile},
            protocol_version=4
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

    def initialize_schema(self):
        """Create keyspace and tables."""
        logger.info("Initializing Featurama schema...")

        statements = get_schema_statements(
            keyspace=self.keyspace,
            replication_factor=self.replication_factor,
            local_dc=self.local_dc
        )
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
            "entity_by_type"
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
            return result.one()['count']
        except Exception as e:
            logger.error(f"Failed to count rows in {table_name}: {e}")
            return -1

