"""
Connection configuration for Featurama.

Reads settings from the environment (optionally via a .env file) so the same
code runs against a local docker-compose ScyllaDB or a Scylla Cloud cluster.

Environment variables:
    SCYLLA_CONTACT_POINTS  Comma-separated node addresses (default: 127.0.0.1)
    SCYLLA_PORT            CQL port (default: 9042)
    SCYLLA_USERNAME        CQL username (optional; omit for a local cluster)
    SCYLLA_PASSWORD        CQL password (optional)
    SCYLLA_LOCAL_DC        Datacenter for token/DC-aware routing (e.g. AWS_US_EAST_1)
    SCYLLA_KEYSPACE        Keyspace name (default: featurama)
    SCYLLA_REPLICATION_FACTOR  Replication factor for the keyspace (default: 1)
    SCYLLA_SSL             Set to 1/true to enable TLS
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from featurama.scylla.schema import KEYSPACE_NAME

try:  # optional, listed in requirements.txt
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ScyllaConfig:
    """Everything needed to connect to a ScyllaDB cluster."""

    contact_points: List[str] = field(default_factory=lambda: ["127.0.0.1"])
    port: int = 9042
    username: Optional[str] = None
    password: Optional[str] = None
    local_dc: Optional[str] = None
    keyspace: str = KEYSPACE_NAME
    replication_factor: int = 1
    ssl: bool = False

    @classmethod
    def from_env(cls) -> "ScyllaConfig":
        """Build a config from environment variables."""
        contact_points = [
            host.strip()
            for host in os.getenv("SCYLLA_CONTACT_POINTS", "127.0.0.1").split(",")
            if host.strip()
        ]

        return cls(
            contact_points=contact_points or ["127.0.0.1"],
            port=int(os.getenv("SCYLLA_PORT", "9042")),
            username=os.getenv("SCYLLA_USERNAME") or None,
            password=os.getenv("SCYLLA_PASSWORD") or None,
            local_dc=os.getenv("SCYLLA_LOCAL_DC") or None,
            keyspace=os.getenv("SCYLLA_KEYSPACE", KEYSPACE_NAME),
            replication_factor=int(os.getenv("SCYLLA_REPLICATION_FACTOR", "1")),
            ssl=_env_bool("SCYLLA_SSL"),
        )

    def describe(self) -> str:
        """Human-readable summary, safe to log (no password)."""
        target = f"{','.join(self.contact_points)}:{self.port}"
        auth = f"user={self.username}" if self.username else "no auth"
        dc = f"dc={self.local_dc}" if self.local_dc else "dc=auto"
        return f"{target} ({auth}, {dc}, keyspace={self.keyspace}, rf={self.replication_factor})"
