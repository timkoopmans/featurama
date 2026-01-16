"""
Example 1: Setup ScyllaDB Schema

Initialize the Featurama keyspace and tables in ScyllaDB.
"""

import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

from featurama.scylla.client import ScyllaClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Initialize ScyllaDB schema."""
    print("=" * 80)
    print("🚀 Featurama - ScyllaDB Schema Initialization")
    print("=" * 80)
    print()

    print("Good news, everyone! Setting up the Featurama database...")
    print()

    # Wait a moment for ScyllaDB to be ready (if just started)
    print("⏳ Waiting for ScyllaDB to be ready...")
    time.sleep(2)

    replication_factor = os.getenv("SCYLLA_REPLICATION_FACTOR", "1")
    contact_points = os.getenv("SCYLLA_CONTACT_POINTS", "127.0.0.1").split(",")

    try:
        # Connect to ScyllaDB
        client = ScyllaClient(contact_points=contact_points, port=9042)

        print("📡 Connecting to ScyllaDB...")
        client.connect()
        print("✅ Connected successfully!")
        print()

        # Initialize schema
        print("🏗️  Creating keyspace and tables...")
        client.initialize_schema(replication_factor)
        print("✅ Schema initialized!")
        print()

        # Verify tables
        print("🔍 Verifying tables...")
        tables = [
            "feature_metadata",
            "feature_values",
            "feature_values_by_name",
            "entity_registry",
            "entity_by_type",
        ]

        for table in tables:
            query = f"SELECT * FROM featurama.{table} LIMIT 1"
            try:
                client.execute(query)
                print(f"  ✅ {table}")
            except Exception as e:
                print(f"  ❌ {table}: {e}")

        print()
        print("=" * 80)
        print("✨ Schema setup complete! The What-If Machine is ready!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Run: python examples/02_generate_data.py")
        print("  2. Run: python examples/03_feature_ingestion.py")
        print()

        client.disconnect()

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print()
        print("❌ Setup failed!")
        print(f"Error: {e}")
        print()
        print("Make sure ScyllaDB is running:")
        print("  docker-compose up -d")
        print()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
