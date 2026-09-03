"""
Example 1: Setup ScyllaDB Schema

Initialize the Featurama keyspace and tables in ScyllaDB.
"""

import logging
import time
from featurama.scylla.client import ScyllaClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    try:
        # Connect to ScyllaDB (settings come from the environment / .env)
        client = ScyllaClient()

        print(f"📡 Connecting to ScyllaDB at {', '.join(client.contact_points)}...")
        client.connect()
        print("✅ Connected successfully!")
        print()

        # Initialize schema
        print("🏗️  Creating keyspace and tables...")
        client.initialize_schema()
        print("✅ Schema initialized!")
        print()

        # Verify tables
        print("🔍 Verifying tables...")
        tables = [
            "feature_metadata",
            "feature_values",
            "feature_values_by_name",
            "entity_registry",
            "entity_by_type"
        ]

        for table in tables:
            query = f"SELECT * FROM {client.keyspace}.{table} LIMIT 1"
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

