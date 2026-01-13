#!/usr/bin/env python
"""
Featurama Validation Script

Validates that all components are properly installed and configured.
"""

import sys
import importlib


def check_module(module_name, friendly_name=None):
    """Check if a module can be imported."""
    friendly_name = friendly_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✅ {friendly_name}")
        return True
    except ImportError as e:
        print(f"❌ {friendly_name}: {e}")
        return False


def main():
    """Run validation checks."""
    print("=" * 80)
    print("🔍 Featurama Validation")
    print("=" * 80)
    print()

    print("Checking Python Modules:")
    print("-" * 40)

    modules = [
        ("featurama", "Featurama Core"),
        ("featurama.core.feature_store", "Feature Store"),
        ("featurama.scylla.client", "ScyllaDB Client"),
        ("featurama.scylla.schema", "ScyllaDB Schema"),
        ("featurama.data_generation.synthetic_data", "Data Generator"),
        ("featurama.ml.training", "ML Training"),
        ("featurama.ml.inference", "ML Inference"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("ray", "Ray"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("faker", "Faker"),
        ("cassandra", "Cassandra/Scylla Driver"),
    ]

    results = []
    for module, name in modules:
        results.append(check_module(module, name))

    print()
    print("Checking Example Scripts:")
    print("-" * 40)

    examples = [
        "examples.01_setup_scylla",
        "examples.02_generate_data",
        "examples.03_feature_ingestion",
        "examples.04_feature_retrieval",
        "examples.05_train_model",
        "examples.06_inference",
        "examples.07_ray_integration",
    ]

    for example in examples:
        results.append(check_module(example, example))

    print()
    print("=" * 80)

    if all(results):
        print("✅ All validations passed!")
        print()
        print("Next steps:")
        print("  1. Start ScyllaDB: docker-compose up -d")
        print("  2. Run quickstart: ./quickstart.sh")
        print("  3. Or run examples manually starting with:")
        print("     python examples/01_setup_scylla.py")
        print()
        return 0
    else:
        print("❌ Some validations failed!")
        print()
        print("Fix issues by running:")
        print("  pip install -r requirements.txt")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

