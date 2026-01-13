"""
Example 6: Inference Server & Demo

Start the inference server and demonstrate real-time predictions.
"""

import logging
import time
import requests
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_health_check(base_url: str):
    """Test health check endpoint."""
    print("🏥 Health Check")
    print("-" * 40)

    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()

        print(f"  Status: {data['status']}")
        print(f"  Feature Store: {'✅' if data['feature_store_connected'] else '❌'}")
        print(f"  Model: {'✅' if data['model_loaded'] else '❌'}")
        print()

        return data['status'] == 'healthy'
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def test_prediction(base_url: str, features: Dict[str, float]):
    """Test prediction endpoint."""
    print("🔮 Prediction Test")
    print("-" * 40)

    request_data = {"features": features}

    print("Request:")
    for key, value in features.items():
        print(f"  • {key}: {value}")
    print()

    try:
        response = requests.post(f"{base_url}/predict", json=request_data)
        response.raise_for_status()
        data = response.json()

        print("Response:")
        print(f"  ✅ Predicted Duration: {data['predicted_duration']:.2f} hours")
        print(f"  Timestamp: {data['timestamp']}")
        print()

        return data
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        return None


def test_batch_scenarios(base_url: str):
    """Test various delivery scenarios."""
    print("🎬 Scenario Testing")
    print("=" * 80)
    print()

    scenarios = [
        {
            "name": "Short, Light Delivery (Earth to Moon)",
            "features": {
                "distance": 384.4,  # km
                "package_weight": 10.0,
                "hazard_level": 0.1,
                "estimated_duration": 0.5,
                "traffic_level": 0.3,
                "weather_index": 0.9
            }
        },
        {
            "name": "Medium Delivery (Earth to Mars)",
            "features": {
                "distance": 225000.0,  # km at closest approach
                "package_weight": 150.0,
                "hazard_level": 0.4,
                "estimated_duration": 8.0,
                "traffic_level": 0.6,
                "weather_index": 0.7
            }
        },
        {
            "name": "Long Haul (Earth to Omicron Persei 8)",
            "features": {
                "distance": 1000000.0,
                "package_weight": 500.0,
                "hazard_level": 0.8,
                "estimated_duration": 72.0,
                "traffic_level": 0.2,
                "weather_index": 0.5
            }
        },
        {
            "name": "Hazardous Delivery (Through Neutral Zone)",
            "features": {
                "distance": 50000.0,
                "package_weight": 300.0,
                "hazard_level": 0.95,
                "estimated_duration": 15.0,
                "traffic_level": 0.4,
                "weather_index": 0.3
            }
        },
        {
            "name": "Express Delivery (Local System)",
            "features": {
                "distance": 5000.0,
                "package_weight": 25.0,
                "hazard_level": 0.15,
                "estimated_duration": 2.0,
                "traffic_level": 0.8,
                "weather_index": 0.95
            }
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"📦 {scenario['name']}")
        print("-" * 80)
        result = test_prediction(base_url, scenario['features'])
        if result:
            results.append({
                "scenario": scenario['name'],
                "estimated": scenario['features']['estimated_duration'],
                "predicted": result['predicted_duration']
            })
        time.sleep(0.5)

    # Summary
    print("=" * 80)
    print("📊 Scenario Summary")
    print("=" * 80)
    print()

    print(f"{'Scenario':<50} {'Estimated':<12} {'Predicted':<12} {'Diff':<10}")
    print("-" * 84)

    for r in results:
        diff = r['predicted'] - r['estimated']
        sign = "+" if diff > 0 else ""
        print(f"{r['scenario']:<50} {r['estimated']:>8.2f}h   {r['predicted']:>8.2f}h   {sign}{diff:>6.2f}h")

    print()


def main():
    """Run inference demonstrations."""
    print("=" * 80)
    print("🚀 Featurama - Inference Server Demo")
    print("=" * 80)
    print()

    print('"Good news, everyone! The What-If Machine is ready!"')
    print("                                    - Professor Farnsworth")
    print()

    print("=" * 80)
    print("📝 Instructions")
    print("=" * 80)
    print()
    print("To start the inference server, run in a separate terminal:")
    print()
    print("  python -m featurama.ml.inference")
    print()
    print("Or use uvicorn directly:")
    print()
    print("  uvicorn featurama.ml.inference:app --reload")
    print()
    print("The server will start on: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print()

    # Wait for user to start server
    input("Press Enter once the server is running...")
    print()

    base_url = "http://localhost:8000"

    # Test health
    print("=" * 80)
    healthy = test_health_check(base_url)

    if not healthy:
        print("❌ Server is not healthy. Please check the server logs.")
        return 1

    # Run scenario tests
    test_batch_scenarios(base_url)

    print("=" * 80)
    print("🌐 Interactive API")
    print("=" * 80)
    print()
    print("The inference server provides the following endpoints:")
    print()
    print("  • POST /predict - Single prediction")
    print("  • POST /features/batch - Batch predictions")
    print("  • GET /features/{entity_id} - Get entity features")
    print("  • GET /health - Health check")
    print()
    print("Try the interactive API docs at:")
    print("  http://localhost:8000/docs")
    print()

    print("=" * 80)
    print("✨ Demo complete!")
    print("=" * 80)
    print()
    print("Example curl commands:")
    print()
    print("# Single prediction")
    print('curl -X POST "http://localhost:8000/predict" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"features": {"distance": 5000, "package_weight": 150, "hazard_level": 0.3, "estimated_duration": 10.5, "traffic_level": 0.7, "weather_index": 0.8}}\'')
    print()
    print("# Health check")
    print('curl "http://localhost:8000/health"')
    print()

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

