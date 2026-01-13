"""
Example 5: Train ML Model

Train a delivery time prediction model using features from the store.
"""

import logging
import pandas as pd
import os

from featurama.core.feature_store import FeatureStore
from featurama.ml.training import train_delivery_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Train delivery time prediction model."""
    print("=" * 80)
    print("🧠 Featurama - ML Model Training")
    print("=" * 80)
    print()

    print('"I\'m going to build my own AI! With blackjack! And hookers!"')
    print("                                              - Bender")
    print()

    # Load entity data
    print("📂 Loading entity data...")
    try:
        entities_df = pd.read_csv('data/entities.csv')
    except FileNotFoundError:
        print("❌ Data files not found!")
        print("Run the previous examples first.")
        return 1

    delivery_ids = entities_df[entities_df['entity_type'] == 'delivery']['entity_id'].tolist()

    print(f"  ✅ Found {len(delivery_ids):,} deliveries for training")
    print()

    # Connect to feature store
    print("🔌 Connecting to feature store...")
    fs = FeatureStore()
    fs.connect()
    print("  ✅ Connected!")
    print()

    # Train model
    print("=" * 80)
    print("🏋️  Training Delivery Time Prediction Model")
    print("=" * 80)
    print()

    print("Model: XGBoost Regressor")
    print(f"Training samples: {min(500, len(delivery_ids))} deliveries")
    print()

    print("Features used:")
    features = [
        'distance',
        'package_weight',
        'hazard_level',
        'estimated_duration',
        'traffic_level',
        'weather_index'
    ]
    for feat in features:
        print(f"  • {feat}")
    print()

    print("Target: actual_duration (hours)")
    print()

    os.makedirs('models', exist_ok=True)

    print("🎯 Starting training...")
    print()

    try:
        predictor, metrics = train_delivery_model(
            feature_store=fs,
            delivery_ids=delivery_ids,
            model_path="models/delivery_predictor.pkl",
            sample_size=500  # Use subset for faster training
        )

        print()
        print("=" * 80)
        print("📊 Training Results")
        print("=" * 80)
        print()

        print(f"✅ Model trained successfully!")
        print()
        print(f"Performance Metrics:")
        print(f"  • MAE (Mean Absolute Error): {metrics['mae']:.4f} hours")
        print(f"  • RMSE (Root Mean Squared Error): {metrics['rmse']:.4f} hours")
        print(f"  • R² Score: {metrics['r2']:.4f}")
        print()
        print(f"Dataset Split:")
        print(f"  • Training samples: {metrics['train_samples']}")
        print(f"  • Test samples: {metrics['test_samples']}")
        print()

        # Test prediction
        print("🧪 Testing prediction on sample delivery...")
        print()

        test_features = {
            'distance': 5000.0,
            'package_weight': 150.0,
            'hazard_level': 0.3,
            'estimated_duration': 10.5,
            'traffic_level': 0.7,
            'weather_index': 0.8
        }

        prediction = predictor.predict_delivery_time(test_features)

        print("Test Input:")
        for key, value in test_features.items():
            print(f"  • {key}: {value}")
        print()
        print(f"Predicted Delivery Time: {prediction:.2f} hours")
        print()

        # Compare with estimated
        print(f"Comparison:")
        print(f"  • Estimated duration: {test_features['estimated_duration']:.2f} hours")
        print(f"  • Predicted duration: {prediction:.2f} hours")
        print(f"  • Difference: {prediction - test_features['estimated_duration']:.2f} hours")
        print()

        print("💾 Model saved to: models/delivery_predictor.pkl")
        print()

    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.exception("Training error")
        return 1
    finally:
        fs.disconnect()

    print("=" * 80)
    print("✨ Model training complete!")
    print("=" * 80)
    print()
    print("Next step:")
    print("  Run: python examples/06_inference.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())

