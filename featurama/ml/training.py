"""
ML Training pipeline for Featurama.

Trains a delivery time prediction model using features from the feature store.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from featurama.core.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class DeliveryTimePredictor:
    """
    ML model for predicting delivery times.

    Uses features from the feature store to predict actual delivery duration.
    """

    def __init__(self, feature_store: FeatureStore = None):
        """
        Initialize predictor.

        Args:
            feature_store: FeatureStore instance
        """
        self.feature_store = feature_store or FeatureStore()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'distance',
            'package_weight',
            'hazard_level',
            'estimated_duration',
            'traffic_level',
            'weather_index'
        ]

    def prepare_training_data(
        self,
        delivery_ids: List[str],
        sample_size: int = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from feature store.

        Args:
            delivery_ids: List of delivery entity IDs
            sample_size: Optional sample size (for testing with subset)

        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing training data from feature store...")

        if sample_size:
            delivery_ids = delivery_ids[:sample_size]

        self.feature_store.connect()

        training_data = []

        for delivery_id in delivery_ids:
            # Get delivery features
            features = {}

            # Get delivery-specific features
            for feature_name in ['distance', 'package_weight', 'hazard_level', 'estimated_duration']:
                hist = self.feature_store.get_feature_history(
                    entity_id=delivery_id,
                    feature_name=feature_name,
                    limit=1
                )
                if not hist.empty:
                    features[feature_name] = hist.iloc[0]['value']

            # Get actual duration (target)
            actual_hist = self.feature_store.get_feature_history(
                entity_id=delivery_id,
                feature_name='actual_duration',
                limit=1
            )

            if actual_hist.empty or len(features) < 4:
                continue

            # Get associated planet features (origin/destination)
            # For simplicity, using mock values - in production, would join with entity metadata
            features['traffic_level'] = np.random.uniform(0.3, 1.0)
            features['weather_index'] = np.random.uniform(0, 1.0)

            features['actual_duration'] = actual_hist.iloc[0]['value']
            training_data.append(features)

            if len(training_data) % 100 == 0:
                logger.info(f"Prepared {len(training_data)} samples...")

        df = pd.DataFrame(training_data)

        if df.empty:
            raise ValueError("No training data available")

        # Separate features and target
        X = df[self.feature_names]
        y = df['actual_duration']

        logger.info(f"Prepared {len(X)} training samples")
        return X, y

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features.

        Args:
            X: Input features

        Returns:
            Enhanced feature DataFrame
        """
        X = X.copy()

        # Create interaction features
        X['distance_weight_interaction'] = X['distance'] * X['package_weight']
        X['hazard_traffic_interaction'] = X['hazard_level'] * X['traffic_level']
        X['speed_estimate'] = X['distance'] / (X['estimated_duration'] + 0.001)
        X['weight_hazard_ratio'] = X['package_weight'] / (X['hazard_level'] + 0.001)

        # Polynomial features for key metrics
        X['distance_squared'] = X['distance'] ** 2
        X['weight_squared'] = X['package_weight'] ** 2

        return X

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the delivery time prediction model.

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Training delivery time prediction model...")

        # Engineer features
        X_engineered = self.engineer_features(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        logger.info(f"Training complete!")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_engineered.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 10 Feature Importances:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_engineered = self.engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)

        return self.model.predict(X_scaled)

    def predict_delivery_time(self, features: Dict) -> float:
        """
        Predict delivery time for a single delivery.

        Args:
            features: Dictionary of feature values

        Returns:
            Predicted delivery time (hours)
        """
        # Ensure all required features are present
        for fname in self.feature_names:
            if fname not in features:
                raise ValueError(f"Missing required feature: {fname}")

        X = pd.DataFrame([features])
        prediction = self.predict(X)

        return float(prediction[0])

    def save_model(self, path: str):
        """
        Save trained model to disk.

        Args:
            path: File path for model
        """
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load trained model from disk.

        Args:
            path: File path for model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        logger.info(f"Model loaded from {path}")


def train_delivery_model(
    feature_store: FeatureStore,
    delivery_ids: List[str],
    model_path: str = "models/delivery_predictor.pkl",
    sample_size: int = None
) -> Tuple[DeliveryTimePredictor, Dict[str, float]]:
    """
    Train and save a delivery time prediction model.

    Args:
        feature_store: FeatureStore instance
        delivery_ids: List of delivery IDs for training
        model_path: Path to save the model
        sample_size: Optional sample size

    Returns:
        Tuple of (trained_predictor, metrics)
    """
    predictor = DeliveryTimePredictor(feature_store)

    # Prepare data
    X, y = predictor.prepare_training_data(delivery_ids, sample_size)

    # Train model
    metrics = predictor.train(X, y)

    # Save model
    predictor.save_model(model_path)

    return predictor, metrics

