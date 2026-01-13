"""
ML Inference server for Featurama.

FastAPI server for real-time delivery time predictions using the feature store.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from featurama.core.feature_store import FeatureStore
from featurama.ml.training import DeliveryTimePredictor

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Featurama Inference API",
    description="Real-time delivery time predictions using ScyllaDB-backed feature store",
    version="1.0.0"
)

# Global instances
feature_store: Optional[FeatureStore] = None
predictor: Optional[DeliveryTimePredictor] = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    delivery_id: Optional[str] = Field(None, description="Delivery ID for feature lookup")
    features: Optional[Dict[str, float]] = Field(None, description="Manual feature values")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "distance": 5000.0,
                    "package_weight": 150.0,
                    "hazard_level": 0.3,
                    "estimated_duration": 10.5,
                    "traffic_level": 0.7,
                    "weather_index": 0.8
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_duration: float = Field(..., description="Predicted delivery time in hours")
    delivery_id: Optional[str] = None
    features_used: Dict[str, float]
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    feature_store_connected: bool
    model_loaded: bool
    timestamp: datetime


@app.on_event("startup")
async def startup_event():
    """Initialize feature store and model on startup."""
    global feature_store, predictor

    logger.info("Starting Featurama Inference Server...")

    # Initialize feature store
    feature_store = FeatureStore()
    try:
        feature_store.connect()
        logger.info("Feature store connected")
    except Exception as e:
        logger.error(f"Failed to connect to feature store: {e}")
        feature_store = None

    # Load model
    predictor = DeliveryTimePredictor(feature_store)
    try:
        predictor.load_model("models/delivery_predictor.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None

    logger.info("Inference server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global feature_store

    if feature_store:
        feature_store.disconnect()
        logger.info("Feature store disconnected")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if (feature_store and predictor) else "degraded",
        feature_store_connected=feature_store is not None and feature_store._connected,
        model_loaded=predictor is not None and predictor.model is not None,
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_delivery_time(request: PredictionRequest):
    """
    Predict delivery time.

    Supports two modes:
    1. Provide delivery_id to fetch features from feature store
    2. Provide features dict directly for immediate prediction
    """
    if not predictor or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = {}

    # Mode 1: Fetch features from feature store
    if request.delivery_id:
        if not feature_store or not feature_store._connected:
            raise HTTPException(status_code=503, detail="Feature store not available")

        try:
            # Fetch latest features for the delivery
            feature_names = predictor.feature_names
            feature_df = feature_store.get_online_features(
                entity_ids=[request.delivery_id],
                feature_names=feature_names
            )

            if feature_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No features found for delivery_id: {request.delivery_id}"
                )

            # Convert to dict
            for _, row in feature_df.iterrows():
                features[row['feature_name']] = row['value']

        except Exception as e:
            logger.error(f"Error fetching features: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Mode 2: Use provided features
    elif request.features:
        features = request.features
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either delivery_id or features"
        )

    # Validate required features
    missing_features = [f for f in predictor.feature_names if f not in features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_features}"
        )

    # Make prediction
    try:
        prediction = predictor.predict_delivery_time(features)

        return PredictionResponse(
            predicted_duration=prediction,
            delivery_id=request.delivery_id,
            features_used=features,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{entity_id}")
async def get_entity_features(entity_id: str, feature_names: Optional[str] = None):
    """
    Get latest features for an entity.

    Args:
        entity_id: Entity identifier
        feature_names: Comma-separated list of feature names (optional)
    """
    if not feature_store or not feature_store._connected:
        raise HTTPException(status_code=503, detail="Feature store not available")

    try:
        # Parse feature names
        if feature_names:
            feature_list = [f.strip() for f in feature_names.split(',')]
        else:
            feature_list = predictor.feature_names if predictor else None

        if not feature_list:
            raise HTTPException(
                status_code=400,
                detail="Must provide feature_names or have model loaded"
            )

        # Fetch features
        feature_df = feature_store.get_online_features(
            entity_ids=[entity_id],
            feature_names=feature_list
        )

        if feature_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for entity_id: {entity_id}"
            )

        return {
            "entity_id": entity_id,
            "features": feature_df.to_dict('records'),
            "timestamp": datetime.now()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/batch")
async def batch_predict(entity_ids: List[str]):
    """
    Batch prediction for multiple deliveries.

    Args:
        entity_ids: List of delivery entity IDs
    """
    if not predictor or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not feature_store or not feature_store._connected:
        raise HTTPException(status_code=503, detail="Feature store not available")

    predictions = []

    for entity_id in entity_ids:
        try:
            # Fetch features
            feature_df = feature_store.get_online_features(
                entity_ids=[entity_id],
                feature_names=predictor.feature_names
            )

            if feature_df.empty:
                predictions.append({
                    "entity_id": entity_id,
                    "status": "error",
                    "message": "No features found"
                })
                continue

            # Convert to dict
            features = {}
            for _, row in feature_df.iterrows():
                features[row['feature_name']] = row['value']

            # Predict
            prediction = predictor.predict_delivery_time(features)

            predictions.append({
                "entity_id": entity_id,
                "status": "success",
                "predicted_duration": prediction,
                "features": features
            })

        except Exception as e:
            predictions.append({
                "entity_id": entity_id,
                "status": "error",
                "message": str(e)
            })

    return {
        "predictions": predictions,
        "timestamp": datetime.now()
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the inference server.

    Args:
        host: Host address
        port: Port number
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()

