"""
Featurama - The Feature Store of Tomorrow!

A high-performance feature store built on ScyllaDB for managing
millions of high-cardinality features in ML pipelines.
"""

__version__ = "1.0.0"
__author__ = "Planet Express Crew"

from featurama.core.feature_store import FeatureStore

__all__ = ["FeatureStore"]

