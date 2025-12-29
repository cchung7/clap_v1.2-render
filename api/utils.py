"""
Shared utilities for Vercel serverless functions
Handles caching of models and data source
"""
import os
import sys
import logging
from datetime import datetime

# Import from local api files (not backend)
from ml_model import AQIPredictor
from data_source import get_data_source
from config import Config

# Global caches (persist across invocations within the same container)
_models_cache = {}
_data_cache = None
_logger = None

def get_logger():
    """Get or create logger"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("CLAP")
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        _logger.addHandler(handler)
    return _logger

def get_data_source_cached():
    """Get or create cached data source"""
    global _data_cache
    if _data_cache is None:
        # Use absolute path for serverless
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        _data_cache = get_data_source(data_path=data_path)
        get_logger().info(f"Data source loaded: {len(_data_cache.df)} records")
    return _data_cache

def get_predictor(model_key="balanced"):
    """Get or create cached predictor"""
    global _models_cache
    if model_key not in _models_cache:
        # Use /tmp for Vercel (models will be downloaded there if not found locally)
        import tempfile
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        # Check if local path exists, otherwise use temp directory
        if not os.path.exists(model_path):
            model_path = os.path.join(tempfile.gettempdir(), 'clap_models')
        
        predictor = AQIPredictor(model_path=model_path)
        
        # Load the model file (will download if not found)
        model_file = f"{model_key}_lightgbm_model.pkl"
        if predictor.load_model(model_file):
            _models_cache[model_key] = predictor
            get_logger().info(f"{model_key} model loaded successfully")
        else:
            get_logger().error(f"Failed to load {model_key} model")
            return None
    
    return _models_cache.get(model_key)

def log_event(level, msg, operation="general", **kv):
    """Log event"""
    logger = get_logger()
    logger.log(level, msg, extra={"operation": operation, **kv})

# Import aqi_utils functions for feature building
# Note: vector_for_model and iterative_forecast have serverless-compatible versions in predict.py
from aqi_utils import build_features_from_recent

