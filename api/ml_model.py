"""
Machine Learning Model for CLAP system (serverless version)
Implements LightGBM for AQI prediction
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import logging
from datetime import datetime
import json

# Import config locally
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AQIPredictor:
    """LightGBM-based AQI prediction model"""
    
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}
        
        # LightGBM hyperparameters
        self.params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
    
    def log_operation(self, operation, status, duration=None, details=None, error_msg=None):
        """Log model operation (serverless version - just logs, no database)"""
        logger.info(f"[{operation}] {status}: {details or ''} {f'({duration:.2f}s)' if duration else ''} {f'Error: {error_msg}' if error_msg else ''}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=1000):
        """Train LightGBM model"""
        import time
        start_time = time.time()
        
        try:
            logger.info("Training LightGBM model")
            
            # Store feature names
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = list(X_train.columns)
                X_train = X_train.values
            
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
                
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            # Train model
            logger.info(f"Training with {len(X_train)} samples, {X_train.shape[1]} features")
            
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_rounds,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            duration = time.time() - start_time
            logger.info(f"Model training completed in {duration:.2f}s")
            logger.info(f"Best iteration: {self.model.best_iteration}")
            
            self.log_operation('TRAIN', 'SUCCESS', duration,
                             f"Trained LightGBM model with {len(X_train)} samples")
            
            return self.model
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Training failed: {str(e)}")
            self.log_operation('TRAIN', 'ERROR', duration, error_msg=str(e))
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        import time
        start_time = time.time()
        
        try:
            logger.info("Evaluating model")
            
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Make predictions
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
            if isinstance(y_test, pd.Series):
                y_test = y_test.values
            
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'samples': len(y_test)
            }
            
            duration = time.time() - start_time
            
            logger.info("="*60)
            logger.info("Model Evaluation Metrics:")
            logger.info(f"  MSE (Mean Squared Error): {mse:.2f}")
            logger.info(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
            logger.info(f"  MAE (Mean Absolute Error): {mae:.2f}")
            logger.info(f"  R² (R-Squared): {r2:.4f}")
            logger.info(f"  Test samples: {len(y_test)}")
            logger.info("="*60)
            
            self.log_operation('EVALUATE', 'SUCCESS', duration,
                             f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return self.metrics
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Evaluation failed: {str(e)}")
            self.log_operation('EVALUATE', 'ERROR', duration, error_msg=str(e))
            raise
    
    def forecast(self, X_forecast, county_name=None, state_name=None, 
                 forecast_date=None, store_predictions=True):
        """Generate AQI predictions"""
        import time
        start_time = time.time()
        
        try:
            logger.info("Forecasting AQI")
            
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Make prediction
            if isinstance(X_forecast, pd.DataFrame):
                X_forecast = X_forecast.values
            
            predicted_aqi = self.model.predict(X_forecast)
            
            # Calculate category and probabilities
            results = []
            for i, aqi_value in enumerate(predicted_aqi):
                category, probabilities = self._calculate_category_probabilities(aqi_value)
                
                result = {
                    'predicted_aqi': float(aqi_value),
                    'predicted_category': category,
                    'probabilities': probabilities,
                    'county_name': county_name,
                    'state_name': state_name,
                    'forecast_date': forecast_date or datetime.utcnow()
                }
                results.append(result)
            
            duration = time.time() - start_time
            
            logger.info(f"Generated {len(results)} predictions in {duration:.2f}s")
            
            self.log_operation('FORECAST', 'SUCCESS', duration,
                             f"Generated {len(results)} predictions")
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Forecasting failed: {str(e)}")
            self.log_operation('FORECAST', 'ERROR', duration, error_msg=str(e))
            raise
    
    def _calculate_category_probabilities(self, aqi_value):
        """Calculate AQI category and probability distribution"""
        # Define AQI categories
        categories = [
            ('Good', 0, 50),
            ('Moderate', 51, 100),
            ('Unhealthy for Sensitive Groups', 101, 150),
            ('Unhealthy', 151, 200),
            ('Very Unhealthy', 201, 300),
            ('Hazardous', 301, 500)
        ]
        
        # Determine primary category
        category = 'Unknown'
        for cat_name, cat_min, cat_max in categories:
            if cat_min <= aqi_value <= cat_max:
                category = cat_name
                break
        
        if aqi_value > 500:
            category = 'Hazardous'
        
        # Calculate probability distribution (simplified using normal distribution)
        # Assume uncertainty of ±15 AQI units (based on RMSE ~13)
        std_dev = 15
        probabilities = {}
        
        for cat_name, cat_min, cat_max in categories:
            # Calculate probability that actual value falls in this category
            from scipy import stats
            lower_z = (cat_min - aqi_value) / std_dev
            upper_z = (cat_max - aqi_value) / std_dev
            prob = stats.norm.cdf(upper_z) - stats.norm.cdf(lower_z)
            probabilities[cat_name] = float(max(0, min(1, prob)))
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return category, probabilities
    
    def save_model(self, filename='lightgbm_model.pkl'):
        """Save trained model to file"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            filepath = os.path.join(self.model_path, filename)
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'params': self.params,
                'metrics': self.metrics,
                'version': self.model_version
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filename='lightgbm_model.pkl'):
        """Load trained model from file (local or download from URL)"""
        import urllib.request
        import tempfile
            
        try:
            filepath = os.path.join(self.model_path, filename)
            
            # If not found locally, try to download
            if not os.path.exists(filepath):
                logger.info(f"Model file not found locally: {filepath}, attempting download...")
                # Use /tmp for Vercel (writable directory)
                tmp_dir = os.path.join(tempfile.gettempdir(), 'clap_models')
                os.makedirs(tmp_dir, exist_ok=True)
                filepath = os.path.join(tmp_dir, filename)
                
                if not os.path.exists(filepath):
                    # Download from GitHub or external URL
                    github_base = os.getenv('GITHUB_BASE', 'https://raw.githubusercontent.com/cchung7/clap_v1.2/main')
                    model_url = os.getenv('MODEL_BASE_URL', f'{github_base}/models/')
                    url = f"{model_url}{filename}"
                    try:
                        logger.info(f"Downloading model from {url}")
                        urllib.request.urlretrieve(url, filepath)
                        logger.info(f"Model downloaded successfully: {filepath}")
                    except Exception as e:
                        logger.error(f"Failed to download model: {str(e)}")
                        raise FileNotFoundError(f"Could not find or download model file: {filename}")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names')
            self.params = model_data.get('params', self.params)
            self.metrics = model_data.get('metrics', {})
            self.model_version = model_data.get('version', 'loaded')
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Model version: {self.model_version}")
            
            if self.metrics:
                logger.info(f"Model metrics: MSE={self.metrics.get('mse', 'N/A'):.2f}, "
                          f"R²={self.metrics.get('r2', 'N/A'):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

