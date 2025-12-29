"""
Prediction endpoint for Vercel
"""
from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_predictor, 
    get_data_source_cached, 
    log_event, 
    get_logger
)
from aqi_utils import build_features_from_recent
import pickle
import numpy as np

def vector_for_model_serverless(model_type, features, county_row, when, base_path):
    """
    Serverless-compatible version of vector_for_model with absolute paths
    """
    state_code = county_row.get("state_code", None)
    if state_code is None:
        state_code = county_row.get("State Code", None)

    county_code = county_row.get("county_code", None)
    if county_code is None:
        county_code = county_row.get("County Code", None)

    if state_code is None or county_code is None:
        raise KeyError(
            "Missing required fields for model vectorization. "
            "Expected columns: 'state_code' and 'county_code' "
            "(or legacy 'State Code' and 'County Code')."
        )

    if model_type == "prototype":
        X = np.array([[
            state_code,
            county_code,
            0, 0, 0, 0, 0,
            features["aqi_lag_1"],
            features["aqi_lag_3"],
            features["aqi_lag_7"],
        ]])
        pipeline_path = os.path.join(base_path, "prototype_pipeline.pkl")
        # If not found, try to download
        if not os.path.exists(pipeline_path):
            import urllib.request
            import tempfile
            tmp_dir = os.path.join(tempfile.gettempdir(), 'clap_models')
            os.makedirs(tmp_dir, exist_ok=True)
            pipeline_path = os.path.join(tmp_dir, "prototype_pipeline.pkl")
            if not os.path.exists(pipeline_path):
                github_base = os.getenv('GITHUB_BASE', 'https://raw.githubusercontent.com/cchung7/clap_v1.2/main')
                model_url = os.getenv('MODEL_BASE_URL', f'{github_base}/models/')
                url = f"{model_url}prototype_pipeline.pkl"
                urllib.request.urlretrieve(url, pipeline_path)
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

    elif model_type == "balanced":
        X = np.array([[
            state_code,                               
            county_code,                               
            features["aqi_lag_1"],                  
            features["aqi_lag_3"],                
            features["aqi_lag_7"],                       
            features["aqi_lag_14"],                      
            features["aqi_rolling_7"],                   
            when.weekday(),                              
            when.month,                                  
            features.get("aqi_rolling_3", features["aqi_lag_1"]),  
            features.get("aqi_std_7", 0.0),                        
        ]])
        pipeline_path = os.path.join(base_path, "balanced_pipeline.pkl")
        # If not found, try to download
        if not os.path.exists(pipeline_path):
            import urllib.request
            import tempfile
            tmp_dir = os.path.join(tempfile.gettempdir(), 'clap_models')
            os.makedirs(tmp_dir, exist_ok=True)
            pipeline_path = os.path.join(tmp_dir, "balanced_pipeline.pkl")
            if not os.path.exists(pipeline_path):
                github_base = os.getenv('GITHUB_BASE', 'https://raw.githubusercontent.com/cchung7/clap_v1.2/main')
                model_url = os.getenv('MODEL_BASE_URL', f'{github_base}/models/')
                url = f"{model_url}balanced_pipeline.pkl"
                urllib.request.urlretrieve(url, pipeline_path)
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return pipeline.transform(X)

def iterative_forecast_serverless(selected_predictor, model_type, recent_df, base_features, days, county, state):
    """
    Serverless-compatible version of iterative_forecast
    """
    from datetime import timedelta
    predictions = []
    current_date = datetime.utcnow()
    current_features = base_features.copy()
    county_row = recent_df.iloc[0]
    
    # Get absolute path to models directory
    models_path = os.path.join(os.path.dirname(__file__), '..', 'models')

    for day in range(days):
        forecast_date = current_date + timedelta(days=day + 1)
        X_day_scaled = vector_for_model_serverless(model_type, current_features, county_row, forecast_date, models_path)
        
        day_pred = selected_predictor.forecast(
            X_day_scaled,
            county_name=county,
            state_name=state,
            forecast_date=forecast_date,
            store_predictions=False
        )
        predictions.append(day_pred)

        # Feedback loop: shift lags forward
        yhat = day_pred["predicted_aqi"]
        current_features["aqi_lag_14"] = current_features.get("aqi_lag_7", yhat)
        current_features["aqi_lag_7"] = current_features.get("aqi_lag_3", yhat)
        current_features["aqi_lag_3"] = current_features.get("aqi_lag_1", yhat)
        current_features["aqi_lag_1"] = yhat

        # Rolling updates
        if "aqi_rolling_7" in current_features:
            current_features["aqi_rolling_7"] = (current_features["aqi_rolling_7"] * 6 + yhat) / 7
        if "aqi_rolling_14" in current_features:
            current_features["aqi_rolling_14"] = (current_features["aqi_rolling_14"] * 13 + yhat) / 14
        if "aqi_rolling_30" in current_features:
            current_features["aqi_rolling_30"] = (current_features["aqi_rolling_30"] * 29 + yhat) / 30

    return predictions

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            t0 = datetime.utcnow()
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode('utf-8'))
            
            county = payload.get("county")
            state = payload.get("state")
            model_type = payload.get("model", "balanced")
            days_input = payload.get("days", 1)
            
            # Validate days parameter
            try:
                days = int(days_input)
            except (ValueError, TypeError):
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Invalid 'days' parameter: '{days_input}'. Must be an integer."
                }).encode())
                return
            
            # Valid days values: 1, 3, 7, or 14
            valid_days = [1, 3, 7, 14]
            if days not in valid_days:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Invalid 'days' value: {days}. Must be one of: {', '.join(map(str, valid_days))}"
                }).encode())
                return

            log_event(logging.INFO, f"Prediction request: county={county}, state={state}, model={model_type}, days={days}",
                     operation="validation")

            if not county or not state:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "County and state are required"
                }).encode())
                return

            selected_predictor = get_predictor(model_type)
            if selected_predictor is None or selected_predictor.model is None:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"{model_type.title()} model not loaded. Please train model first."
                }).encode())
                return

            source = get_data_source_cached()
            if source is None:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "Data source not available"
                }).encode())
                return

            recent_df = source.get_recent_data_for_prediction(county, state, 30)
            if recent_df is None or len(recent_df) < 7:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Insufficient historical data for {county}, {state}. Need at least 7 days."
                }).encode())
                return

            features, recent_df = build_features_from_recent(recent_df)
            log_event(logging.INFO, f"Recent rows for features: {len(recent_df)}", operation="feature_generation")

            preds = iterative_forecast_serverless(selected_predictor, model_type, recent_df, features, days, county, state)

            elapsed_ms = int((datetime.utcnow() - t0).total_seconds() * 1000)
            log_event(logging.INFO, f"Prediction completed in {elapsed_ms} ms (days={days})", operation="prediction")

            # Helper function to serialize datetime
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            if days == 1:
                pred = preds[0]
                forecast_date_str = serialize_datetime(pred["forecast_date"])
                response = {
                    "success": True,
                    "county": county,
                    "state": state,
                    "forecast_date": forecast_date_str,
                    "prediction": {
                        "predicted_aqi": pred["predicted_aqi"],
                        "predicted_category": pred["predicted_category"],
                        "probabilities": pred["probabilities"],
                        "county_name": pred.get("county_name"),
                        "state_name": pred.get("state_name"),
                        "forecast_date": forecast_date_str
                    }
                }
            else:
                response = {
                    "success": True,
                    "county": county,
                    "state": state,
                    "forecast_days": days,
                    "predictions": [
                        {
                            "predicted_aqi": p["predicted_aqi"],
                            "predicted_category": p["predicted_category"],
                            "probabilities": p["probabilities"],
                            "county_name": p.get("county_name"),
                            "state_name": p.get("state_name"),
                            "forecast_date": serialize_datetime(p["forecast_date"])
                        }
                        for p in preds
                    ]
                }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            get_logger().exception("Error making prediction", extra={"operation": "prediction"})
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e)
            }).encode())

