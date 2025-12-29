"""
Model Metrics endpoint for Vercel
"""
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_predictor, log_event, get_logger

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        try:
            # Parse query parameters
            parsed_url = urlparse(self.path)
            params = parse_qs(parsed_url.query)
            model_type = params.get('model', ['balanced'])[0]
            
            log_event(logging.INFO, f"Metrics request: model={model_type}", operation="validation")
            predictor = get_predictor(model_type)
            
            if predictor and getattr(predictor, "metrics", None):
                response = {
                    "success": True,
                    "model_type": model_type,
                    "metrics": predictor.metrics,
                    "version": predictor.model_version
                }
                self.send_response(200)
            else:
                response = {
                    "success": False,
                    "error": f"No metrics available for {model_type} model"
                }
                self.send_response(404)
            
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            get_logger().exception("Error fetching metrics", extra={"operation": "validation"})
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e)
            }).encode())

