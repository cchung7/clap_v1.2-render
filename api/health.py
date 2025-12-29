"""
Health check endpoint for Vercel
"""
from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_predictor

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
            predictor = get_predictor("balanced")
            model_loaded = bool(predictor and predictor.model is not None)
            
            response = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": model_loaded,
                "database_connected": False  # CSV mode
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

