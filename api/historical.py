"""
Historical AQI endpoint for Vercel
"""
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_data_source_cached, log_event, get_logger

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
            
            county = params.get('county', [None])[0]
            state = params.get('state', [None])[0]
            days = int(params.get('days', ['30'])[0])
            
            log_event(logging.INFO, f"Historical request: county={county}, state={state}, days={days}", 
                     operation="validation")
            
            if not county or not state:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": "County and state parameters are required"
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
            
            historical_data = source.get_historical_data(county, state, days)
            log_event(logging.INFO, f"Historical rows returned: {len(historical_data)}", operation="ingestion")
            
            response = {
                "success": True,
                "county": county,
                "state": state,
                "days": days,
                "data": historical_data,
                "count": len(historical_data),
                "source": "csv"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            get_logger().exception("Error fetching historical data", extra={"operation": "ingestion"})
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e)
            }).encode())

