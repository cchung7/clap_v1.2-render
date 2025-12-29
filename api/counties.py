"""
Counties endpoint for Vercel
"""
from http.server import BaseHTTPRequestHandler
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_data_source_cached, log_event, get_logger
import logging

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
            log_event(logging.INFO, "Fetching counties", operation="ingestion")
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
            
            counties = source.get_counties()
            log_event(logging.INFO, f"Counties retrieved: {len(counties)}", operation="ingestion")
            
            response = {
                "success": True,
                "counties": counties,
                "count": len(counties),
                "source": "csv"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            get_logger().exception("Error fetching counties", extra={"operation": "ingestion"})
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e)
            }).encode())

