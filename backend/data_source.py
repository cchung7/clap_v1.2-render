"""
Data source adapter - reads from CSV files instead of database
Supports loading from local files or downloading from external URL
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
import urllib.request
import tempfile

logger = logging.getLogger(__name__)

# External URLs for data files from GitHub
GITHUB_BASE = 'https://raw.githubusercontent.com/cchung7/clap_v1.2/main'
DATA_URLS = {
    'daily_aqi_by_county_2024.csv': os.getenv('DATA_CSV_URL', f'{GITHUB_BASE}/data/daily_aqi_by_county_2024.csv'),
    'encoded_dataset.csv': os.getenv('ENCODED_CSV_URL', f'{GITHUB_BASE}/data/encoded_dataset.csv')
}

def download_file(url, local_path):
    """Download file from URL to local path"""
    try:
        logger.info(f"Downloading {url} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
        logger.info(f"Downloaded successfully: {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False


class CSVDataSource:
    """Read AQI data from CSV files"""
    
    def __init__(self, data_path='../data/'):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file (local or download from URL)"""
        try:
            # Try local file first
            csv_file = os.path.join(self.data_path, 'daily_aqi_by_county_2024.csv')
            
            # If not found locally, try to download
            if not os.path.exists(csv_file):
                logger.info(f"Local file not found: {csv_file}, attempting download...")
                # Use /tmp for Vercel (writable directory)
                tmp_dir = os.path.join(tempfile.gettempdir(), 'clap_data')
                os.makedirs(tmp_dir, exist_ok=True)
                csv_file = os.path.join(tmp_dir, 'daily_aqi_by_county_2024.csv')
                
                if not os.path.exists(csv_file):
                    if not download_file(DATA_URLS['daily_aqi_by_county_2024.csv'], csv_file):
                        # Fall back to encoded dataset
                        csv_file = os.path.join(tmp_dir, 'encoded_dataset.csv')
                        if not os.path.exists(csv_file):
                            download_file(DATA_URLS['encoded_dataset.csv'], csv_file)
            
            # Fallback to encoded dataset if daily file doesn't exist
            if not os.path.exists(csv_file):
                csv_file = os.path.join(self.data_path, 'encoded_dataset.csv')
                if not os.path.exists(csv_file):
                    tmp_dir = os.path.join(tempfile.gettempdir(), 'clap_data')
                    csv_file = os.path.join(tmp_dir, 'encoded_dataset.csv')
                    if not os.path.exists(csv_file):
                        download_file(DATA_URLS['encoded_dataset.csv'], csv_file)
            
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Could not find or download CSV file")
            
            logger.info(f"Loading data from {csv_file}")
            self.df = pd.read_csv(csv_file, parse_dates=['Date'])
            
            # Rename columns to match expected format
            if 'county Name' in self.df.columns:
                self.df = self.df.rename(columns={'county Name': 'county_name'})
            if 'State Name' in self.df.columns:
                self.df = self.df.rename(columns={'State Name': 'state_name'})
            if 'State Code' in self.df.columns:
                self.df = self.df.rename(columns={'State Code': 'state_code'})
            if 'County Code' in self.df.columns:
                self.df = self.df.rename(columns={'County Code': 'county_code'})
            if 'Category' in self.df.columns:
                self.df = self.df.rename(columns={'Category': 'category'})
            if 'Defining Parameter' in self.df.columns:
                self.df = self.df.rename(columns={'Defining Parameter': 'defining_parameter'})
            
            # Create county_name if it doesn't exist (for encoded dataset)
            if 'county_name' not in self.df.columns and 'county_code' in self.df.columns:
                # Create a mapping from county codes to names
                self.df['county_name'] = self.df['county_code'].astype(str) + '_County'
            
            logger.info(f"Loaded {len(self.df)} records from CSV")
            logger.info(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            logger.info(f"Counties: {self.df['county_name'].nunique()}")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {str(e)}")
            raise
    
    def get_counties(self):
        """Get list of available counties"""
        if self.df is None:
            return []
        
        counties = self.df.groupby(['county_name', 'state_name']).size().reset_index(name='count')
        counties = counties.sort_values(['state_name', 'county_name'])
        
        result = []
        for _, row in counties.iterrows():
            result.append({
                'county': row['county_name'],
                'state': row['state_name'],
                'display_name': f"{row['county_name']}, {row['state_name']}"
            })
        
        return result
    
    def get_historical_data(self, county, state, days=30):
        """Get historical AQI data for a county"""
        if self.df is None:
            return []
        
        # Filter by county and state
        filtered = self.df[
            (self.df['county_name'] == county) & 
            (self.df['state_name'] == state)
        ].copy()
        
        if len(filtered) == 0:
            return []
        
        # Get the most recent N days
        filtered = filtered.sort_values('Date', ascending=False).head(days)
        filtered = filtered.sort_values('Date')  # Sort chronologically for display
        
        result = []
        for _, row in filtered.iterrows():
            result.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'aqi': int(row['AQI']) if pd.notna(row['AQI']) else None,
                'category': row.get('category', 'Unknown'),
                'defining_parameter': row.get('defining_parameter', 'Unknown')
            })
        
        return result
    
    def get_recent_data_for_prediction(self, county, state, days=30):
        """Get recent data for generating prediction features"""
        if self.df is None:
            return None
        
        # Filter by county and state
        filtered = self.df[
            (self.df['county_name'] == county) & 
            (self.df['state_name'] == state)
        ].copy()
        
        if len(filtered) == 0:
            return None
        
        # Get the most recent N days
        filtered = filtered.sort_values('Date', ascending=False).head(days)
        filtered = filtered.sort_values('Date')  # Sort chronologically
        
        return filtered


# Global instance
csv_data_source = None


def get_data_source(data_path='../data/'):
    """Get or create the CSV data source"""
    global csv_data_source
    if csv_data_source is None:
        csv_data_source = CSVDataSource(data_path)
    return csv_data_source

