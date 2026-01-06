# CLAP_V1.2 - County-level Air Quality Prediction System

The CLAP system is a web-based predictive analytics application that forecasts the next-day average Air Quality Index (AQI) for U.S. counties using historical AQI measurements published by the U.S. Environmental Protection Agency (EPA). 

CLAP ingests EPA datasets, validates and preprocesses the dataset, generates lag-based and temporal features, and applies a trained machine-learning model to produce predictions with probability distributions across EPA-defined AQI categories and labels.

The dashboard provides intuitive access to next-day predictions, 30-day AQI trends, confidence scores, and category-based visualizations consistent with EPA color standards.

### URL
https://clap-v1-2-render.onrender.com

## Features

- **Next-Day Forecasting**: Predict AQI for the following day of the current time/date.
- **Two ML Models**: Balanced (new) & prototype (old)
- **Interactive Dashboard**: Reactive web interface with EPA-compliant color coding
- **County Coverage**: Over 770+ U.S. counties

## Model Performance

| Model | R² Score | MSE | Use Case |
|-------|----------|-----|----------|
| Balanced | 0.72 | 93.47 | **Recommended** - Best overall performance |
| Prototype | 0.46 | 177.72 | Baseline comparison |

### Prerequisites
- Python 3.12, 3.14
- Node.js v22.21.0+

## Render Settings

- **Repository**
   ```
   https://github.com/cchung7/clap_v1.2-render
   ```
- **Branch**
   ```
   main
   ```
- **Build Command**
   ```
   $  cd frontend && npm ci && npm run build && cd .. && pip install -r requirements.txt
   ```
- **Start Command**
   ```
   $  gunicorn --chdir backend --bind 0.0.0.0:$PORT app:app
   ```

## Project Structure

```
clap_v1.2-render/
├── api/                   # Backend API routes
├── data/                  # AQI datasets & Python Data Pipeline template
├── models/                # Trained ML models (.pkl)
├── frontend/              # React/Html/CSS web dashboard & Vite server
└── backend/               # Python API backend & Flask server
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/counties` - List available counties
- `GET /api/categories` - List AQI categories
- `GET /api/aqi/historical` - Historical AQI data
- `POST /api/aqi/predict` - Generate predictions
- `GET /api/model/metrics` - Model performance metrics

## Usage Examples

### Single-Day Prediction
```bash
curl -X POST http://localhost:5001/api/aqi/predict \
  -H "Content-Type: application/json" \
  -d '{"county": "Dallas", "state": "Texas", "model": "balanced", "days": 1}'
```

## Technical Details

- **Framework**: Flask (Python), Vite (React)
- **ML Library**: LightGBM
- **Frontend**: HTML5, CSS3, JavaScript (React), Chart.js
- **Data Source**: EPA 2024 Daily AQI dataset (Last Updated: Sept 2025)
- **Deployment**: Render
- **Repo**: https://github.com/cchung7/clap_v1.2-render

---

For requirements information, see [REQUIREMENTS.md](REQUIREMENTS.md)
