# FinPulse - Stock Data Intelligence Dashboard

## Project Overview
FinPulse is a Python-based stock data intelligence dashboard developed as an internship assignment. It collects historical market data, computes practical analytics such as moving averages and volatility, and serves both API endpoints and a browser dashboard for quick exploration. The goal is to demonstrate an end-to-end data workflow: ingestion, storage, analysis, and lightweight forecasting.

## Tech Stack
- Python
- FastAPI
- SQLite
- yfinance
- Pandas
- NumPy
- scikit-learn
- Chart.js

## Project Structure
```text
FinPulse/
|-- main.py
|-- data_ingestion.py
|-- database.py
|-- requirements.txt
|-- Dockerfile
`-- static/
    `-- index.html
```

## Setup & Installation
1. Clone the repository.
```bash
git clone <your-repo-url>
cd FinPulse
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Ingest stock data into SQLite.
```bash
python data_ingestion.py
```

4. Start the FastAPI server.
```bash
uvicorn main:app --reload
```

5. Open the application in your browser.
```text
http://127.0.0.1:8000
```

## API Endpoints
| Endpoint | Method | Description | Example Response |
|---|---|---|---|
| /companies | GET | Returns available stock symbols and display names from the database. | [{"symbol":"RELIANCE.NS","name":"Reliance Industries Ltd"}] |
| /data/{symbol} | GET | Returns up to the latest 365 records with OHLCV and computed metrics for a symbol. | [{"date":"2026-03-28","open":1234.5,"high":1240.0,"low":1225.1,"close":1238.9,"volume":1234567,"daily_return":0.0036,"ma_7":1228.4,"volatility":0.012}] |
| /summary/{symbol} | GET | Returns aggregate summary values: 52-week high, 52-week low, and average close. | {"high_52w":1560.3,"low_52w":1201.2,"avg_close":1378.44} |
| /compare?symbol1=RELIANCE.NS&symbol2=TCS.NS | GET | Compares recent close prices of two symbols, aligned by date. | [{"date":"2026-03-28","symbol1":1238.9,"symbol2":4012.7}] |
| /predict/{symbol} | GET | Predicts the next 7 days of closing prices using linear regression on recent history. | [{"date":"2026-03-29","prediction":1241.27}] |

## Custom Metrics Explained
### Volatility Score
FinPulse computes volatility as the 30-day rolling standard deviation of daily returns:

$$
\text{volatility\_score}_t = \text{std}\left(r_{t-29}, r_{t-28}, \dots, r_t\right), \quad
r_t = \frac{\text{Close}_t - \text{Open}_t}{\text{Open}_t}
$$

This metric is useful because it captures short-term risk. Higher volatility often means larger price swings, which can indicate higher uncertainty, stronger momentum, or greater opportunity depending on strategy.

### Linear Regression Prediction
For forecasting, the project fits a scikit-learn Linear Regression model on the latest closing prices (with day index as the feature) and extrapolates the next 7 days. It is a simple baseline model that highlights trend direction and serves as an interpretable starting point for more advanced forecasting methods.

## Docker Usage
```bash
docker build -t finpulse .
docker run -p 8000:8000 finpulse
```

## Screenshots
Add screenshots here.

## Insights
- Technology stocks and diversified large-cap symbols in the sample frequently show different volatility profiles even when short-term trend direction appears similar.
- A simple linear trend model often tracks direction reasonably in stable periods, but prediction reliability decreases during abrupt market swings, reinforcing the need to read forecasts alongside volatility.
