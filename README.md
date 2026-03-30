# FinPulse: Stock Data Intelligence Dashboard

FinPulse is a comprehensive stock monitoring and intelligence dashboard designed for an Indian fintech internship assessment. It fetches real-time data from the NSE/BSE using `yfinance`, performs technical analysis, and provides predictive insights using Machine Learning.

## 🚀 Features

- **Real-Time Data Ingestion**: Automatically fetches and cleans data for major Indian companies (RELIANCE, TCS, INFY, HDFCBANK, WIPRO).
- **Technical Metrics**: 
  - `Daily Return`: Percentage change from Open to Close.
  - `7-Day MA`: 7-day rolling moving average for trend smoothing.
  - `52-Week High/Low`: Tracks yearly extremes.
  - `Volatility Score`: 30-day rolling standard deviation of daily returns (custom metric).
- **Price Prediction**: Uses Scikit-Learn Linear Regression to predict the next 7 days of closing prices.
- **Comparison Tool**: Compare two stocks side-by-side to analyze relative performance.
- **Top Gainers / Losers**: Real-time identification of market performers.
- **Premium UI**: Modern dark-themed dashboard built with Chart.js and vanilla CSS/JS.

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Ingestion
This script will fetch historical data and populate the SQLite database.
```bash
python data_ingestion.py
```

### 3. Start the Server
```bash
uvicorn main:app --reload
```
Navigate to `http://127.0.0.1:8000` to view the dashboard.

## 🐳 Running with Docker
```bash
docker build -t finpulse .
docker run -p 8000:8000 finpulse
```

## 📊 API Endpoints

- `GET /`: Serves the frontend dashboard.
- `GET /companies`: List of available stock symbols and names.
- `GET /data/{symbol}`: Last 1 year of record history and calculated metrics.
- `GET /summary/{symbol}`: 52-week high, low, and average prices.
- `GET /compare?symbol1=X&symbol2=Y`: Side-by-side historical comparison.
- `GET /predict/{symbol}`: 7-day ML-based price prediction.
- `GET /top_performers`: Today's gainers and losers.
- `GET /docs`: Interactive Swagger UI documentation.

## 📈 Custom Metric: Volatility Score
The **Volatility Score** is calculated as the 30-day rolling standard deviation of the daily returns. This provides a quantifiable measure of risk, indicating how much the stock's price fluctuates relative to its recent average. A higher score signifies higher risk and potential opportunity.

## 🔮 Prediction Feature
The prediction feature utilizes a **Linear Regression** model from `scikit-learn`. It fits a trend line to the most recent 60 days of closing prices to extrapolate the likely trajectory for the next 7 days. This is displayed as a dashed line on the main chart.
