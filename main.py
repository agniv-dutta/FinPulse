from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from database import SessionLocal, StockData
from sqlalchemy import func
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from typing import List
import os

app = FastAPI(
    title="FinPulse API",
    description="Stock Data Intelligence Dashboard — NSE market data, technical analysis, ML predictions & performance tracking for Indian equities",
    version="1.0.0"
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import FileResponse

@app.get("/", response_class=FileResponse, tags=["UI"])
async def read_root():
    """Serves the FinPulse dashboard frontend."""
    return os.path.join("static", "index.html")

@app.get("/companies", tags=["Data"])
def get_companies():
    """Retrieve list of all available stock symbols and company names.
    
    Returns:
        list: Array of {symbol, name} objects for all 5 tracked companies
    """
    db = SessionLocal()
    companies = db.query(StockData.symbol).distinct().all()
    # Mocking names for this assessment
    company_names = {
        "RELIANCE.NS": "Reliance Industries Ltd",
        "TCS.NS": "Tata Consultancy Services Ltd",
        "INFY.NS": "Infosys Ltd",
        "HDFCBANK.NS": "HDFC Bank Ltd",
        "WIPRO.NS": "Wipro Ltd"
    }
    db.close()
    return [{"symbol": symbol[0], "name": company_names.get(symbol[0], symbol[0])} for symbol in companies]

@app.get("/data/{symbol}", tags=["Data"])
def get_stock_data(symbol: str, days: int = 30):
    """Fetch historical stock data with technical indicators.
    
    Args:
        symbol: Stock ticker symbol (e.g., RELIANCE.NS)
        days: Number of historical days (30, 90, or 365)
    
    Returns:
        list: Array of OHLCV data with daily_return, 7_day_ma, 52_week_high/low, volatility_score
    """
    # Validate days parameter
    if days not in [30, 90, 365]:
        raise HTTPException(status_code=400, detail="days must be 30, 90, or 365")
    
    db = SessionLocal()
    data = db.query(StockData).filter(StockData.symbol == symbol).order_by(StockData.date.desc()).limit(days).all()
    db.close()
    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    # Return in chronological order
    result = sorted([
        {
            "date": d.date.strftime("%Y-%m-%d"),
            "open": d.open, "high": d.high, "low": d.low, "close": d.close, 
            "volume": d.volume, "daily_return": d.daily_return, 
            "7_day_ma": d.ma_7 if d.ma_7 is not None and not pd.isna(d.ma_7) else None,
            "52_week_high": d.high_52w if d.high_52w is not None and not pd.isna(d.high_52w) else None,
            "52_week_low": d.low_52w if d.low_52w is not None and not pd.isna(d.low_52w) else None,
            "volatility_score": d.volatility_score if d.volatility_score is not None and not pd.isna(d.volatility_score) else None
        } for d in data
    ], key=lambda x: x['date'])
    
    return result

@app.get("/summary/{symbol}", tags=["Data"])
def get_summary(symbol: str):
    """Get 52-week statistics and average closing price.
    
    Args:
        symbol: Stock ticker symbol (e.g., INFY.NS)
    
    Returns:
        dict: Contains high_52w, low_52w, avg_close
    """
    db = SessionLocal()
    stats = db.query(
        func.max(StockData.high_52w), 
        func.min(StockData.low_52w), 
        func.avg(StockData.close)
    ).filter(StockData.symbol == symbol).first()
    db.close()
    if not stats:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return {
        "high_52w": stats[0],
        "low_52w": stats[1],
        "avg_close": round(stats[2], 2)
    }

@app.get("/compare", tags=["Analysis"])
def compare_stocks(symbol1: str, symbol2: str):
    """Compare two stocks side-by-side with last 30 days of data.
    
    Args:
        symbol1: First stock ticker symbol
        symbol2: Second stock ticker symbol
    
    Returns:
        dict: Keys are symbol names, values are arrays of {date, close, daily_return} objects
    """
    db = SessionLocal()
    
    # Fetch data for both symbols
    data1 = db.query(StockData).filter(StockData.symbol == symbol1).order_by(StockData.date.desc()).limit(30).all()
    data2 = db.query(StockData).filter(StockData.symbol == symbol2).order_by(StockData.date.desc()).limit(30).all()
    db.close()
    
    # Handle missing symbols
    if not data1:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol1} not found")
    if not data2:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol2} not found")
    
    # Format data for each symbol in chronological order
    result1 = sorted([
        {
            "date": d.date.strftime("%Y-%m-%d"),
            "close": d.close,
            "daily_return": d.daily_return if d.daily_return is not None and not pd.isna(d.daily_return) else None
        }
        for d in data1
    ], key=lambda x: x['date'])
    
    result2 = sorted([
        {
            "date": d.date.strftime("%Y-%m-%d"),
            "close": d.close,
            "daily_return": d.daily_return if d.daily_return is not None and not pd.isna(d.daily_return) else None
        }
        for d in data2
    ], key=lambda x: x['date'])
    
    return {
        symbol1: result1,
        symbol2: result2
    }

@app.get("/predict/{symbol}", tags=["Analysis"])
def predict_stock(symbol: str):
    """Generate 7-day price forecast using Linear Regression on 60 days of history.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        list: Array of {date, prediction} objects for next 7 trading days
    """
    db = SessionLocal()
    data = db.query(StockData.date, StockData.close).filter(StockData.symbol == symbol).order_by(StockData.date.desc()).limit(60).all()
    db.close()
    
    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    
    df = pd.DataFrame(data, columns=['date', 'close']).sort_values('date')
    df['day_index'] = np.arange(len(df))
    
    X = df[['day_index']].values
    y = df['close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
    predictions = model.predict(next_days)
    
    last_date = df['date'].iloc[-1]
    prediction_results = []
    for i, pred in enumerate(predictions):
        future_date = last_date + pd.Timedelta(days=i+1)
        prediction_results.append({"date": future_date.strftime("%Y-%m-%d"), "prediction": round(pred, 2)})
        
    return prediction_results

@app.get("/top_performers", tags=["Data"])
def get_top_performers():
    """Get today's top 3 gainers and top 3 losers by daily return.
    
    Returns:
        dict: Contains 'gainers' and 'losers' arrays with symbol and return percentage
    """
    db = SessionLocal()
    # Get the latest date available in the DB
    latest_date = db.query(func.max(StockData.date)).scalar()
    if not latest_date:
        db.close()
        return {"gainers": [], "losers": []}
    
    latest_data = db.query(StockData.symbol, StockData.daily_return).filter(StockData.date == latest_date).all()
    db.close()
    
    sorted_data = sorted(latest_data, key=lambda x: x[1], reverse=True)
    gainers = [{"symbol": s, "return": round(r * 100, 2)} for s, r in sorted_data[:3]]
    losers = [{"symbol": s, "return": round(r * 100, 2)} for s, r in sorted_data[-3:]]
    
    return {"gainers": gainers, "losers": sorted(losers, key=lambda x: x['return'])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
