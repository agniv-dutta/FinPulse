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

app = FastAPI(title="FinPulse API")

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

@app.get("/", response_class=FileResponse)
async def read_root():
    return os.path.join("static", "index.html")

@app.get("/companies")
def get_companies():
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

@app.get("/data/{symbol}")
def get_stock_data(symbol: str):
    db = SessionLocal()
    data = db.query(StockData).filter(StockData.symbol == symbol).order_by(StockData.date.desc()).limit(365).all()
    db.close()
    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    # Return in chronological order
    return sorted([
        {
            "date": d.date.strftime("%Y-%m-%d"),
            "open": d.open, "high": d.high, "low": d.low, "close": d.close, 
            "volume": d.volume, "daily_return": d.daily_return, 
            "ma_7": d.ma_7, "volatility": d.volatility_score
        } for d in data
    ], key=lambda x: x['date'])

@app.get("/summary/{symbol}")
def get_summary(symbol: str):
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

@app.get("/compare")
def compare_stocks(symbol1: str, symbol2: str):
    db = SessionLocal()
    data1 = db.query(StockData.date, StockData.close).filter(StockData.symbol == symbol1).order_by(StockData.date.desc()).limit(30).all()
    data2 = db.query(StockData.date, StockData.close).filter(StockData.symbol == symbol2).order_by(StockData.date.desc()).limit(30).all()
    db.close()
    
    # Map by date to align
    res1 = {d[0].strftime("%Y-%m-%d"): d[1] for d in data1}
    res2 = {d[0].strftime("%Y-%m-%d"): d[1] for d in data2}
    
    dates = sorted(list(set(res1.keys()) | set(res2.keys())))
    
    return [
        {"date": d, "symbol1": res1.get(d), "symbol2": res2.get(d)}
        for d in dates
    ]

@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
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

@app.get("/top_performers")
def get_top_performers():
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
