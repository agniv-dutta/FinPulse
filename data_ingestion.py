import yfinance as yf
import pandas as pd
import numpy as np
from database import SessionLocal, StockData, engine
from datetime import datetime, timedelta

def fetch_and_prepare_data(symbols):
    all_data = []
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        # Fetch 2 years of data to ensure we can calculate 52w high/low and rolling metrics
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        
        if df.empty:
            print(f"No data found for {symbol}")
            continue

        # Data Cleaning
        df = df.dropna()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) # Remove timezone for SQLite compatibility
        
        # Calculations
        df['daily_return'] = (df['Close'] - df['Open']) / df['Open']
        df['ma_7'] = df['Close'].rolling(window=7).mean()
        df['high_52w'] = df['Close'].rolling(window=252, min_periods=1).max()
        df['low_52w'] = df['Close'].rolling(window=252, min_periods=1).min()
        df['volatility_score'] = df['daily_return'].rolling(window=30).std()
        
        # Fill NaNs from rolling metrics with 0 or local values
        df = df.fillna(0)
        
        for index, row in df.iterrows():
            stock_entry = StockData(
                symbol=symbol,
                date=row['Date'],
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                daily_return=float(row['daily_return']),
                ma_7=float(row['ma_7']),
                high_52w=float(row['high_52w']),
                low_52w=float(row['low_52w']),
                volatility_score=float(row['volatility_score'])
            )
            all_data.append(stock_entry)
            
    return all_data

def save_to_db(data):
    db = SessionLocal()
    try:
        # Clear existing data to avoid duplicates for this assessment
        db.query(StockData).delete()
        db.bulk_save_objects(data)
        db.commit()
        print(f"Successfully saved {len(data)} records to database.")
    except Exception as e:
        print(f"Error saving to database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]
    data = fetch_and_prepare_data(symbols)
    if data:
        save_to_db(data)
    else:
        print("No data fetched.")
