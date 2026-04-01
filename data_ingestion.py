import yfinance as yf
import pandas as pd
import numpy as np
from database import SessionLocal, StockData, engine
from datetime import datetime, timedelta

def fetch_and_prepare_data(symbols):
    """Fetch and prepare stock data for all symbols with proper calculations."""
    all_data = []
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        
        # Fetch 2 years of data with auto_adjust=True for corporate actions
        df = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
        
        if df.empty:
            print(f"  ✗ No data found for {symbol}")
            continue
        
        # Handle MultiIndex columns (flatten if needed)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Drop rows where close is NaN (use capital C since yfinance uses that)
        df = df.dropna(subset=['Close'])
        
        if df.empty:
            print(f"  ✗ No valid data for {symbol} after cleaning")
            continue
        
        # Reset index so date becomes a column
        df = df.reset_index()
        
        # Standardize all column names to lowercase (do this AFTER reset_index)
        df.columns = [str(col).lower() for col in df.columns]
        
        # Ensure date column exists and format it
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:
            print(f"  ✗ Date column not found for {symbol}")
            continue
        
        # Compute daily_return = (Close - Open) / Open
        df['daily_return'] = (df['close'] - df['open']) / df['open']
        
        # Compute 7-day moving average of Close
        df['ma_7'] = df['close'].rolling(window=7).mean()
        
        # Compute 52-week (252 trading days) high and low
        df['high_52w'] = df['close'].rolling(window=252, min_periods=1).max()
        df['low_52w'] = df['close'].rolling(window=252, min_periods=1).min()
        
        # Compute volatility_score = 30-day rolling std of daily_return
        df['volatility_score'] = df['daily_return'].rolling(window=30).std()
        
        # Replace NaN values with None for proper SQLite storage
        df = df.where(pd.notnull(df), None)
        
        # Insert records to database list
        row_count = 0
        for index, row in df.iterrows():
            stock_entry = StockData(
                symbol=symbol,
                date=row['date'],
                open=float(row['open']) if pd.notnull(row['open']) else None,
                high=float(row['high']) if pd.notnull(row['high']) else None,
                low=float(row['low']) if pd.notnull(row['low']) else None,
                close=float(row['close']) if pd.notnull(row['close']) else None,
                volume=int(row['volume']) if pd.notnull(row['volume']) else None,
                daily_return=float(row['daily_return']) if pd.notnull(row['daily_return']) else None,
                ma_7=float(row['ma_7']) if pd.notnull(row['ma_7']) else None,
                high_52w=float(row['high_52w']) if pd.notnull(row['high_52w']) else None,
                low_52w=float(row['low_52w']) if pd.notnull(row['low_52w']) else None,
                volatility_score=float(row['volatility_score']) if pd.notnull(row['volatility_score']) else None
            )
            all_data.append(stock_entry)
            row_count += 1
        
        print(f"  ✓ {symbol} — {row_count} rows loaded")
            
    return all_data


def save_to_db(data):
    """Save prepared stock data to SQLite database."""
    db = SessionLocal()
    try:
        # Clear existing data to avoid duplicates
        db.query(StockData).delete()
        db.bulk_save_objects(data)
        db.commit()
        print(f"\n✓ Successfully saved {len(data)} total records to database.")
    except Exception as e:
        print(f"\n✗ Error saving to database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]
    print(f"Starting data ingestion for {len(symbols)} stocks...\n")
    data = fetch_and_prepare_data(symbols)
    if data:
        save_to_db(data)
    else:
        print("\n✗ No data fetched.")
