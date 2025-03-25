import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configurable file path for trade_signal.txt (default path for local development)
SIGNAL_FILE_PATH = "trade_signal.txt"  # Change this path when deploying to a server

def initialize_mt5():
    """
    Initialize connection to MetaTrader 5.
    Returns True if successful, False otherwise.
    """
    try:
        if not mt5.initialize():
            logger.error("Failed to connect to MT5")
            return False
        logger.info("Successfully logged into MT5")
        return True
    except Exception as e:
        logger.error(f"Error initializing MT5: {str(e)}")
        return False

def fetch_historical_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, start_date=None, end_date=None):
    """
    Fetch historical data from MT5 for the given symbol and timeframe.
    Returns a DataFrame with the data, or None if the fetch fails.
    """
    try:
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.error("No data returned from MT5. Check symbol, timeframe, or date range.")
            return None
        
        df = pd.DataFrame(rates)
        logger.debug(f"Fetched {len(df)} rows of data from MT5 for {symbol}")
        
        # Ensure 'time' column exists and convert to datetime
        if 'time' not in df.columns:
            logger.error(f"'time' column not found in data. Available columns: {df.columns.tolist()}")
            return None
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data from MT5: {str(e)}")
        return None

def fetch_external_data(symbol="EURUSD=X", period="1mo"):
    """
    Fetch external data using yfinance.
    Returns a DataFrame with the data, or None if the fetch fails.
    """
    try:
        eurusd = yf.Ticker(symbol)
        external_data = eurusd.history(period=period)
        logger.debug(f"Fetched external data from yfinance:\n{external_data.tail()}")
        return external_data
    except Exception as e:
        logger.error(f"Error fetching external data from yfinance: {str(e)}")
        return None

def prepare_features(df):
    """
    Prepare features for machine learning model.
    Returns the DataFrame with features and signals, or None if preparation fails.
    """
    try:
        df['returns'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['signal'] = np.where(df['ma5'] > df['ma20'], 1, -1)  # 1 for BUY, -1 for SELL
        df = df.dropna()
        logger.debug(f"Prepared features: {df[['returns', 'ma5', 'ma20', 'signal']].tail()}")
        return df
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None

def train_model(X, y):
    """
    Train a Random Forest Classifier model.
    Returns the trained model, or None if training fails.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.info(f"Model accuracy: {accuracy}")
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

def predict_signal(model, X):
    """
    Predict the latest trading signal using the trained model.
    Returns the signal (1 for BUY, -1 for SELL), or None if prediction fails.
    """
    try:
        latest_data = X.iloc[-1:].copy()
        signal = model.predict(latest_data)[0]
        logger.info(f"Predicted signal: {signal}")
        return signal
    except Exception as e:
        logger.error(f"Error predicting signal: {str(e)}")
        return None

def write_signal_to_file(signal, file_path=SIGNAL_FILE_PATH):
    """
    Write the trading signal to a file.
    Returns True if successful, False otherwise.
    """
    try:
        signal_text = "BUY" if signal == 1 else "SELL"
        with open(file_path, "w") as f:
            f.write(signal_text)
        logger.info(f"Signal written to file: {signal_text} at {file_path}")
        return True
    except PermissionError as e:
        logger.error(f"Permission error writing to file: {e}. Try running as administrator or closing MT5.")
        return False
    except Exception as e:
        logger.error(f"Failed to write signal file: {e}")
        return False

def main():
    """
    Main function to generate a trading signal and write it to a file.
    Returns the signal text ("BUY", "SELL", or "HOLD" on error).
    """
    # Initialize MT5
    if not initialize_mt5():
        logger.error("MT5 initialization failed, returning default signal")
        return "HOLD"

    try:
        # Fetch historical data from MT5
        df = fetch_historical_data("EURUSD", mt5.TIMEFRAME_H1)
        if df is None:
            logger.error("Failed to fetch historical data, returning default signal")
            return "HOLD"

        # Fetch external data (optional, for logging purposes)
        external_data = fetch_external_data("EURUSD=X", "1mo")
        if external_data is None:
            logger.warning("Failed to fetch external data, proceeding without it")

        # Prepare features for machine learning
        df = prepare_features(df)
        if df is None:
            logger.error("Failed to prepare features, returning default signal")
            return "HOLD"

        # Train the machine learning model
        X = df[['returns', 'ma5', 'ma20']]
        y = df['signal']
        model = train_model(X, y)
        if model is None:
            logger.error("Failed to train model, returning default signal")
            return "HOLD"

        # Predict the latest signal
        signal = predict_signal(model, X)
        if signal is None:
            logger.error("Failed to predict signal, returning default signal")
            return "HOLD"

        # Write the signal to file
        if not write_signal_to_file(signal):
            logger.error("Failed to write signal to file, returning default signal")
            return "HOLD"

        # Return the signal text
        signal_text = "BUY" if signal == 1 else "SELL"
        return signal_text

    finally:
        # Only shutdown MT5 if we initialized it
        if mt5.terminal_info():
            mt5.shutdown()
            logger.info("MT5 connection closed")

if __name__ == "__main__":
    signal = main()
    print(f"Generated signal: {signal}")