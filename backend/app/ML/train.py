import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure backend directory is in path (three levels up from backend/app/ML/train.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ML.features import generate_technical_features
from app.db import SessionLocal, EvaluationMetric

# Define path to save model
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_stock_model.json")

# List of Nifty 50 tickers for training data diversity
TRAIN_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "SHRIRAMFIN.NS"
]

def prepare_train_test_datasets():
    """Fetch historical daily candles, generate features, label targets, and split chronologically per ticker."""
    print("Fetching training data from yfinance...")
    train_data = []
    test_data = []
    
    for ticker in TRAIN_TICKERS:
        try:
            print(f" Downloading {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(period="5y")
            if df.empty:
                continue
                
            # Rename columns to lowercase standard
            df.columns = [col.lower() for col in df.columns]
            df["ticker"] = ticker
            
            # Generate technical indicators
            df = generate_technical_features(df)
            
            # Create date column from index
            df["date"] = pd.to_datetime(df.index)
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            
            # 1. Label target: Predict the 30-day forward return trend class
            # Forward return = (Close[t+30] - Close[t]) / Close[t]
            df["forward_return"] = (df["close"].shift(-30) - df["close"]) / df["close"].replace(0, 1e-9)
            
            # Replace inf with nan in forward_return
            df["forward_return"] = df["forward_return"].replace([np.inf, -np.inf], np.nan)
            
            # Drop the tail rows where we can't calculate forward returns
            df = df.dropna(subset=["forward_return"])
            
            # Label classes: 0 -> Bearish (< -3.0%), 1 -> Neutral (-3% to 3%), 2 -> Bullish (> 3.0%)
            df["target"] = 1  # Neutral default
            df.loc[df["forward_return"] > 0.03, "target"] = 2   # Bullish
            df.loc[df["forward_return"] < -0.03, "target"] = 0  # Bearish
            
            # Chronological Split: 80% Train, 20% Test
            split_idx = int(len(df) * 0.8)
            
            # To prevent overlapping label leakage (since target shifts forward by 30 days),
            # we introduce a 30-day gap before the test split.
            train_df = df.iloc[:split_idx - 30]
            test_df = df.iloc[split_idx:]
            
            train_data.append(train_df)
            test_data.append(test_df)
        except Exception as e:
            print(f"❌ Failed to download/process {ticker}: {e}")
            
    if not train_data or not test_data:
        raise ValueError("No training or test data could be collected.")
        
    combined_train = pd.concat(train_data, ignore_index=True)
    combined_test = pd.concat(test_data, ignore_index=True)
    return combined_train, combined_test

def train_model():
    """Train XGBoost model, evaluate accuracy, save weights, and log metrics in SQLite."""
    print("Starting ML Model training pipeline...")
    
    # Get prepared dataset split chronologically
    train_df, test_df = prepare_train_test_datasets()
    
    # Define features
    features = [
        "rsi_14", "macd_line", "macd_signal", "macd_hist", 
        "dist_sma_50", "bb_width", "price_change_5d", 
        "volume_change_5d"
    ]
    
    X_train = train_df[features]
    y_train = train_df["target"]
    
    X_test = test_df[features]
    y_test = test_df["target"]
    
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train class distribution: Neutral (1): {sum(y_train==1)}, Bullish (2): {sum(y_train==2)}, Bearish (0): {sum(y_train==0)}")
    print(f"Test class distribution:  Neutral (1): {sum(y_test==1)}, Bullish (2): {sum(y_test==2)}, Bearish (0): {sum(y_test==0)}")
    
    # Setup XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss"
    )
    
    # Train
    print("Fitting model on training set...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Bearish", "Neutral", "Bullish"], output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=["Bearish", "Neutral", "Bullish"])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n✔ Model training complete. Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report_text)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Calculate feature importances
    importances = model.feature_importances_
    feature_importance_dict = {feat: float(imp) for feat, imp in zip(features, importances)}
    
    # Save model weights
    model.save_model(MODEL_PATH)
    print(f"✔ XGBoost model parameters saved to {MODEL_PATH}")
    
    # Write accuracy metric directly to SQLite database for dashboard observability
    db = SessionLocal()
    try:
        details_dict = {
            "features_count": len(features),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "tickers_used": len(TRAIN_TICKERS),
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "feature_importances": feature_importance_dict
        }
        metric = EvaluationMetric(
            metric_type="prediction_accuracy",
            metric_value=float(accuracy),
            details=json.dumps(details_dict)
        )
        db.add(metric)
        db.commit()
        print("✔ Saved prediction accuracy metrics to database.")
    except Exception as e:
        print(f"❌ Failed to log accuracy to database: {e}")
    finally:
        db.close()
        
    return accuracy

if __name__ == "__main__":
    train_model()
