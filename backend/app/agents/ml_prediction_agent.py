import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from app.ML.features import generate_technical_features

logger = logging.getLogger("financial_platform.agents.ml_prediction")

def run_ml_prediction_agent(state: dict) -> dict:
    """
    ML Prediction Agent runner node for LangGraph workflow.
    Loads the trained XGBoost model, performs inference on the latest data point,
    and computes SHAP feature importance explanations.
    """
    ticker = state.get("ticker")
    history_data = state.get("history_data")
    sentiment_data = state.get("sentiment_data") or {}
    
    if not history_data or not history_data.get("close") or len(history_data["close"]) < 50:
        logger.warning(f"Insufficient history data for ML Prediction Agent ({ticker}). Minimum 50 candles required.")
        return {
            "ml_prediction": {
                "prediction": "Neutral",
                "probabilities": {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33},
                "shap_values": [],
                "error": "Insufficient history data (minimum 50 days required)."
            }
        }
        
    logger.info(f"Running ML Prediction Agent for ticker: {ticker}")
    
    # 1. Reconstruct historical DataFrame
    df = pd.DataFrame({
        "open": history_data["open"],
        "high": history_data["high"],
        "low": history_data["low"],
        "close": history_data["close"],
        "volume": history_data["volume"]
    }, index=pd.to_datetime(history_data["dates"]))
    
    # 2. Generate features
    try:
        df_feat = generate_technical_features(df)
    except Exception as e:
        logger.error(f"Failed to generate technical features: {e}")
        return {
            "ml_prediction": {
                "prediction": "Neutral",
                "probabilities": {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33},
                "shap_values": [],
                "error": f"Feature generation failed: {str(e)}"
            }
        }
        
    # Take the latest row for inference
    latest_row = df_feat.iloc[[-1]].copy()
    
    # Define features expected by the model
    features = [
        "rsi_14", "macd_line", "macd_signal", "macd_hist", 
        "dist_sma_50", "bb_width", "price_change_5d", 
        "volume_change_5d"
    ]
    
    X_latest = latest_row[features]
    
    # 4. Load XGBoost model
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(model_dir, "ML", "xgboost_stock_model.json")
    
    if not os.path.exists(model_path):
        logger.warning(f"Trained XGBoost model not found at {model_path}. Using default predictions.")
        return {
            "ml_prediction": {
                "prediction": "Neutral",
                "probabilities": {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33},
                "shap_values": [],
                "error": "Model weights file not found."
            }
        }
        
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        # 5. Inference
        prob = model.predict_proba(X_latest)[0]  # shape: (3,)
        pred_class = int(np.argmax(prob))
        
        classes_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
        prediction_label = classes_map[pred_class]
        
        probabilities_dict = {
            "Bearish": round(float(prob[0]), 3),
            "Neutral": round(float(prob[1]), 3),
            "Bullish": round(float(prob[2]), 3)
        }
        
        # 6. SHAP Explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)
        
        # Human-readable feature descriptions
        feature_descriptions = {
            "rsi_14": "RSI (14-day momentum)",
            "macd_line": "MACD Line",
            "macd_signal": "MACD Signal Line",
            "macd_hist": "MACD Histogram (divergence)",
            "dist_sma_50": "Distance from 50-day SMA",
            "bb_width": "Bollinger Bands width (volatility)",
            "price_change_5d": "5-day Price Return",
            "volume_change_5d": "5-day Volume Change"
        }
        
        shap_list = []
        for idx, feat in enumerate(features):
            # Extract shap value for the predicted class. Handle potential output formats (list or 3D array)
            if isinstance(shap_values, list):
                val = float(shap_values[pred_class][0, idx])
            else:
                val = float(shap_values[0, idx, pred_class])
                
            # Compute qualitative descriptions based on raw feature values
            raw_val = float(latest_row[feat].iloc[0])
            desc = ""
            if feat == "rsi_14":
                if raw_val < 30:
                    desc = "RSI Oversold (Reversion Potential)"
                elif raw_val > 70:
                    desc = "RSI Overbought (Pullback Risk)"
                elif raw_val < 45:
                    desc = "Weak Short-Term RSI"
                elif raw_val > 55:
                    desc = "Strong Short-Term RSI"
                else:
                    desc = "Neutral RSI Levels"
            elif feat == "macd_hist":
                desc = "MACD Bullish Divergence" if raw_val > 0 else "MACD Bearish Divergence"
            elif feat == "macd_line":
                desc = "MACD Line Bullish Zone" if raw_val > 0 else "MACD Line Bearish Zone"
            elif feat == "macd_signal":
                desc = "MACD Signal Bullish Zone" if raw_val > 0 else "MACD Signal Bearish Zone"
            elif feat == "dist_sma_50":
                desc = "Trading Above 50-day SMA" if raw_val > 0 else "Trading Below 50-day SMA"
            elif feat == "bb_width":
                desc = "High Market Volatility" if raw_val > 0.1 else "Low Market Volatility"
            elif feat == "price_change_5d":
                desc = "Positive 5d Price return" if raw_val > 0 else "Negative 5d Price return"
            elif feat == "volume_change_5d":
                desc = "Rising Trading Volume" if raw_val > 0 else "Weak Trading Volume"
            else:
                desc = feature_descriptions.get(feat, feat)
                
            # Apply context-aware defensive suffixes for counter-trending indicators that support the prediction
            if val > 0:
                if prediction_label == "Bullish":
                    if feat == "rsi_14" and raw_val < 45:
                        desc += " (historically associated with rebounds)"
                    elif feat == "price_change_5d" and raw_val <= 0:
                        desc += " (historically associated with rebounds)"
                    elif feat in ["macd_line", "macd_signal", "macd_hist", "dist_sma_50", "volume_change_5d"] and raw_val <= 0:
                        desc += " (potential mean-reversion signal)"
                elif prediction_label == "Bearish":
                    if feat == "rsi_14" and raw_val > 55:
                        desc += " (historically associated with pullbacks)"
                    elif feat == "price_change_5d" and raw_val > 0:
                        desc += " (historically associated with pullbacks)"
                    elif feat in ["macd_line", "macd_signal", "macd_hist", "dist_sma_50", "volume_change_5d"] and raw_val > 0:
                        desc += " (potential mean-reversion signal)"
                
            shap_list.append({
                "feature": feat,
                "description": desc,
                "shap_value": round(val, 4)
            })
            
        # Sort by absolute SHAP values descending
        shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        result = {
            "prediction": prediction_label,
            "probabilities": probabilities_dict,
            "shap_values": shap_list
        }
        
        return {
            "ml_prediction": result
        }
        
    except Exception as e:
        logger.error(f"Error during ML inference/SHAP: {e}")
        return {
            "ml_prediction": {
                "prediction": "Neutral",
                "probabilities": {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33},
                "shap_values": [],
                "error": f"Inference error: {str(e)}"
            }
        }
