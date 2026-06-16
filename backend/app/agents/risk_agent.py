import logging
import math
import numpy as np
import pandas as pd
from app.agents.financial_agent import get_stock_history
from app.cache import get_cache, set_cache

logger = logging.getLogger("financial_platform.agents.risk")

def calculate_volatility(close_prices: list) -> float:
    """Calculate annualized volatility from historical close prices."""
    if len(close_prices) < 2:
        return 0.0
    series = pd.Series(close_prices)
    daily_returns = series.pct_change(fill_method=None).dropna()
    daily_vol = daily_returns.std()
    # Annualize by multiplying by square root of 252 trading days
    annualized_vol = daily_vol * math.sqrt(252)
    return float(annualized_vol)

def calculate_max_drawdown(close_prices: list) -> float:
    """Calculate the maximum peak-to-trough decline as a percentage."""
    if len(close_prices) < 2:
        return 0.0
    series = pd.Series(close_prices)
    roll_max = series.cummax()
    drawdowns = (series - roll_max) / roll_max
    max_dd = drawdowns.min()
    return float(max_dd)

def calculate_sharpe_ratio(close_prices: list, risk_free_rate: float = 0.06) -> float:
    """Calculate the annualized Sharpe Ratio (default Indian risk-free rate of 6%)."""
    if len(close_prices) < 5:
        return 0.0
    series = pd.Series(close_prices)
    daily_returns = series.pct_change(fill_method=None).dropna()
    
    # De-annualize risk-free rate
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    
    mean_excess = excess_returns.mean()
    std_returns = daily_returns.std()
    
    if std_returns == 0:
        return 0.0
        
    daily_sharpe = mean_excess / std_returns
    annualized_sharpe = daily_sharpe * math.sqrt(252)
    return float(annualized_sharpe)

def calculate_beta(stock_history: dict, period: str = "2y") -> float:
    """
    Calculate the stock's Beta compared to the Nifty 50 Index (^NSEI).
    Fetches and caches Nifty 50 historical candles for alignment.
    """
    stock_dates = stock_history.get("dates")
    stock_closes = stock_history.get("close")
    
    if not stock_dates or len(stock_closes) < 10:
        return 1.0  # Default beta
        
    # Get Nifty 50 Index history (cached for 12 hours)
    nifty_history = get_stock_history("^NSEI", period=period)
    nifty_dates = nifty_history.get("dates")
    nifty_closes = nifty_history.get("close")
    
    if not nifty_dates or len(nifty_closes) < 10:
        logger.warning("Could not fetch Nifty 50 benchmark data. Defaulting beta to 1.0.")
        return 1.0
        
    # Align dates using Pandas DataFrames
    stock_df = pd.DataFrame({"close_stock": stock_closes}, index=stock_dates)
    nifty_df = pd.DataFrame({"close_nifty": nifty_closes}, index=nifty_dates)
    
    # Merge on date index
    merged = stock_df.join(nifty_df, how="inner")
    
    if len(merged) < 10:
        logger.warning("Insufficient overlapping trading days to calculate Beta. Defaulting to 1.0.")
        return 1.0
        
    # Compute daily percentage returns
    merged["returns_stock"] = merged["close_stock"].pct_change(fill_method=None)
    merged["returns_nifty"] = merged["close_nifty"].pct_change(fill_method=None)
    merged = merged.dropna()
    
    if len(merged) < 5:
        return 1.0
        
    # Calculate Beta = Covariance(Stock, Index) / Variance(Index)
    covariance_matrix = np.cov(merged["returns_stock"], merged["returns_nifty"])
    covariance = covariance_matrix[0, 1]
    variance_nifty = covariance_matrix[1, 1]
    
    if variance_nifty == 0:
        return 1.0
        
    beta = covariance / variance_nifty
    return float(beta)

def run_risk_agent(state: dict) -> dict:
    """Risk Agent runner node for LangGraph workflow."""
    ticker = state.get("ticker")
    try:
        history_data = state.get("history_data")
        if not history_data or not history_data.get("close"):
            raise ValueError(f"No price history data available in state for Risk Agent ({ticker}).")
            
        logger.info(f"Running Risk Agent for ticker: {ticker}")
        close_prices = history_data["close"]
        
        volatility = calculate_volatility(close_prices)
        max_dd = calculate_max_drawdown(close_prices)
        sharpe = calculate_sharpe_ratio(close_prices)
        beta = calculate_beta(history_data)
        
        # Determine risk level category based on volatility and beta
        # Volatility bounds: <15% Low, 15%-30% Medium, >30% High
        # Beta bounds: <0.8 Low, 0.8-1.2 Medium, >1.2 High
        risk_level = "Medium"
        vol_pct = volatility * 100
        
        if vol_pct < 15.0 and beta < 0.85:
            risk_level = "Low"
        elif vol_pct > 32.0 or beta > 1.3:
            risk_level = "High"
            
        risk_metrics = {
            "annualized_volatility": volatility,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "beta": beta,
            "risk_level": risk_level
        }
        return {
            "risk_metrics": risk_metrics
        }
    except Exception as e:
        logger.error(f"Error in Risk Agent execution: {e}")
        return {
            "risk_metrics": {
                "annualized_volatility": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "beta": 1.0,
                "risk_level": "Medium",
                "error": str(e)
            }
        }
