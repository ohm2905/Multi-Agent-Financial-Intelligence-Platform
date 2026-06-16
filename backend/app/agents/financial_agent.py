import logging
import yfinance as yf
import pandas as pd
from app.cache import get_cache, set_cache

logger = logging.getLogger("financial_platform.agents.financial")

def normalize_ticker(ticker: str) -> str:
    """Normalize user input tickers for the Indian Stock Market (NSE)."""
    ticker = ticker.strip().upper()
    # Append .NS for Indian market if no exchange suffix is provided
    # and it is not an index symbol starting with '^'
    if not ticker.startswith("^") and "." not in ticker:
        ticker = f"{ticker}.NS"
    return ticker

def get_stock_info(ticker: str) -> dict:
    """
    Fetch stock metadata (P/E, Market Cap, EPS, Debt, Revenue, Profit) using yfinance.
    Uses Redis cache if available.
    """
    normalized = normalize_ticker(ticker)
    cache_key = f"info:{normalized.lower()}"
    
    # Try reading from cache
    cached_info = get_cache(cache_key)
    if cached_info:
        logger.info(f"Loaded stock info from cache for {normalized}")
        get_stock_info.cache_hit = True
        return cached_info
        
    logger.info(f"Fetching live stock info for {normalized} via yfinance...")
    get_stock_info.cache_hit = False
    try:
        stock = yf.Ticker(normalized)
        info = stock.info
        
        d_e = info.get("debtToEquity")
        if d_e is not None:
            try:
                d_e = float(d_e) / 100.0
            except (ValueError, TypeError):
                d_e = None

        # Extract required fields with fallbacks
        extracted = {
            "ticker": normalized,
            "company_name": info.get("longName") or info.get("shortName") or normalized.replace(".NS", ""),
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or 0.0,
            "market_cap": info.get("marketCap") or 0,
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or None,
            "eps": info.get("trailingEps") or None,
            "revenue": info.get("totalRevenue") or None,
            "net_profit": info.get("netIncomeToCommon") or None,
            "debt_to_equity": d_e,
            "currency": info.get("currency") or "INR",
            "summary": info.get("longBusinessSummary") or "No business summary available."
        }
        
        # Cache results for 12 hours (43200 seconds)
        set_cache(cache_key, extracted, expire_seconds=43200)
        return extracted
    except Exception as e:
        logger.error(f"Error fetching stock info for {normalized}: {e}")
        # Fallback empty profile
        return {
            "ticker": normalized,
            "company_name": normalized.replace(".NS", ""),
            "sector": "N/A",
            "industry": "N/A",
            "current_price": 0.0,
            "market_cap": 0,
            "pe_ratio": None,
            "eps": None,
            "revenue": None,
            "net_profit": None,
            "debt_to_equity": None,
            "currency": "INR",
            "summary": f"Failed to retrieve metadata. Error: {str(e)}"
        }

def get_stock_history(ticker: str, period: str = "2y") -> dict:
    """
    Fetch daily historical candles (Open, High, Low, Close, Volume) using yfinance.
    Returns a serializable dictionary. Cache expires in 6 hours (21600 seconds).
    """
    normalized = normalize_ticker(ticker)
    cache_key = f"history:{normalized.lower()}:{period}"
    
    # Try reading from cache
    cached_history = get_cache(cache_key)
    if cached_history:
        logger.info(f"Loaded stock history from cache for {normalized} ({period})")
        get_stock_history.cache_hit = True
        return cached_history
        
    logger.info(f"Fetching live stock history for {normalized} ({period}) via yfinance...")
    get_stock_history.cache_hit = False
    try:
        stock = yf.Ticker(normalized)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No price history returned for {normalized}")
            
        # Convert index to string dates for serialization
        df.index = df.index.strftime('%Y-%m-%d')
        
        # Prepare serializable format
        history_data = {
            "ticker": normalized,
            "dates": df.index.tolist(),
            "open": df["Open"].tolist(),
            "high": df["High"].tolist(),
            "low": df["Low"].tolist(),
            "close": df["Close"].tolist(),
            "volume": df["Volume"].tolist()
        }
        
        # Cache history for 6 hours
        set_cache(cache_key, history_data, expire_seconds=21600)
        return history_data
    except Exception as e:
        logger.error(f"Error fetching stock history for {normalized}: {e}")
        return {
            "ticker": normalized,
            "dates": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }

def run_financial_agent(state: dict) -> dict:
    """Financial Agent runner node for LangGraph workflow."""
    ticker = state.get("ticker")
    if not ticker:
        raise ValueError("No ticker symbol provided in agent state.")
        
    logger.info(f"Running Financial Agent for ticker: {ticker}")
    info = get_stock_info(ticker)
    history = get_stock_history(ticker)
    
    # Both sub-calls must be cache hits to count as a node cache hit
    cache_hit = getattr(get_stock_info, "cache_hit", False) and getattr(get_stock_history, "cache_hit", False)
    
    # Return updated state elements
    return {
        "company_name": info["company_name"],
        "financials_data": info,
        "history_data": history,
        "cache_hit": cache_hit
    }
