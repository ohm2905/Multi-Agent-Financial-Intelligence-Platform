import os
import sys
import json

# Ensure backend directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.agents.financial_agent import get_stock_info, get_stock_history, normalize_ticker
from app.agents.risk_agent import run_risk_agent

def main():
    print("--- Testing Financial Agent ---")
    ticker_input = "TCS"
    normalized = normalize_ticker(ticker_input)
    print(f"Normalized Ticker: '{ticker_input}' -> '{normalized}'")
    
    info = get_stock_info(normalized)
    print(f"Company Name: {info.get('company_name')}")
    print(f"Sector: {info.get('sector')}")
    print(f"Current Price: {info.get('currency')} {info.get('current_price')}")
    print(f"PE Ratio: {info.get('pe_ratio')}")
    print(f"Debt-to-Equity: {info.get('debt_to_equity')}")
    
    history = get_stock_history(normalized, period="1y")
    print(f"Fetched price history with {len(history.get('close', []))} trading days.")
    
    print("\n--- Testing Risk Agent ---")
    state = {
        "ticker": normalized,
        "history_data": history
    }
    
    risk_output = run_risk_agent(state)
    metrics = risk_output["risk_metrics"]
    print(f"Annualized Volatility: {metrics['annualized_volatility'] * 100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Beta vs Nifty 50: {metrics['beta']:.3f}")
    print(f"Assessed Risk Level: {metrics['risk_level']}")

if __name__ == "__main__":
    main()
