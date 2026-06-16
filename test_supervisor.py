import os
import sys
import json

# Ensure backend directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.agents.supervisor import run_financial_workflow
from app.db import SessionLocal, ExecutionLog

def main():
    print("--- Testing LangGraph Supervisor Node ---")
    ticker = "TCS"
    
    # Run workflow (Financial -> News/Risk -> END)
    print(f"Triggering workflow for ticker: {ticker}")
    output = run_financial_workflow(ticker)
    
    print("\n✔ Workflow Execution Complete!")
    print(f"Ticker: {output.get('ticker')}")
    print(f"Company: {output.get('company_name')}")
    print(f"Financial Sector: {output.get('financials_data', {}).get('sector')}")
    print(f"Price History Length: {len(output.get('history_data', {}).get('close', []))} days")
    print(f"Risk Rating: {output.get('risk_metrics', {}).get('risk_level')}")
    print(f"News Scraped: {len(output.get('news_data', []))} articles")
    
    ml_pred = output.get("ml_prediction") or {}
    print(f"ML Prediction: {ml_pred.get('prediction')}")
    print(f"ML Probabilities: {ml_pred.get('probabilities')}")
    print("ML SHAP Explanations:")
    for shap_val in ml_pred.get("shap_values", [])[:5]:
        print(f"  • {shap_val['description']}: {shap_val['shap_value']}")
    
    print("\n--- Observability logs in State ---")
    for log in output.get("execution_logs", []):
        print(f"• {log['agent_name']}: {log['duration_seconds']}s | Status: {log['status']}")
        
    print("\n--- Observability logs in DB ---")
    db = SessionLocal()
    db_logs = db.query(ExecutionLog).order_by(ExecutionLog.timestamp.desc()).limit(5).all()
    for log in db_logs:
        print(f"• [{log.timestamp}] {log.agent_name}: {log.duration_seconds}s | {log.status}")
    db.close()

if __name__ == "__main__":
    main()
