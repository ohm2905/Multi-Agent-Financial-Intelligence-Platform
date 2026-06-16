import os
import sys
import json

# Ensure backend directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.agents.supervisor import run_financial_workflow
from app.agents.portfolio_agent import analyze_portfolio
from app.db import SessionLocal, Analysis, PortfolioItem

def test_supervisor_and_reporter():
    print("--- 1. Testing LangGraph supervisor with Reporter Agent node ---")
    ticker = "TCS"
    
    # Run the workflow which now includes Financial -> News/Risk/RAG -> Reporter -> END
    print(f"Triggering workflow for ticker: {ticker}...")
    output = run_financial_workflow(ticker)
    
    print("\n✔ Workflow Execution Complete!")
    print(f"Ticker: {output.get('ticker')}")
    print(f"Consolidated Score: {output.get('investment_score')}/100")
    print(f"Recommendation: {output.get('recommendation')}")
    
    # Verify the markdown report is generated
    report = output.get("report_markdown")
    if report:
        print(f"✔ Generated Advisory Report Length: {len(report)} characters")
        print("\nReport Excerpt:")
        print("\n".join(report.split("\n")[:15])) # print first 15 lines
    else:
        print("❌ Failed: No report markdown returned in output state.")
        
    # Verify saved state in DB
    db = SessionLocal()
    try:
        saved = db.query(Analysis).filter(Analysis.ticker == "TCS.NS").order_by(Analysis.timestamp.desc()).first()
        if saved and saved.report_markdown:
            print(f"\n✔ Saved Analysis in DB ID: {saved.id}")
            print(f"✔ DB Score: {saved.investment_score} | DB Recommendation: {saved.recommendation}")
            print(f"✔ DB Report is not null (Length: {len(saved.report_markdown)})")
        else:
            print("❌ Failed: Analysis record not successfully finalized/saved in DB.")
    finally:
        db.close()

def test_portfolio_agent():
    print("\n--- 2. Testing Portfolio Analysis Agent ---")
    
    # Let us mock some holdings
    holdings = [
        {"ticker": "TCS.NS", "weight": 40.0},
        {"ticker": "RELIANCE.NS", "weight": 60.0}
    ]
    
    print(f"Mock portfolio holdings: {holdings}")
    res = analyze_portfolio(holdings)
    
    if res.get("success"):
        print("✔ Portfolio analysis succeeded!")
        print(f"Portfolio Beta: {res.get('portfolio_beta')}")
        print(f"Risk Rating: {res.get('portfolio_risk_level')}")
        print(f"Diversification Score: {res.get('diversification_score')}/100")
        print("Sector Allocations:")
        print(json.dumps(res.get("sector_allocations"), indent=2))
        print("Holdings Details:")
        for h in res.get("holdings"):
            print(f" - {h['ticker']} ({h['company_name']}): Weight {h['weight']}% | Sector: {h['sector']} | Beta: {h['beta']}")
    else:
        print(f"❌ Portfolio analysis failed: {res.get('error')}")

if __name__ == "__main__":
    test_supervisor_and_reporter()
    test_portfolio_agent()
