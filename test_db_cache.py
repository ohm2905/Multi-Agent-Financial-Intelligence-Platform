import os
import sys

# Ensure backend directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.db import init_db, SessionLocal, Analysis, ExecutionLog, PortfolioItem
from app.cache import set_cache, get_cache

def main():
    print("--- Testing Database Setup ---")
    try:
        init_db()
        print("✔ Database tables initialized successfully.")
        
        # Test DB Session
        db = SessionLocal()
        
        # Insert test analysis
        test_analysis = Analysis(
            ticker="TCS.NS",
            company_name="Tata Consultancy Services",
            ml_prediction="Bullish",
            investment_score=85,
            recommendation="Buy"
        )
        db.add(test_analysis)
        db.commit()
        db.refresh(test_analysis)
        print(f"✔ Analysis record created for {test_analysis.ticker} with ID: {test_analysis.id}")
        
        # Fetch it back
        fetched = db.query(Analysis).filter_by(ticker="TCS.NS").first()
        print(f"✔ Fetched analysis from DB: {fetched.company_name} - Recommendation: {fetched.recommendation}")
        
        # Clean up
        db.delete(fetched)
        db.commit()
        db.close()
        print("✔ DB cleanup complete.")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        
    print("\n--- Testing Redis Cache Setup ---")
    try:
        test_key = "test_tcs_data"
        test_value = {"ticker": "TCS.NS", "price": 3850.50, "indicators": {"rsi": 62.4}}
        
        # Set cache
        success = set_cache(test_key, test_value, expire_seconds=10)
        if success:
            print("✔ Redis cache set succeeded.")
            # Get cache
            cached = get_cache(test_key)
            if cached == test_value:
                print("✔ Redis cache retrieve succeeded with matching data!")
            else:
                print(f"❌ Redis cache retrieve mismatch: {cached}")
        else:
            print("❌ Redis cache write failed.")
    except Exception as e:
        print(f"❌ Redis cache test failed: {e}")

if __name__ == "__main__":
    main()
