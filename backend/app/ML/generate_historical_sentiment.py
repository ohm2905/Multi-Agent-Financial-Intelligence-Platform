import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import random
import time

# Ensure backend directory is in path (three levels up from backend/app/ML/generate_historical_sentiment.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define output path
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_sentiment.csv")

# Set up templates
POSITIVE_TEMPLATES = [
    "{company} shares surge as market registers positive momentum.",
    "{company} reports strong revenue growth and expanded operating margins.",
    "{company} signs strategic technology partnership to drive digital transformation.",
    "{company} shares gain as brokerages raise target price following earnings.",
    "{company} order book reaches record high on strong domestic demand."
]

NEGATIVE_TEMPLATES = [
    "{company} shares drop on macro headwinds and delayed client spending.",
    "{company} reports net profit contraction due to rising operating costs.",
    "{company} stock declines amid margin compression in recent quarter.",
    "{company} faces supply chain bottlenecks and labor cost pressures.",
    "{company} shares slip as global market sell-off impacts financial sector."
]

NEUTRAL_TEMPLATES = [
    "{company} launches new enterprise solutions and expands service delivery.",
    "{company} steady on market trends as analyst sentiment remains balanced.",
    "{company} announces transition in leadership team, board remains confident.",
    "{company} continues branch expansion and business digitization projects.",
    "{company} trades in narrow range ahead of upcoming corporate filings."
]

# Clean company name mapping
COMPANY_CLEAN_NAMES = {
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals", "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJAJFINSV.NS": "Bajaj Finserv", "BAJFINANCE.NS": "Bajaj Finance",
    "BHARTIARTL.NS": "Bharti Airtel", "BPCL.NS": "BPCL",
    "BRITANNIA.NS": "Britannia Industries", "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India", "DIVISLAB.NS": "Divi's Labs",
    "DRREDDY.NS": "Dr. Reddy's", "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries", "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank", "HDFCLIFE.NS": "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp", "HINDALCO.NS": "Hindalco",
    "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank", "INFY.NS": "Infosys",
    "ITC.NS": "ITC Limited", "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "LT.NS": "Larsen & Toubro",
    "LTIM.NS": "LTIMindtree", "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki", "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC", "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid", "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life", "SBIN.NS": "State Bank of India",
    "SUNPHARMA.NS": "Sun Pharma", "TATACONSUM.NS": "Tata Consumer",
    "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "Tata Consultancy Services", "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company", "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro", "SHRIRAMFIN.NS": "Shriram Finance"
}

def clean_company_name(ticker: str) -> str:
    return COMPANY_CLEAN_NAMES.get(ticker, ticker.replace(".NS", ""))

def main():
    print("--- Starting Historical News Sentiment Generator (Option B) ---")
    
    # Load tickers from train.py
    from app.ML.train import TRAIN_TICKERS
    
    # 1. Initialize FinBERT pipeline
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for FinBERT inference")
    
    from transformers import BertTokenizer, BertForSequenceClassification, pipeline
    print("Loading yiyanghkust/finbert-tone model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
        device=0 if device in ["cuda", "mps"] else -1
    )
    
    rows = []
    
    for ticker in TRAIN_TICKERS:
        print(f"Processing {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="5y")
            if df.empty:
                print(f"  Empty history for {ticker}")
                continue
                
            # Rename columns to lowercase standard
            df.columns = [col.lower() for col in df.columns]
            
            # Resample to weekly frequency (Friday close)
            weekly_df = df.resample("W").last()
            
            # Drop empty weeks
            weekly_df = weekly_df.dropna(subset=["close"])
            
            # Calculate weekly returns
            weekly_df["weekly_return"] = weekly_df["close"].pct_change()
            weekly_df = weekly_df.dropna(subset=["weekly_return"])
            
            company_name = clean_company_name(ticker)
            
            # Generate headlines based on weekly returns
            headlines = []
            dates = []
            for date, row in weekly_df.iterrows():
                ret = row["weekly_return"]
                if ret > 0.015:
                    template = random.choice(POSITIVE_TEMPLATES)
                elif ret < -0.015:
                    template = random.choice(NEGATIVE_TEMPLATES)
                else:
                    template = random.choice(NEUTRAL_TEMPLATES)
                
                headline = template.format(company=company_name)
                headlines.append(headline)
                dates.append(date.strftime("%Y-%m-%d"))
                
            # Run FinBERT classification in batches
            print(f"  Classifying {len(headlines)} weekly headlines for {ticker}...")
            batch_size = 64
            sentiment_scores = []
            
            for i in range(0, len(headlines), batch_size):
                batch_headlines = headlines[i:i+batch_size]
                outputs = nlp(batch_headlines)
                for out in outputs:
                    label = out.get("label", "Neutral").capitalize()
                    score = out.get("score", 0.0)
                    
                    # Compute a score on -1 to 1 scale
                    if "Pos" in label:
                        s_score = float(score)
                    elif "Neg" in label:
                        s_score = -float(score)
                    else:
                        s_score = 0.0
                    sentiment_scores.append(round(s_score, 3))
                    
            # Add to rows
            for date, s_score in zip(dates, sentiment_scores):
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "sentiment_score": s_score
                })
                
        except Exception as e:
            print(f"❌ Failed to process {ticker}: {e}")
            
    # Save to CSV
    sentiment_df = pd.DataFrame(rows)
    sentiment_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✔ Successfully saved {len(sentiment_df)} weekly historical sentiment records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
