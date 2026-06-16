import datetime
import json
import logging
import yfinance as yf
from app.db import SessionLocal, Analysis, EvaluationMetric

logger = logging.getLogger("financial_platform.utils.evaluation")

def seed_historical_analyses(db):
    """Seed historical analyses from 35-45 days ago if none exist."""
    # Count analyses older than 25 days
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=25)
    old_count = db.query(Analysis).filter(Analysis.timestamp <= cutoff).count()
    
    if old_count > 0:
        logger.info(f"Historical analyses already exist ({old_count} found). Skipping seeding.")
        return
        
    logger.info("No historical analyses found. Seeding mock matured runs for evaluation...")
    
    # We will seed 6 analyses across TCS, RELIANCE, and INFY at different historical dates:
    # 45 days ago and 35 days ago.
    # This allows yfinance to fetch real historical stock returns.
    now = datetime.datetime.utcnow()
    
    def get_past_date(days_ago):
        d = now - datetime.timedelta(days=days_ago)
        # Set to 10:00 AM UTC to be within trading hours
        return d.replace(hour=10, minute=0, second=0, microsecond=0)

    # Let's seed the following:
    seeds = [
        # TCS.NS
        {
            "ticker": "TCS.NS",
            "company_name": "Tata Consultancy Services Limited",
            "timestamp": get_past_date(45),
            "financials": '{"current_price": 3850.0, "pe_ratio": 28.5}',
            "sentiment": '{"label": "Positive", "score": 0.45}',
            "ml_prediction": "Bullish",
            "risk_metrics": '{"risk_level": "Medium"}',
            "recommendation": "Buy",
            "investment_score": 80
        },
        {
            "ticker": "TCS.NS",
            "company_name": "Tata Consultancy Services Limited",
            "timestamp": get_past_date(35),
            "financials": '{"current_price": 3890.0, "pe_ratio": 29.0}',
            "sentiment": '{"label": "Neutral", "score": 0.05}',
            "ml_prediction": "Neutral",
            "risk_metrics": '{"risk_level": "Medium"}',
            "recommendation": "Hold",
            "investment_score": 60
        },
        # RELIANCE.NS
        {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries Limited",
            "timestamp": get_past_date(45),
            "financials": '{"current_price": 2920.0, "pe_ratio": 26.2}',
            "sentiment": '{"label": "Negative", "score": -0.35}',
            "ml_prediction": "Bearish",
            "risk_metrics": '{"risk_level": "Medium"}',
            "recommendation": "Sell",
            "investment_score": 35
        },
        {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries Limited",
            "timestamp": get_past_date(35),
            "financials": '{"current_price": 2880.0, "pe_ratio": 25.8}',
            "sentiment": '{"label": "Positive", "score": 0.55}',
            "ml_prediction": "Bullish",
            "risk_metrics": '{"risk_level": "Medium"}',
            "recommendation": "Buy",
            "investment_score": 85
        },
        # INFY.NS
        {
            "ticker": "INFY.NS",
            "company_name": "Infosys Limited",
            "timestamp": get_past_date(45),
            "financials": '{"current_price": 1420.0, "pe_ratio": 22.1}',
            "sentiment": '{"label": "Neutral", "score": 0.0}',
            "ml_prediction": "Neutral",
            "risk_metrics": '{"risk_level": "Low"}',
            "recommendation": "Hold",
            "investment_score": 55
        },
        {
            "ticker": "INFY.NS",
            "company_name": "Infosys Limited",
            "timestamp": get_past_date(35),
            "financials": '{"current_price": 1450.0, "pe_ratio": 22.8}',
            "sentiment": '{"label": "Negative", "score": -0.4}',
            "ml_prediction": "Bearish",
            "risk_metrics": '{"risk_level": "Low"}',
            "recommendation": "Sell",
            "investment_score": 30
        }
    ]
    
    for s in seeds:
        item = Analysis(**s)
        db.add(item)
    db.commit()
    logger.info(f"Successfully seeded {len(seeds)} historical analyses.")

def evaluate_past_analyses():
    """
    Finds unevaluated mature analyses, downloads yfinance price history,
    evaluates prediction/sentiment correctness, updates the database,
    and recalculates/stores the aggregate metrics.
    """
    # Use a 30-day evaluation window to match the 30-day forward return prediction target
    horizon_days = 30
    db = SessionLocal()
    try:
        # 1. Seed historical runs if none exist
        seed_historical_analyses(db)
        
        # 2. Fetch unevaluated analyses older than horizon_days
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=horizon_days)
        unevaluated = db.query(Analysis).filter(
            Analysis.prediction_correct.is_(None),
            Analysis.timestamp <= cutoff
        ).all()
        
        if not unevaluated:
            logger.info("No matured unevaluated analyses found.")
            return
            
        logger.info(f"Evaluating {len(unevaluated)} matured analyses...")
        
        # Group by ticker to batch yfinance queries (much faster!)
        from collections import defaultdict
        by_ticker = defaultdict(list)
        for analysis in unevaluated:
            by_ticker[analysis.ticker].append(analysis)
            
        for ticker, analyses in by_ticker.items():
            try:
                # Find min and max timestamps to download a single history block
                timestamps = [a.timestamp for a in analyses]
                min_time = min(timestamps)
                max_time = max(timestamps)
                
                # Fetch history from min_time to max_time + horizon_days + padding
                start_date = min_time.strftime("%Y-%m-%d")
                end_date = (max_time + datetime.timedelta(days=horizon_days + 5)).strftime("%Y-%m-%d")
                
                logger.info(f"Downloading historical prices for {ticker} from {start_date} to {end_date}...")
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if hist.empty or len(hist) < 2:
                    logger.warning(f"No price history found for {ticker} in range. Skipping evaluation.")
                    continue
                    
                prices = hist["Close"]
                
                for analysis in analyses:
                    # Find closest trading price at start
                    # We look for the first trading day on or after analysis.timestamp
                    price_start = None
                    start_dt = analysis.timestamp.date()
                    
                    # Sort prices by date
                    sorted_dates = sorted(prices.index.tolist())
                    
                    # Find start price
                    for date_ts in sorted_dates:
                        d = date_ts.date()
                        if d >= start_dt:
                            price_start = prices.loc[date_ts]
                            start_trading_date = date_ts
                            break
                            
                    if price_start is None:
                        # Fallback to the first available price
                        price_start = prices.iloc[0]
                        start_trading_date = sorted_dates[0]
                        
                    # Find end price (first trading day at least horizon_days calendar days after start_trading_date)
                    price_end = None
                    target_end_date = start_trading_date.date() + datetime.timedelta(days=horizon_days)
                    
                    for date_ts in sorted_dates:
                        d = date_ts.date()
                        if d >= target_end_date:
                            price_end = prices.loc[date_ts]
                            break
                            
                    if price_start is None or price_end is None:
                        logger.warning(f"Could not resolve price bounds for analysis {analysis.id} of {ticker}")
                        continue
                        
                    # Calculate return
                    price_return = (price_end - price_start) / price_start
                    
                    # Determine actual trend:
                    # Bullish: return > +3.0% (0.03)
                    # Bearish: return < -3.0% (-0.03)
                    # Neutral: otherwise (matches the model training classes)
                    threshold = 0.03
                    if price_return > threshold:
                        actual_trend = "Bullish"
                    elif price_return < -threshold:
                        actual_trend = "Bearish"
                    else:
                        actual_trend = "Neutral"
                        
                    # 1. Compare ML Prediction
                    predicted_trend = analysis.ml_prediction # "Bullish", "Bearish", "Neutral"
                    prediction_correct = 1 if predicted_trend == actual_trend else 0
                    
                    # 2. Compare Sentiment
                    sentiment_label = "Neutral"
                    if analysis.sentiment:
                        try:
                            sent_dict = json.loads(analysis.sentiment)
                            sentiment_label = sent_dict.get("label", "Neutral")
                        except:
                            sentiment_label = analysis.sentiment
                            
                    # Map news sentiment to trend: Positive -> Bullish, Negative -> Bearish, Neutral -> Neutral
                    sent_map = {"Positive": "Bullish", "Negative": "Bearish", "Neutral": "Neutral"}
                    predicted_sent_trend = sent_map.get(sentiment_label, "Neutral")
                    sentiment_correct = 1 if predicted_sent_trend == actual_trend else 0
                    
                    # Update DB columns
                    analysis.actual_trend = actual_trend
                    analysis.prediction_correct = prediction_correct
                    analysis.sentiment_correct = sentiment_correct
                    
                    logger.info(
                        f"Evaluated analysis {analysis.id} for {ticker}: "
                        f"Start Price: {price_start:.2f}, End Price: {price_end:.2f}, Return: {price_return*100:.2f}%. "
                        f"Actual: {actual_trend} | Pred: {predicted_trend} ({'CORRECT' if prediction_correct else 'WRONG'}) | "
                        f"Sent: {sentiment_label} ({'CORRECT' if sentiment_correct else 'WRONG'})"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to evaluate analyses for ticker {ticker}: {e}")
                
        db.commit()
        
        # 3. Recalculate aggregate metrics and save to EvaluationMetric table
        # 3A. ML Prediction Accuracy
        pred_total = db.query(Analysis).filter(Analysis.prediction_correct.isnot(None)).count()
        if pred_total > 0:
            pred_hits = db.query(Analysis).filter(Analysis.prediction_correct == 1).count()
            pred_acc = pred_hits / pred_total
            
            # Save to EvaluationMetric table
            metric_pred = EvaluationMetric(
                metric_type="prediction_accuracy",
                metric_value=float(pred_acc),
                details=json.dumps({"total_evaluated": pred_total, "hits": pred_hits})
            )
            db.add(metric_pred)
            logger.info(f"Updated aggregate prediction accuracy: {pred_acc*100:.2f}% ({pred_hits}/{pred_total})")
            
        # 3B. Sentiment Accuracy
        sent_total = db.query(Analysis).filter(Analysis.sentiment_correct.isnot(None)).count()
        if sent_total > 0:
            sent_hits = db.query(Analysis).filter(Analysis.sentiment_correct == 1).count()
            sent_acc = sent_hits / sent_total
            
            metric_sent = EvaluationMetric(
                metric_type="sentiment_accuracy",
                metric_value=float(sent_acc),
                details=json.dumps({"total_evaluated": sent_total, "hits": sent_hits})
            )
            db.add(metric_sent)
            logger.info(f"Updated aggregate sentiment accuracy: {sent_acc*100:.2f}% ({sent_hits}/{sent_total})")
            
        db.commit()
        
    finally:
        db.close()
