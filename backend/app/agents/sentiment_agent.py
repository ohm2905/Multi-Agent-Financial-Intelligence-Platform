import logging
import time
from app.config import settings
from app.cache import get_cache, set_cache

logger = logging.getLogger("financial_platform.agents.sentiment")

# Lazy load local transformers model to save memory on server startup
_finbert_pipeline = None

def get_finbert_pipeline():
    """Lazily initialize the local FinBERT sentiment pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
        
    logger.info("Initializing FinBERT model (yiyanghkust/finbert-tone) locally...")
    start_time = time.time()
    try:
        from transformers import BertTokenizer, BertForSequenceClassification, pipeline
        
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        _finbert_pipeline = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer,
            max_length=512,
            truncation=True
        )
        logger.info(f"FinBERT initialized successfully in {time.time() - start_time:.2f}s")
        return _finbert_pipeline
    except Exception as e:
        logger.error(f"Failed to load FinBERT model locally: {e}. Fallback logic will be used.")
        return None

def analyze_sentiment_llm_fallback(headlines: list) -> list:
    """Analyze sentiment of news headlines using Gemini 2.5 Flash as an extremely robust fallback."""
    if not settings.GEMINI_API_KEY or "your_gemini_api_key" in settings.GEMINI_API_KEY:
        logger.warning("Gemini API key not found. Cannot run LLM sentiment fallback.")
        return []
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Formulate prompt
        prompt = (
            "Analyze the sentiment of the following financial news headlines. "
            "For each headline, respond with exactly one of: 'Positive', 'Negative', or 'Neutral'. "
            "Return the labels in order, one per line. Do not write any other explanation or text.\n\n"
            "Headlines:\n" + "\n".join([f"- {h}" for h in headlines])
        )
        
        response = model.generate_content(prompt)
        labels = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
        
        # Map back to standard labels
        mapped_results = []
        for label in labels:
            clean_label = label.replace("-", "").strip().upper()
            if "POSITIVE" in clean_label:
                mapped_results.append("Positive")
            elif "NEGATIVE" in clean_label:
                mapped_results.append("Negative")
            else:
                mapped_results.append("Neutral")
        logger.info(f"Successfully ran LLM fallback sentiment analysis for {len(mapped_results)} headlines")
        return mapped_results
    except Exception as e:
        logger.error(f"LLM sentiment fallback failed: {e}")
        return []

def get_articles_sentiment(articles: list, ticker: str) -> dict:
    """
    Classify news articles sentiment and aggregate the results.
    Caches the final aggregated results in Redis for 6 hours (21600 seconds).
    """
    clean_ticker = ticker.strip().lower()
    cache_key = f"sentiment:{clean_ticker}"
    
    # Try reading cache
    cached_sentiment = get_cache(cache_key)
    if cached_sentiment is not None:
        logger.info(f"Loaded news sentiment from cache for {ticker}")
        get_articles_sentiment.cache_hit = True
        return cached_sentiment
    get_articles_sentiment.cache_hit = False
        
    if not articles:
        return {
            "label": "Neutral",
            "score": 0.0,
            "positive_pct": 0.0,
            "neutral_pct": 100.0,
            "negative_pct": 0.0,
            "total_articles": 0,
            "individual_labels": []
        }
        
    headlines = [art.get("title") for art in articles if art.get("title")]
    results = []
    
    # Attempt 1: Try local FinBERT
    nlp = get_finbert_pipeline()
    if nlp is not None:
        try:
            logger.info(f"Classifying {len(headlines)} headlines using local FinBERT...")
            classifications = nlp(headlines)
            # FinBERT labels: 'Positive', 'Negative', 'Neutral' (varies by model outputs)
            # The model 'yiyanghkust/finbert-tone' outputs labels like: 'Positive', 'Negative', 'Neutral'
            for c in classifications:
                label = c.get("label", "Neutral").capitalize()
                # Ensure it aligns to standard 'Positive', 'Negative', 'Neutral'
                if "Pos" in label:
                    results.append("Positive")
                elif "Neg" in label:
                    results.append("Negative")
                else:
                    results.append("Neutral")
        except Exception as e:
            logger.error(f"Local FinBERT inference failed: {e}. Falling back to LLM.")
            nlp = None  # Force fallback
            
    # Attempt 2: Try Gemini 2.5 Flash Fallback if local failed
    if nlp is None:
        results = analyze_sentiment_llm_fallback(headlines)
        
    # Attempt 3: If both failed, use a very basic heuristic rule classifier
    if not results:
        logger.warning("Both local FinBERT and LLM fallbacks failed. Using rule-based fallback.")
        positive_keywords = ["up", "rise", "gain", "profit", "growth", "high", "buy", "bullish", "strong", "positive"]
        negative_keywords = ["down", "fall", "loss", "decline", "low", "sell", "bearish", "weak", "negative", "drop"]
        
        for h in headlines:
            hl = h.lower()
            pos_count = sum(1 for kw in positive_keywords if kw in hl)
            neg_count = sum(1 for kw in negative_keywords if kw in hl)
            if pos_count > neg_count:
                results.append("Positive")
            elif neg_count > pos_count:
                results.append("Negative")
            else:
                results.append("Neutral")
                
    # Align results length to headlines
    while len(results) < len(headlines):
        results.append("Neutral")
        
    # Aggregate counts
    pos_count = results.count("Positive")
    neu_count = results.count("Neutral")
    neg_count = results.count("Negative")
    total = len(results)
    
    # Calculate percentages
    pos_pct = round((pos_count / total) * 100, 1) if total > 0 else 0.0
    neu_pct = round((neu_count / total) * 100, 1) if total > 0 else 100.0
    neg_pct = round((neg_count / total) * 100, 1) if total > 0 else 0.0
    
    # Calculate score on -1 to 1 scale
    # score = (pos - neg) / total
    score = round((pos_count - neg_count) / total, 3) if total > 0 else 0.0
    
    # Classify overall label
    overall_label = "Neutral"
    if score > 0.15:
        overall_label = "Positive"
    elif score < -0.15:
        overall_label = "Negative"
        
    aggregated = {
        "label": overall_label,
        "score": score,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "total_articles": total,
        "individual_labels": results
    }
    
    # Cache results for 6 hours
    set_cache(cache_key, aggregated, expire_seconds=21600)
    return aggregated

def run_sentiment_agent(state: dict) -> dict:
    """Sentiment Agent runner node for LangGraph workflow."""
    ticker = state.get("ticker")
    news_data = state.get("news_data") or []
    
    logger.info(f"Running Sentiment Agent for ticker: {ticker}")
    sentiment = get_articles_sentiment(news_data, ticker)
    cache_hit = getattr(get_articles_sentiment, "cache_hit", False)
    
    return {
        "sentiment_data": sentiment,
        "cache_hit": cache_hit
    }
