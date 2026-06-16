import logging
import requests
from app.config import settings
from app.cache import get_cache, set_cache

logger = logging.getLogger("financial_platform.agents.news")

def fetch_via_newsapi(company_name: str, ticker: str) -> list:
    """Fetch news articles using the NewsAPI everything endpoint."""
    if not settings.NEWS_API_KEY or "your_news_api_key" in settings.NEWS_API_KEY:
        logger.warning("NewsAPI key is missing or placeholder. Skipping NewsAPI.")
        return []
        
    # Query using company name or ticker (e.g. "Tata Consultancy Services" OR "TCS")
    query = f'"{company_name}" OR "{ticker}"'
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,  # Fetch top 5 relevant articles
        "apiKey": settings.NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            formatted = []
            for art in articles:
                formatted.append({
                    "title": art.get("title"),
                    "description": art.get("description") or art.get("title"),
                    "source": art.get("source", {}).get("name") or "NewsAPI",
                    "url": art.get("url"),
                    "published_at": art.get("publishedAt")
                })
            logger.info(f"Successfully fetched {len(formatted)} news articles from NewsAPI for {ticker}")
            return formatted
        else:
            logger.error(f"NewsAPI error (Status {response.status_code}): {response.text}")
    except Exception as e:
        logger.error(f"Failed to fetch from NewsAPI: {e}")
    return []

def fetch_via_tavily(company_name: str, ticker: str) -> list:
    """Fetch financial news articles using the Tavily Search API as failover/fallback."""
    if not settings.TAVILY_API_KEY or "your_tavily_api_key" in settings.TAVILY_API_KEY:
        logger.warning("Tavily API key is missing or placeholder. Skipping Tavily news.")
        return []
        
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
        query = f"latest financial news and market analysis for {company_name} {ticker}"
        # Request news category search
        results = client.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )
        
        formatted = []
        for res in results.get("results", []):
            formatted.append({
                "title": res.get("title"),
                "description": res.get("content") or res.get("title"),
                "source": "Tavily Search",
                "url": res.get("url"),
                "published_at": None  # Tavily general search doesn't guarantee published date
            })
        logger.info(f"Successfully fetched {len(formatted)} search results from Tavily for {ticker}")
        return formatted
    except Exception as e:
        logger.error(f"Failed to fetch from Tavily: {e}")
    return []

def get_stock_news(ticker: str, company_name: str) -> list:
    """
    Get recent financial news for the stock.
    Attempts NewsAPI first, falling back to Tavily.
    Caches results in Redis for 6 hours (21600 seconds).
    """
    clean_ticker = ticker.strip().lower()
    cache_key = f"news:{clean_ticker}"
    
    # Try reading cache
    cached_news = get_cache(cache_key)
    if cached_news is not None:
        logger.info(f"Loaded news from cache for {ticker}")
        get_stock_news.cache_hit = True
        return cached_news
        
    # Fetch live
    get_stock_news.cache_hit = False
    articles = fetch_via_newsapi(company_name, ticker)
    
    # If NewsAPI returned nothing or failed, try Tavily
    if not articles:
        logger.info(f"No NewsAPI results. Falling back to Tavily for {ticker}...")
        articles = fetch_via_tavily(company_name, ticker)
        
    # If both failed, return dummy news so app flow doesn't break
    if not articles:
        logger.warning(f"Both news fetchers returned empty for {ticker}. Using dummy mock news.")
        articles = [
            {
                "title": f"Market updates for {company_name} ({ticker})",
                "description": f"Standard market trading reports for {company_name}. Live news fetching requires active API keys.",
                "source": "System Monitor",
                "url": "https://finance.yahoo.com/quote/" + ticker,
                "published_at": None
            }
        ]
        
    # Cache news for 6 hours
    set_cache(cache_key, articles, expire_seconds=21600)
    return articles

def run_news_agent(state: dict) -> dict:
    """News Agent runner node for LangGraph workflow."""
    ticker = state.get("ticker")
    company_name = state.get("company_name") or ticker
    
    logger.info(f"Running News Agent for {ticker} ({company_name})")
    news_items = get_stock_news(ticker, company_name)
    cache_hit = getattr(get_stock_news, "cache_hit", False)
    
    return {
        "news_data": news_items,
        "cache_hit": cache_hit
    }
