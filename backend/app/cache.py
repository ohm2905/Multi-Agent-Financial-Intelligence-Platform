import json
import logging
from functools import wraps
import redis
from app.config import settings

logger = logging.getLogger("financial_platform.cache")

# Initialize Redis client connection
redis_client = None

try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        socket_connect_timeout=2.0,
        decode_responses=True  # Automatically decodes strings
    )
    # Ping to check if server is actually running
    redis_client.ping()
    logger.info(f"Redis client initialized and connected at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
except Exception as e:
    logger.warning(
        f"Redis connection failed: {e}. Caching will be disabled and live fallbacks used."
    )
    redis_client = None

def get_cache(key: str):
    """Retrieve data from Redis cache. Returns parsed dict/list or None."""
    if not redis_client:
        return None
    try:
        data = redis_client.get(key)
        if data:
            logger.info(f"Cache HIT for key: {key}")
            return json.loads(data)
        logger.info(f"Cache MISS for key: {key}")
    except Exception as e:
        logger.error(f"Error reading from Redis cache: {e}")
    return None

def set_cache(key: str, value, expire_seconds: int = 86400):
    """Set key-value pair in Redis cache with expiration."""
    if not redis_client:
        return False
    try:
        serialized = json.dumps(value)
        redis_client.set(key, serialized, ex=expire_seconds)
        logger.info(f"Cache SET for key: {key} (Expires in {expire_seconds}s)")
        return True
    except Exception as e:
        logger.error(f"Error writing to Redis cache: {e}")
    return False

def cache_data(key_prefix: str, expire_seconds: int = 86400):
    """
    Decorator to cache function return values.
    Expects functions where the first argument (or keyword arg 'ticker') is the stock ticker.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve the ticker to formulate the cache key
            ticker = None
            if args:
                ticker = args[0]
            elif "ticker" in kwargs:
                ticker = kwargs["ticker"]
                
            if not ticker or not isinstance(ticker, str):
                # Fallback to standard execution if no ticker found
                return func(*args, **kwargs)
                
            # Create a clean cache key (lowercase, stripped)
            clean_ticker = ticker.strip().lower()
            cache_key = f"{key_prefix}:{clean_ticker}"
            
            # Check cache
            cached_result = get_cache(cache_key)
            if cached_result is not None:
                # Add a hit tag for evaluation tracking later
                wrapper.cache_hit = True
                return cached_result
                
            # Fetch fresh data
            wrapper.cache_hit = False
            result = func(*args, **kwargs)
            
            # Save to cache
            if result:
                set_cache(cache_key, result, expire_seconds)
                
            return result
        
        # Track hits on the wrapper
        wrapper.cache_hit = False
        return wrapper
    return decorator
