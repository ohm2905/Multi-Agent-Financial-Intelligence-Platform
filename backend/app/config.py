import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    DATABASE_URL: str = "sqlite:///./data/financial_platform.db"
    
    HF_HOME: str = "./data/cache/huggingface"
    SENTENCE_TRANSFORMERS_HOME: str = "./data/cache/sentence_transformers"
    CHROMA_DB_DIR: str = "./data/chromadb"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "sqlite" in settings.DATABASE_URL:
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if not os.path.isabs(db_path):
        db_abs_path = os.path.abspath(os.path.join(base_dir, db_path))
        settings.DATABASE_URL = f"sqlite:///{db_abs_path}"
        os.makedirs(os.path.dirname(db_abs_path), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Convert relative paths to absolute paths
for attr in ["HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "CHROMA_DB_DIR"]:
    val = getattr(settings, attr)
    if not os.path.isabs(val):
        abs_val = os.path.abspath(os.path.join(base_dir, val))
        setattr(settings, attr, abs_val)
    os.makedirs(getattr(settings, attr), exist_ok=True)

# Export HuggingFace environment variables to override default home folder
os.environ["HF_HOME"] = settings.HF_HOME
os.environ["SENTENCE_TRANSFORMERS_HOME"] = settings.SENTENCE_TRANSFORMERS_HOME
