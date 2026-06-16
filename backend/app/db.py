import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from app.config import settings

# Setup database engine and session
connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL, 
    connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    company_name = Column(String, nullable=True)
    
    # Store aggregated JSON representations
    financials = Column(Text, nullable=True)        # JSON string of P/E, EPS, Debt, etc.
    sentiment = Column(Text, nullable=True)         # JSON string of Pos, Neu, Neg scores
    ml_prediction = Column(String, nullable=True)    # "Bullish", "Bearish", "Neutral"
    ml_confidence = Column(Float, nullable=True)
    shap_explanation = Column(Text, nullable=True)   # JSON string of top feature weights
    risk_metrics = Column(Text, nullable=True)       # JSON string of Sharpe, beta, volatility, drawdown
    investment_score = Column(Integer, nullable=True)# Aggregated score out of 100
    
    recommendation = Column(String, nullable=True)   # "Buy", "Hold", "Sell", "Accumulate on Dips"
    report_markdown = Column(Text, nullable=True)    # Detailed generated markdown report
    
    # Evaluation tracking columns (Step 9)
    prediction_correct = Column(Integer, nullable=True) # 1 if correct, 0 if incorrect, None if unevaluated
    sentiment_correct = Column(Integer, nullable=True)  # 1 if correct, 0 if incorrect, None if unevaluated
    actual_trend = Column(String, nullable=True)        # "Bullish", "Bearish", "Neutral" or None
    
    # Relationships
    execution_logs = relationship("ExecutionLog", back_populates="analysis", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "company_name": self.company_name,
            "financials": json.loads(self.financials) if self.financials else {},
            "sentiment": json.loads(self.sentiment) if self.sentiment else {},
            "ml_prediction": self.ml_prediction,
            "ml_confidence": self.ml_confidence,
            "shap_explanation": json.loads(self.shap_explanation) if self.shap_explanation else {},
            "risk_metrics": json.loads(self.risk_metrics) if self.risk_metrics else {},
            "investment_score": self.investment_score,
            "recommendation": self.recommendation,
            "report_markdown": self.report_markdown,
            "prediction_correct": self.prediction_correct,
            "sentiment_correct": self.sentiment_correct,
            "actual_trend": self.actual_trend
        }

class ExecutionLog(Base):
    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    agent_name = Column(String, index=True, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    status = Column(String, nullable=False)          # "success", "failed"
    error_message = Column(Text, nullable=True)
    cache_hit = Column(Integer, default=0, nullable=True)  # 0 for Miss, 1 for Hit, None for N/A

    analysis = relationship("Analysis", back_populates="execution_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error_message": self.error_message,
            "cache_hit": bool(self.cache_hit) if self.cache_hit is not None else False
        }

class PortfolioItem(Base):
    __tablename__ = "portfolio_items"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    allocation_percentage = Column(Float, nullable=False)
    shares = Column(Float, nullable=True)
    avg_price = Column(Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "allocation_percentage": self.allocation_percentage,
            "shares": self.shares,
            "avg_price": self.avg_price
        }

class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    metric_type = Column(String, index=True, nullable=False)  # "prediction_accuracy", "sentiment_accuracy", "cache_hit_rate", "average_latency"
    metric_value = Column(Float, nullable=False)
    details = Column(Text, nullable=True)                    # JSON string with metadata/context

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type,
            "metric_value": self.metric_value,
            "details": json.loads(self.details) if self.details else {}
        }

# Helper dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize tables
def init_db():
    Base.metadata.create_all(bind=engine)
    # Check if new columns exist, if not add them (safe SQLite migration)
    from sqlalchemy import inspect, text
    try:
        inspector = inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("analyses")]
        
        with engine.begin() as conn:
            if "prediction_correct" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN prediction_correct INTEGER"))
            if "sentiment_correct" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN sentiment_correct INTEGER"))
            if "actual_trend" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN actual_trend TEXT"))
    except Exception as e:
        import logging
        logging.getLogger("financial_platform.db").warning(f"Could not verify/migrate DB columns: {e}")
