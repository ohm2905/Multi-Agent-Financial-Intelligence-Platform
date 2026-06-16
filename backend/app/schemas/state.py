from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

class AgentState(TypedDict):
    # Inputs & Metadata
    ticker: str
    company_name: Optional[str]
    analysis_id: Optional[int]
    
    # Financials and historical data
    financials_data: Optional[Dict[str, Any]]
    history_data: Optional[Dict[str, Any]]
    
    # News and Sentiment
    news_data: Optional[List[Dict[str, Any]]]
    sentiment_data: Optional[Dict[str, Any]]
    
    # Calculations
    risk_metrics: Optional[Dict[str, Any]]
    ml_prediction: Optional[Dict[str, Any]]
    
    # RAG outputs
    rag_data: Optional[Dict[str, Any]]
    
    # Final consolidated report
    report_markdown: Optional[str]
    investment_score: Optional[int]
    recommendation: Optional[str]
    
    # System observability logs (use operator.add reducer to merge parallel updates)
    execution_logs: Annotated[List[Dict[str, Any]], operator.add]
