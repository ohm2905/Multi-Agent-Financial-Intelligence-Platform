import time
import datetime
import logging
import functools
import json
from langgraph.graph import StateGraph, END
from app.schemas.state import AgentState
from app.db import SessionLocal, ExecutionLog, Analysis

logger = logging.getLogger("financial_platform.agents.supervisor")

def track_agent(agent_name: str):
    """
    Decorator to measure agent execution duration and log the results to the
    SQLite database (linked to analysis_id) and graph state.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            start_time = time.time()
            status = "success"
            err_msg = None
            result = {}
            
            try:
                result = func(state, *args, **kwargs)
                return result
            except Exception as e:
                status = "failed"
                err_msg = str(e)
                logger.error(f"Agent {agent_name} failed: {e}")
                
                # Formulate safe fallback result based on agent name
                if agent_name == "Risk Agent":
                    result = {
                        "risk_metrics": {
                            "annualized_volatility": 0.0,
                            "max_drawdown": 0.0,
                            "sharpe_ratio": 0.0,
                            "beta": 1.0,
                            "risk_level": "Medium",
                            "error": err_msg
                        }
                    }
                elif agent_name == "ML Prediction Agent":
                    result = {
                        "ml_prediction": {
                            "prediction": "Neutral",
                            "probabilities": {"Bearish": 0.33, "Neutral": 0.34, "Bullish": 0.33},
                            "shap_values": [],
                            "error": err_msg
                        }
                    }
                elif agent_name == "RAG Agent":
                    result = {
                        "rag_data": {
                            "summary": f"RAG Agent failed: {err_msg}",
                            "available": False,
                            "sources": [],
                            "error": err_msg
                        }
                    }
                elif agent_name == "Sentiment Agent":
                    result = {
                        "sentiment_data": {
                            "label": "Neutral",
                            "score": 0.0,
                            "positive_pct": 0.0,
                            "neutral_pct": 100.0,
                            "negative_pct": 0.0,
                            "total_articles": 0,
                            "individual_labels": [],
                            "error": err_msg
                        }
                    }
                elif agent_name == "News Agent":
                    result = {
                        "news_data": []
                    }
                elif agent_name == "Financial Agent":
                    result = {
                        "financials_data": {},
                        "history_data": {}
                    }
                else:
                    result = {}
            finally:
                duration = time.time() - start_time
                logger.info(f"Agent {agent_name} finished in {duration:.3f}s with status: {status}")
                
                # Determine cache hit flag (None for N/A, 0 for Miss, 1 for Hit)
                is_cacheable = agent_name in ["Financial Agent", "News Agent", "Sentiment Agent"]
                cache_hit = False
                if is_cacheable:
                    if isinstance(result, dict) and "cache_hit" in result:
                        cache_hit = result.get("cache_hit")
                    elif isinstance(result, dict) and any(k in result for k in ["financials_data", "news_data", "sentiment_data"]):
                        # Check inside sub-dictionaries if present
                        for key in ["financials_data", "news_data", "sentiment_data"]:
                            if key in result and isinstance(result[key], dict) and result[key].get("cache_hit"):
                                cache_hit = True

                # Log execution details to database
                db = SessionLocal()
                try:
                    log_item = ExecutionLog(
                        analysis_id=state.get("analysis_id"),
                        agent_name=agent_name,
                        duration_seconds=duration,
                        status=status,
                        error_message=err_msg,
                        cache_hit=1 if (is_cacheable and cache_hit) else (0 if is_cacheable else None)
                    )
                    db.add(log_item)
                    db.commit()
                    
                    # Formulate state log item
                    log_details = {
                        "agent_name": agent_name,
                        "duration_seconds": round(duration, 3),
                        "status": status,
                        "timestamp": log_item.timestamp.isoformat() if log_item.timestamp else datetime.datetime.utcnow().isoformat(),
                        "cache_hit": cache_hit if is_cacheable else False
                    }
                    
                    if not isinstance(result, dict):
                        result = {}
                    if "execution_logs" not in result:
                        result["execution_logs"] = []
                    result["execution_logs"] = result["execution_logs"] + [log_details]
                except Exception as db_err:
                    logger.error(f"Failed to write agent observability log: {db_err}")
                finally:
                    db.close()
                    
            return result
        return wrapper
    return decorator

# Graph node definitions decorated with @track_agent
@track_agent("Financial Agent")
def financial_node(state: AgentState) -> dict:
    from app.agents.financial_agent import run_financial_agent
    return run_financial_agent(state)

@track_agent("News Agent")
def news_node(state: AgentState) -> dict:
    from app.agents.news_agent import run_news_agent
    return run_news_agent(state)

@track_agent("Risk Agent")
def risk_node(state: AgentState) -> dict:
    from app.agents.risk_agent import run_risk_agent
    return run_risk_agent(state)

@track_agent("Sentiment Agent")
def sentiment_node(state: AgentState) -> dict:
    from app.agents.sentiment_agent import run_sentiment_agent
    return run_sentiment_agent(state)

@track_agent("ML Prediction Agent")
def ml_prediction_node(state: AgentState) -> dict:
    from app.agents.ml_prediction_agent import run_ml_prediction_agent
    return run_ml_prediction_agent(state)

@track_agent("RAG Agent")
def rag_node(state: AgentState) -> dict:
    from app.agents.rag_agent import run_rag_agent
    return run_rag_agent(state)

@track_agent("Reporter Agent")
def reporter_node(state: AgentState) -> dict:
    from app.agents.reporter_agent import run_reporter_agent
    return run_reporter_agent(state)

def should_report(state: AgentState) -> str:
    """
    Conditional router to act as a barrier join (synchronization gate) for
    parallel paths: ml_predict, risk, and rag.
    Only proceeds to the reporter agent when all three parallel outputs are present.
    """
    if (state.get("risk_metrics") is not None and 
        state.get("ml_prediction") is not None and 
        state.get("rag_data") is not None):
        logger.info("Join barrier satisfied: all parallel paths completed. Routing to reporter.")
        return "reporter"
    
    logger.info("Join barrier not satisfied yet. Terminating current execution branch.")
    return "end"

# Build Graph Orchestration
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("financial", financial_node)
workflow.add_node("news", news_node)
workflow.add_node("sentiment", sentiment_node)
workflow.add_node("risk", risk_node)
workflow.add_node("ml_predict", ml_prediction_node)
workflow.add_node("rag", rag_node)
workflow.add_node("reporter", reporter_node)

# Set Entry Point
workflow.set_entry_point("financial")

# Add parallel routing edges
workflow.add_edge("financial", "news")
workflow.add_edge("financial", "risk")
workflow.add_edge("financial", "rag")
workflow.add_edge("news", "sentiment")
workflow.add_edge("sentiment", "ml_predict")

# Join parallel paths to the Reporter Agent using conditional barrier join
workflow.add_conditional_edges(
    "ml_predict",
    should_report,
    {
        "reporter": "reporter",
        "end": END
    }
)
workflow.add_conditional_edges(
    "risk",
    should_report,
    {
        "reporter": "reporter",
        "end": END
    }
)
workflow.add_conditional_edges(
    "rag",
    should_report,
    {
        "reporter": "reporter",
        "end": END
    }
)

# Finish after Reporter compiles the consolidated report
workflow.add_edge("reporter", END)

# Compile Graph
graph = workflow.compile()

def save_analysis_to_db(analysis_id: int, state: dict):
    """Save finalized multi-agent pipeline state back to SQLite database."""
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.company_name = state.get("company_name")
            analysis.financials = json.dumps(state.get("financials_data")) if state.get("financials_data") else None
            analysis.sentiment = json.dumps(state.get("sentiment_data")) if state.get("sentiment_data") else None
            analysis.risk_metrics = json.dumps(state.get("risk_metrics")) if state.get("risk_metrics") else None
            
            ml_pred = state.get("ml_prediction")
            if ml_pred:
                analysis.ml_prediction = ml_pred.get("prediction")
                # Confidence is the max probability
                probs = ml_pred.get("probabilities", {})
                if probs:
                    analysis.ml_confidence = max(probs.values())
                analysis.shap_explanation = json.dumps(ml_pred.get("shap_values")) if ml_pred.get("shap_values") else None
            
            # Read score, recommendation, and report from state (compiled by Reporter Agent)
            analysis.investment_score = state.get("investment_score") or 50
            analysis.recommendation = state.get("recommendation") or "Hold"
            analysis.report_markdown = state.get("report_markdown")
            
            db.commit()
            logger.info(f"Successfully finalized and saved analysis record to DB for ticker: {analysis.ticker} (ID: {analysis_id})")
    except Exception as e:
        logger.error(f"Failed to update Analysis record {analysis_id}: {e}")
    finally:
        db.close()

def run_financial_workflow(ticker: str) -> dict:
    """Entry point to run the multi-agent graph analysis for a ticker."""
    logger.info(f"Triggering LangGraph workflow for: {ticker}")
    
    # 1. Pre-create Analysis record in DB to obtain analysis_id
    db = SessionLocal()
    analysis_id = None
    try:
        from app.agents.financial_agent import normalize_ticker
        norm_ticker = normalize_ticker(ticker)
        
        analysis = Analysis(ticker=norm_ticker)
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        analysis_id = analysis.id
        logger.info(f"Pre-created Analysis record in database with ID: {analysis_id} for ticker: {norm_ticker}")
    except Exception as e:
        logger.error(f"Failed to pre-create Analysis record in database: {e}")
    finally:
        db.close()

    initial_state = {
        "ticker": ticker,
        "company_name": None,
        "analysis_id": analysis_id,
        "financials_data": None,
        "history_data": None,
        "news_data": None,
        "sentiment_data": None,
        "risk_metrics": None,
        "ml_prediction": None,
        "rag_data": None,
        "report_markdown": None,
        "execution_logs": []
    }
    
    # 2. Run the compiled StateGraph
    final_output = graph.invoke(initial_state)
    
    # 3. Post-save analysis aggregated state to DB
    if analysis_id:
        save_analysis_to_db(analysis_id, final_output)
        
    return final_output
