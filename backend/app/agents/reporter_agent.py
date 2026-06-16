import logging
import json
import datetime
from app.db import SessionLocal, Analysis

logger = logging.getLogger("financial_platform.agents.reporter")

def run_reporter_agent(state: dict) -> dict:
    """
    Reporter Agent node for LangGraph workflow.
    Aggregates results from Financial, News, Sentiment, Risk, ML, and RAG agents,
    checks for recommendation drift against previous DB records,
    and compiles a beautiful markdown investment advisory report.
    """
    ticker = state.get("ticker")
    company_name = state.get("company_name") or ticker
    analysis_id = state.get("analysis_id")
    
    logger.info(f"Running Reporter Agent for ticker: {ticker} (Analysis ID: {analysis_id})")
    
    financials = state.get("financials_data") or {}
    sentiment = state.get("sentiment_data") or {}
    risk = state.get("risk_metrics") or {}
    ml = state.get("ml_prediction") or {}
    rag = state.get("rag_data") or {}
    
    # 1. Synthesize Investment Score & Recommendation
    score = 50
    
    # ML Prediction Impact
    ml_pred = ml.get("prediction", "Neutral")
    if ml_pred == "Bullish":
        score += 15
    elif ml_pred == "Bearish":
        score -= 15
        
    # Sentiment Impact
    sent_label = sentiment.get("label", "Neutral")
    if sent_label == "Positive":
        score += 15
    elif sent_label == "Negative":
        score -= 15
        
    # Risk Impact
    risk_level = risk.get("risk_level", "Medium")
    if risk_level == "Low":
        score += 20
    elif risk_level == "Medium":
        score += 10
    elif risk_level == "High":
        score -= 10
        
    # Boundary constraints
    score = max(0, min(100, score))
    
    # Recommendation Mapping
    if score >= 75:
        recommendation = "Buy"
    elif score >= 60:
        recommendation = "Accumulate on Dips"
    elif score >= 45:
        recommendation = "Hold"
    else:
        recommendation = "Sell"
        
    # 2. Compute Confidence Level & Query Database for Previous Analysis (Drift Analysis)
    probs = ml.get("probabilities") or {}
    max_prob = max(probs.values()) if probs else 0.0
    prob_val = max_prob * 100.0 if max_prob <= 1.0 else max_prob
    confidence_level = "Low" if prob_val < 45.0 else ("Medium" if prob_val <= 60.0 else "High")

    prev_rec = None
    prev_score = None
    prev_date = None
    drift_text = ""
    
    db = SessionLocal()
    try:
        # Get the most recent completed analysis for this ticker before the current one
        query_ticker = ticker
        if not query_ticker.endswith(".NS") and not query_ticker.startswith("^"):
            query_ticker = f"{query_ticker}.NS"
            
        prev_run = db.query(Analysis).filter(
            Analysis.ticker == query_ticker,
            Analysis.id != analysis_id,
            Analysis.recommendation.isnot(None)
        ).order_by(Analysis.timestamp.desc()).first()
        
        if prev_run:
            prev_rec = prev_run.recommendation
            prev_score = prev_run.investment_score
            prev_date = prev_run.timestamp.strftime("%Y-%m-%d %H:%M")
            
            # Extract previous values for comparison
            prev_sentiment = json.loads(prev_run.sentiment) if prev_run.sentiment else {}
            prev_sent_label = prev_sentiment.get("label", "Neutral") if isinstance(prev_sentiment, dict) else "Neutral"
            
            prev_risk = json.loads(prev_run.risk_metrics) if prev_run.risk_metrics else {}
            prev_risk_level = prev_risk.get("risk_level", "Medium") if isinstance(prev_risk, dict) else "Medium"
            
            prev_ml = prev_run.ml_prediction or "Neutral"
            
            # Build reasons list
            drift_reasons = []
            if prev_sent_label != sent_label:
                sentiment_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
                prev_val = sentiment_map.get(prev_sent_label, 1)
                curr_val = sentiment_map.get(sent_label, 1)
                if curr_val > prev_val:
                    drift_reasons.append(f"Sentiment improved from **{prev_sent_label}** to **{sent_label}**")
                else:
                    drift_reasons.append(f"Sentiment deteriorated from **{prev_sent_label}** to **{sent_label}**")
                    
            if prev_risk_level != risk_level:
                risk_map = {"Low": 2, "Medium": 1, "High": 0}
                prev_val = risk_map.get(prev_risk_level, 1)
                curr_val = risk_map.get(risk_level, 1)
                if curr_val > prev_val:
                    drift_reasons.append(f"Volatility/Risk level reduced from **{prev_risk_level}** to **{risk_level}**")
                else:
                    drift_reasons.append(f"Volatility/Risk level increased from **{prev_risk_level}** to **{risk_level}**")
                    
            if prev_ml != ml_pred:
                drift_reasons.append(f"ML prediction changed from **{prev_ml}** to **{ml_pred}**")
            
            reasons_str = ""
            if drift_reasons:
                reasons_str = "\n**Reasons for Drift:**\n" + "\n".join([f"• {r}" for r in drift_reasons])
            
            # Formulate drift message
            score_diff = score - prev_score
            sign = "+" if score_diff >= 0 else ""
            
            if prev_rec != recommendation:
                drift_text = (
                    f"⚠️ **Recommendation Drift Detected!**\n"
                    f"- **Previous Analysis ({prev_date})**: Rating was **{prev_rec}** (Score: {prev_score}/100)\n"
                    f"- **Current Analysis**: Rating is **{recommendation}** (Score: {score}/100)\n"
                    f"- **Shift**: Recommendation drifted from {prev_rec} to {recommendation} (Score change: {sign}{score_diff} points).{reasons_str}"
                )
            else:
                drift_text = (
                    f"✅ **Recommendation Stable.**\n"
                    f"- **Previous Analysis ({prev_date})**: Rating was **{prev_rec}** (Score: {prev_score}/100)\n"
                    f"- **Current Analysis**: Rating is **{recommendation}** (Score: {score}/100)\n"
                    f"- **Shift**: No rating drift. (Score change: {sign}{score_diff} points).{reasons_str}"
                )
        else:
            drift_text = (
                f"ℹ️ **Baseline Analysis**\n"
                f"- No previous completed analysis found for {ticker} in the database.\n"
                f"- This run establishes the baseline score and recommendation for tracking future drift."
            )
    except Exception as e:
        logger.error(f"Error querying previous analyses for drift: {e}")
        drift_text = "ℹ️ Recommendation drift comparison failed or unavailable."
    finally:
        db.close()
        
    # 3. Format Sub-Sections
    currency_symbol = "₹" if financials.get("currency") == "INR" else financials.get("currency", "$")
    current_price = f"{currency_symbol} {financials.get('current_price', 0.0):,.2f}"
    
    mcap_val = financials.get("market_cap")
    if mcap_val:
        market_cap = f"{currency_symbol} {mcap_val / 10000000:,.2f} Cr"
    else:
        market_cap = "N/A"
        
    pe_ratio = f"{financials.get('pe_ratio'):.2f}" if financials.get("pe_ratio") else "N/A"
    eps = f"{currency_symbol} {financials.get('eps'):.2f}" if financials.get("eps") else "N/A"
    debt_to_equity = f"{financials.get('debt_to_equity'):.2f}" if financials.get("debt_to_equity") else "N/A"
    
    volatility = f"{risk.get('annualized_volatility', 0.0)*100:.2f}%" if risk.get("annualized_volatility") is not None else "N/A"
    beta = f"{risk.get('beta'):.3f}" if risk.get("beta") is not None else "N/A"
    sharpe = f"{risk.get('sharpe_ratio'):.3f}" if risk.get("sharpe_ratio") is not None else "N/A"
    
    sent_score = sentiment.get("score", 0.0)
    pos_pct = sentiment.get("positive_pct", 0.0)
    neu_pct = sentiment.get("neutral_pct", 100.0)
    neg_pct = sentiment.get("negative_pct", 0.0)
    
    # SHAP Features formatting
    shap_vals = ml.get("shap_values", [])
    shap_rows = []
    if shap_vals:
        for idx, item in enumerate(shap_vals[:5]):
            impact = "Positive" if item["shap_value"] > 0 else ("Negative" if item["shap_value"] < 0 else "Neutral")
            shap_rows.append(f"{idx+1}. **{item['description']}** | SHAP Value: `{item['shap_value']:.4f}` | Impact Direction: *{impact}*")
        shap_text = "\n".join(shap_rows)
    else:
        shap_text = "*No SHAP explanation features computed or model not trained yet.*"
        
    # RAG Corporate Audit formatting
    if rag.get("available"):
        rag_text = (
            f"### Dynamic PDF Audit Insights\n"
            f"{rag.get('summary')}\n\n"
            f"*Sources used: " + ", ".join([f"[{s['source']}, Page {s['page']}]" for s in rag.get("sources", [])]) + "*"
        )
    else:
        rag_text = (
            f"❌ *No corporate report PDF uploaded or indexed for {ticker}.*\n"
            f"Please upload an annual report PDF to automatically run context-aware corporate filing audits."
        )
        
    # Executive Summary Paragraph Formulation
    exec_summary = (
        f"Based on the multi-agent analysis, **{company_name}** ({ticker}) is rated as a **{recommendation}** "
        f"with a consolidated investment score of **{score}/100** and a **{confidence_level}** ML prediction confidence. "
        f"This score is calculated using a rule-based weighted aggregation framework combining the XGBoost technical trend prediction (30%), real-time FinBERT media sentiment (30%), and Nifty-relative volatility/risk metrics (40%). "
    )
    if recommendation == "Buy":
        exec_summary += "The stock displays strong positive indicators across market sentiment and technical parameters with acceptable risk tolerances."
    elif recommendation == "Accumulate on Dips":
        exec_summary += "The stock presents solid fundamentals, but momentum or market indicators suggest entry during consolidation is preferred."
    elif recommendation == "Hold":
        exec_summary += "The stock is fairly valued with balanced indicators. Existing investors should hold their positions while monitoring risk parameters."
    else:
        exec_summary += "Multiple indicators (financial, sentiment, or ML predictions) suggest bearish signals. Exposure reduction is advised."

    # 4. Assemble full markdown report
    markdown_report = f"""# Investment Advisory Report: {company_name} ({ticker})
*Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

---

## 1. Executive Summary
- **Recommendation Rating**: **{recommendation}**
- **Consolidated Score**: **{score}/100** (Weighted Aggregation Framework)
- **ML Confidence Level**: **{confidence_level}**
- **Current Price**: **{current_price}**

{exec_summary}

---

## 2. Recommendation Drift Analysis
{drift_text}

---

## 3. Financial Profile
- **Market Capitalization**: {market_cap}
- **Trailing P/E Ratio**: {pe_ratio}
- **Earnings Per Share (EPS)**: {eps}
- **Debt to Equity Ratio**: {debt_to_equity}

The stock currently trades at {current_price} with a P/E ratio of {pe_ratio}. Debt-to-equity levels stand at {debt_to_equity}.

---

## 4. Risk Assessment
- **Annualized Volatility**: {volatility}
- **Beta (vs Nifty 50 Index)**: {beta}
- **Sharpe Ratio**: {sharpe}
- **Risk Level**: **{risk_level}**

With an annualized volatility of {volatility} and a Sharpe ratio of {sharpe}, the asset displays a **{risk_level}** risk profile. The Beta against Nifty 50 is {beta}.

---

## 5. Market Sentiment
- **Overall Sentiment**: **{sent_label}** (Score: `{sent_score:+.3f}`)
- **Sentiment Breakdown**:
  - Positive: `{pos_pct}%`
  - Neutral: `{neu_pct}%`
  - Negative: `{neg_pct}%`

News headlines scraped from financial networks reflect an overall **{sent_label.lower()}** sentiment.

---

## 6. Machine Learning Price Forecast
- **Predicted 30-Day Trend**: **{ml_pred}**
- **Trend Probability Breakdown**:
  - Bullish: `{ml.get('probabilities', {}).get('Bullish', 0.0)*100:.1f}%`
  - Neutral: `{ml.get('probabilities', {}).get('Neutral', 0.0)*100:.1f}%`
  - Bearish: `{ml.get('probabilities', {}).get('Bearish', 0.0)*100:.1f}%`

### Top Feature Drivers (SHAP Explanations)
{shap_text}

---

## 7. Corporate Report RAG Audit
{rag_text}

---

*Disclaimer: This report is automatically generated by the Multi-Agent Financial Intelligence Platform using local LLMs and ML forecasts. It is intended for educational purposes only and does not constitute formal investment advice or brokerage recommendations.*
"""

    # Return updated state elements including consolidated score, recommendation, and report
    return {
        "investment_score": score,
        "recommendation": recommendation,
        "report_markdown": markdown_report
    }
