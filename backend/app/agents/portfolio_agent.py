import logging
from collections import defaultdict
from app.agents.financial_agent import get_stock_info, get_stock_history, normalize_ticker
from app.agents.risk_agent import run_risk_agent

logger = logging.getLogger("financial_platform.agents.portfolio")

def analyze_portfolio(holdings: list) -> dict:
    """
    Accepts a list of holdings, e.g. [{"ticker": "TCS", "weight": 40.0}, {"ticker": "RELIANCE", "weight": 60.0}].
    Returns sector allocations, weighted Beta, portfolio risk level, and diversification score.
    """
    if not holdings:
        return {
            "success": True,
            "holdings": [],
            "sector_allocations": {},
            "portfolio_beta": 0.0,
            "portfolio_risk_level": "Low",
            "diversification_score": 0,
            "total_weight": 0.0
        }
        
    logger.info(f"Analyzing portfolio with {len(holdings)} holdings...")
    
    total_weight = sum(float(h.get("weight", h.get("allocation_percentage", 0.0))) for h in holdings)
    if total_weight == 0.0:
        return {
            "success": False,
            "error": "Total portfolio weight cannot be zero."
        }
        
    holdings_details = []
    sector_weights = defaultdict(float)
    weighted_beta_sum = 0.0
    
    for h in holdings:
        raw_ticker = h.get("ticker")
        weight = float(h.get("weight", h.get("allocation_percentage", 0.0)))
        
        if not raw_ticker or weight <= 0:
            continue
            
        norm_ticker = normalize_ticker(raw_ticker)
        
        try:
            # 1. Fetch info and history (leveraging caching)
            info = get_stock_info(norm_ticker)
            history = get_stock_history(norm_ticker)
            
            # 2. Get dynamic Beta from Risk Agent by passing a simulated state dictionary
            risk_state = {"ticker": norm_ticker, "history_data": history}
            risk_res = run_risk_agent(risk_state)
            risk_metrics = risk_res.get("risk_metrics", {})
            
            beta = risk_metrics.get("beta", 1.0)
            sector = info.get("sector") or "Other"
            company_name = info.get("company_name") or norm_ticker
            price = info.get("current_price") or 0.0
            
            # Group weights by sector
            sector_weights[sector] += weight
            
            # Accumulate weighted Beta
            weighted_beta_sum += (weight * beta)
            
            holdings_details.append({
                "ticker": norm_ticker,
                "company_name": company_name,
                "weight": weight,
                "sector": sector,
                "beta": round(beta, 3),
                "current_price": price
            })
        except Exception as e:
            logger.error(f"Failed to analyze portfolio holding {raw_ticker}: {e}")
            # Fallback for holding if yfinance/risk calculation fails
            sector = "Other"
            beta = 1.0
            sector_weights[sector] += weight
            weighted_beta_sum += (weight * beta)
            
            holdings_details.append({
                "ticker": norm_ticker,
                "company_name": raw_ticker,
                "weight": weight,
                "sector": sector,
                "beta": beta,
                "current_price": 0.0,
                "error": str(e)
            })
            
    # Calculate weighted Beta
    portfolio_beta = weighted_beta_sum / total_weight
    
    # Calculate risk level
    if portfolio_beta <= 0.8:
        risk_level = "Low Risk"
    elif portfolio_beta <= 1.2:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
        
    # Calculate sector concentration (Herfindahl-Hirschman Index - HHI)
    # HHI = sum of squared sector weights (as percentages of total weight)
    hhi = 0.0
    for sector, s_weight in sector_weights.items():
        pct = (s_weight / total_weight) * 100
        hhi += (pct ** 2)
        
    # Convert HHI to diversification score out of 100
    # HHI = 10000 (least diversified, single sector) -> Score = 0
    # HHI = 1500 (highly diversified) -> Score = 100
    # Score formula: map [1500, 10000] to [100, 0]
    div_score = int((10000 - hhi) / 85)
    div_score = max(0, min(100, div_score))
    
    # Round sector weights
    sector_allocations = {sec: round((wt / total_weight) * 100, 1) for sec, wt in sector_weights.items()}
    
    return {
        "success": True,
        "holdings": holdings_details,
        "sector_allocations": sector_allocations,
        "portfolio_beta": round(portfolio_beta, 3),
        "portfolio_risk_level": risk_level,
        "diversification_score": div_score,
        "total_weight": total_weight
    }
