import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import yfinance as yf

# Ensure backend directory is in path (three levels up from backend/app/ML/backtest.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ML.train import prepare_train_test_datasets, MODEL_PATH

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")

def find_optimal_threshold(train_df, model, features):
    """
    Perform a grid search on the training split to select the probability threshold
    that maximizes the Sharpe Ratio.
    """
    print("--- Running Dynamic Threshold Optimization ---")
    train_df = train_df.copy()
    train_df["date"] = pd.to_datetime(train_df["date"])
    
    # Predict probabilities on training set
    X_train = train_df[features]
    probs_train = model.predict_proba(X_train)
    train_df["prob_bullish"] = probs_train[:, 2]
    
    # Calculate next-day daily return for each stock (buying at close t, selling at close t+1)
    train_df["daily_return"] = train_df.groupby("ticker")["close"].pct_change().shift(-1)
    train_df = train_df.dropna(subset=["daily_return"])
    
    grouped = train_df.groupby("date")
    train_dates = sorted(grouped.groups.keys())
    
    # Define optimization grid
    thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60]
    best_threshold = 0.42
    best_sharpe = -999.0
    optimization_logs = []
    
    for th in thresholds:
        strategy_cum = 100.0
        strat_daily_returns = []
        
        for date in train_dates:
            group = grouped.get_group(date)
            # Strategy: allocate equally among bullish stocks
            bullish_stocks = group[group["prob_bullish"] > th]
            if not bullish_stocks.empty:
                strat_ret = bullish_stocks["daily_return"].mean()
            else:
                strat_ret = 0.0  # stay in cash
            
            strat_daily_returns.append(strat_ret)
            strategy_cum *= (1 + strat_ret)
            
        # Compute Sharpe Ratio
        strat_daily_arr = np.array(strat_daily_returns)
        n_days = len(strat_daily_returns)
        years = n_days / 252.0
        cum_ret_pct = strategy_cum - 100.0
        ann_ret = ((strategy_cum / 100.0) ** (1.0 / (years if years > 0 else 1.0)) - 1.0) * 100.0
        vol = np.std(strat_daily_arr) * np.sqrt(252.0) * 100.0
        sharpe = (ann_ret / vol) if vol > 0.0 else 0.0
        
        print(f"  Threshold: {th:.2f} | Cumulative Return: {cum_ret_pct:+.2f}% | Volatility: {vol:.2f}% | Sharpe: {sharpe:.3f}")
        optimization_logs.append({
            "threshold": th,
            "cumulative_return": round(cum_ret_pct, 2),
            "volatility": round(vol, 2),
            "sharpe_ratio": round(sharpe, 3)
        })
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = th
            
    print(f"✔ Optimized Probability Threshold: {best_threshold:.2f} (Sharpe: {best_sharpe:.3f})")
    return best_threshold, optimization_logs

def run_backtest():
    print("--- Running Strategy Backtest (2025-2026) ---")
    
    # 1. Load prepared datasets
    train_df, test_df = prepare_train_test_datasets()
    
    # Sort test_df by date
    test_df["date"] = pd.to_datetime(test_df["date"])
    test_df = test_df.sort_values("date")
    
    # 2. Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights file not found at {MODEL_PATH}. Run training first.")
        
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Define features
    features = [
        "rsi_14", "macd_line", "macd_signal", "macd_hist", 
        "dist_sma_50", "bb_width", "price_change_5d", 
        "volume_change_5d"
    ]
    
    # Run Grid Search Threshold Optimization
    optimal_th, optimization_logs = find_optimal_threshold(train_df, model, features)
    
    # 3. Predict probabilities on test split
    print("Predicting probabilities on test set...")
    X_test = test_df[features]
    probs = model.predict_proba(X_test)
    
    # Add probabilities to test_df
    test_df["prob_bearish"] = probs[:, 0]
    test_df["prob_neutral"] = probs[:, 1]
    test_df["prob_bullish"] = probs[:, 2]
    
    # Calculate next-day daily return for each stock (buying at close t, selling at close t+1)
    test_df["daily_return"] = test_df.groupby("ticker")["close"].pct_change().shift(-1)
    test_df = test_df.dropna(subset=["daily_return"])
    
    # 4. Fetch Nifty 50 Index benchmark (^NSEI) from yfinance for alignment
    print("Downloading ^NSEI benchmark index returns for out-of-sample split...")
    bench_returns_dict = {}
    try:
        start_date = test_df["date"].min().strftime("%Y-%m-%d")
        end_date = (test_df["date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        nifty_index = yf.Ticker("^NSEI")
        nifty_df = nifty_index.history(start=start_date, end=end_date)
        if not nifty_df.empty:
            nifty_df.columns = [col.lower() for col in nifty_df.columns]
            nifty_df["daily_return"] = nifty_df["close"].pct_change()
            bench_returns_dict = {k.strftime("%Y-%m-%d"): v for k, v in nifty_df["daily_return"].items() if not pd.isna(v)}
            print(f"  Successfully loaded {len(bench_returns_dict)} daily benchmark returns from ^NSEI.")
        else:
            print("  Warning: yfinance returned empty Nifty 50 Index. Falling back to average stock returns.")
    except Exception as e:
        print(f"  Warning: Failed to load ^NSEI benchmark returns: {e}. Falling back to average stock returns.")
        
    # Group by date to simulate daily portfolio rebalancing
    grouped = test_df.groupby("date")
    
    dates = []
    strategy_returns = []
    benchmark_returns = []
    
    strat_daily_returns = []
    bench_daily_returns = []
    
    strategy_cum = 100.0
    benchmark_cum = 100.0
    
    print("Simulating trading strategy...")
    for date, group in sorted(grouped):
        # Benchmark return: Nifty 50 Index (^NSEI), falling back to average stock returns
        date_str = date.strftime("%Y-%m-%d")
        if date_str in bench_returns_dict:
            bench_ret = bench_returns_dict[date_str]
        else:
            bench_ret = group["daily_return"].mean()
            
        # Strategy return: select stocks where model predicts Bullish probability > optimal_th
        # If there are none, we stay in cash (0% daily return)
        bullish_stocks = group[group["prob_bullish"] > optimal_th]
        
        if not bullish_stocks.empty:
            strat_ret = bullish_stocks["daily_return"].mean()
        else:
            strat_ret = 0.0 # cash
            
        strategy_cum *= (1 + strat_ret)
        benchmark_cum *= (1 + bench_ret)
        
        dates.append(date_str)
        strategy_returns.append(round(strategy_cum - 100.0, 2))
        benchmark_returns.append(round(benchmark_cum - 100.0, 2))
        
        strat_daily_returns.append(strat_ret)
        bench_daily_returns.append(bench_ret)
        
    # Calculate advanced quantitative metrics
    strat_daily_arr = np.array(strat_daily_returns)
    bench_daily_arr = np.array(bench_daily_returns)
    
    n_days = len(dates)
    years = n_days / 252.0
    
    strat_ann_ret = ((strategy_cum / 100.0) ** (1.0 / years) - 1.0) * 100.0
    bench_ann_ret = ((benchmark_cum / 100.0) ** (1.0 / years) - 1.0) * 100.0
    
    # Daily return standard deviation, annualized (252 trading days)
    strat_vol = np.std(strat_daily_arr) * np.sqrt(252.0) * 100.0
    bench_vol = np.std(bench_daily_arr) * np.sqrt(252.0) * 100.0
    
    # Sharpe Ratio (assuming risk-free rate = 0%)
    strat_sharpe = (strat_ann_ret / strat_vol) if strat_vol > 0.0 else 0.0
    bench_sharpe = (bench_ann_ret / bench_vol) if bench_vol > 0.0 else 0.0
    
    # Maximum Drawdown calculation
    strat_equity = np.cumprod(1.0 + strat_daily_arr)
    bench_equity = np.cumprod(1.0 + bench_daily_arr)
    
    def calc_mdd(equity):
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        return float(np.max(drawdowns)) * 100.0 if len(equity) > 0 else 0.0
        
    strat_mdd = calc_mdd(strat_equity)
    bench_mdd = calc_mdd(bench_equity)
    
    # Win Rate (percentage of days with positive daily return)
    strat_win_rate = (sum(1 for r in strat_daily_returns if r > 0.0) / n_days) * 100.0
    bench_win_rate = (sum(1 for r in bench_daily_returns if r > 0.0) / n_days) * 100.0
    
    # Alpha (Strategy Return - Benchmark Return)
    alpha = strategy_returns[-1] - benchmark_returns[-1]
    
    # Information Ratio (Average daily active return / Tracking error, annualized)
    active_daily = strat_daily_arr - bench_daily_arr
    mean_active = np.mean(active_daily)
    tracking_error = np.std(active_daily)
    info_ratio = (mean_active / tracking_error) * np.sqrt(252.0) if tracking_error > 0.0 else 0.0
    
    # Compile results
    daily_results = []
    for d, s, b in zip(dates, strategy_returns, benchmark_returns):
        daily_results.append({
            "date": d,
            "strategy_return": s,
            "benchmark_return": b
        })
        
    output_data = {
        "daily_returns": daily_results,
        "optimal_threshold": optimal_th,
        "optimization_grid": optimization_logs,
        "metrics": {
            "strategy": {
                "cumulative_return": round(strategy_returns[-1], 2),
                "annualized_return": round(strat_ann_ret, 2),
                "volatility": round(strat_vol, 2),
                "sharpe_ratio": round(strat_sharpe, 3),
                "max_drawdown": round(strat_mdd, 2),
                "win_rate": round(strat_win_rate, 2)
            },
            "benchmark": {
                "cumulative_return": round(benchmark_returns[-1], 2),
                "annualized_return": round(bench_ann_ret, 2),
                "volatility": round(bench_vol, 2),
                "sharpe_ratio": round(bench_sharpe, 3),
                "max_drawdown": round(bench_mdd, 2),
                "win_rate": round(bench_win_rate, 2)
            },
            "alpha": round(alpha, 2),
            "information_ratio": round(info_ratio, 3)
        }
    }
    
    # Write to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"✔ Backtest results saved to {OUTPUT_PATH}")
    print(f"Final Strategy Return: {strategy_returns[-1]:+.2f}%")
    print(f"Final Benchmark Return (Nifty 50 Index): {benchmark_returns[-1]:+.2f}%")
    print(f"Strategy Alpha vs Benchmark: {alpha:+.2f}%")
    print(f"Information Ratio: {info_ratio:+.3f}")

if __name__ == "__main__":
    run_backtest()
