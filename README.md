# Multi-Agent Financial Intelligence Platform

An advanced, AI-powered financial intelligence and automated stock advisory copilot for the National Stock Exchange (NSE) of India. This platform leverages a multi-agent workflow (built with LangGraph) to analyze stocks, predict 30-day price trends, summarize corporate filings via local RAG, evaluate portfolio risk, and track system performance observability.

---

## 📸 Platform Showcases

### 1. Main Advisory Dashboard
*Real-time NSE stock analysis aggregating technical indicators, live FinBERT news sentiment, and XGBoost forecasts with dynamic SHAP explainability.*
![Advisory Dashboard](./s_c/Screenshot%202026-06-16%20at%207.27.36%20AM.png)

### 2. Shares-Based Portfolio Risk Analyzer
*Add asset share counts; the system automatically fetches live prices, computes portfolio allocations, calculates portfolio Beta, and scores diversification.*
![Portfolio Manager](./s_c/Screenshot%202026-06-16%20at%207.47.23%20AM.png)

### 3. Corporate Report Audit & Chat Assistant
*Upload corporate PDFs, auto-generate strategic highlights, and chat conversationally with Gemini 2.5 Flash using local vector indexing.*
![RAG Chat Assistant](./s_c/Screenshot%202026-06-16%20at%207.48.16%20AM.png)

### 4. Consolidated AI Advisory Report
*Generate and download complete markdown/PDF executive summaries detailing drift analysis, financial ratios, and confidence scores.*
![Advisory Report Preview](./s_c/Screenshot%202026-06-16%20at%207.50.17%20AM.png)

---

## 🚀 Key Features

1. **Multi-Agent Graph Orchestration (LangGraph)**
   * Coordinates specialized agents: **Financial Agent** (yfinance ratios), **News Agent** (NewsAPI articles), **Risk Agent** (Beta & volatility), **ML Prediction Agent** (XGBoost trend), and **Reporter Agent** (PDF template compiling).
2. **Defensible Technical-Only XGBoost Forecaster**
   * Predicts 30-day stock direction (Bullish, Bearish, Neutral) using 8 technical indicators (RSI, MACD, Bollinger Bands, Price/Volume returns) trained on 5 years of historical Nifty 50 data.
3. **SHAP Explainability & Confidence Levels**
   * Attributes predictions to features using `shap.TreeExplainer`. Includes dynamic, context-aware headers ("Bullish Support Signals") and confidence levels (Low, Medium, High).
4. **Corporate Report RAG Pipeline**
   * Parses annual filings, splits text recursively, computes local embeddings via `BAAI/bge-small-en-v1.5`, indexes vectors into a multi-tenant **ChromaDB**, and provides Q&A with source and page citations.
5. **Shares-Based Portfolio Advisor**
   * Calculates dynamic weights based on asset share counts and current market values. Evaluates portfolio Beta, calculates a diversification index, and provides asset concentration advice.
6. **System Observability & Backtest Engine**
   * Computes out-of-sample backtesting metrics (Sharpe, Information Ratio, Alpha, Drawdown). Automatically runs an evaluation loop comparing past predictions against matured market return prices.

---

## 🛠️ Technology Stack

* **Backend**: FastAPI (Python 3.9)
* **AI Orchestration**: LangGraph, LangChain, Gemini 2.5 Flash
* **Machine Learning**: XGBoost, SHAP, Scikit-Learn
* **NLP**: HuggingFace Transformers (FinBERT, PyTorch)
* **Vector Database**: ChromaDB
* **Databases & Caching**: PostgreSQL (Production) / SQLite (Local), Redis
* **Frontend**: HTML5, Vanilla CSS, Bootstrap 5, Chart.js

---

## 🐳 Docker Deployment (Production)

The production configuration runs a multi-container stack comprising the FastAPI backend, a PostgreSQL relational database, and a Redis cache container. All database schemas and migrations are initialized automatically.

### Prerequisites
* Docker & Docker Compose installed.
* Gemini API Key, Tavily API Key, and NewsAPI Key.

### 1. Configure Credentials
Create a `.env` file inside the `backend/` directory:
```bash
# backend/.env
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
NEWS_API_KEY=your_news_api_key
```

### 2. Spin Up Stack
Run the following command from the root directory:
```bash
docker-compose up -d --build
```
This will:
* Build the python backend image and spin up the FastAPI server on port `8000`.
* Initialize a Postgres database on port `5433` (internal `5432`).
* Initialize a Redis cache server on port `6379`.
* Persist caches and vector indices using persistent Docker volume mounts (`backend_data`, `postgres_data`, `redis_data`).

---

## 💻 Local Development Setup

If you wish to run the application outside of Docker:

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Run Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The local configuration runs on **SQLite** by default, saving data inside `./backend/data/financial_platform.db`.
