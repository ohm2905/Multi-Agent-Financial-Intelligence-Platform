import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

logger = logging.getLogger("financial_platform.utils.preload_reports")

def create_report_pdf(filename, title, content_paragraphs):
    """Programmatically generate a styled PDF report for testing/preloading."""
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        rightMargin=54, 
        leftMargin=54,
        topMargin=54, 
        bottomMargin=54
    )
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_p = Paragraph(f"<b><font size='20' color='#1A365D'>{title}</font></b>", styles['Heading1'])
    story.append(title_p)
    story.append(Spacer(1, 15))
    
    # Body text
    for p in content_paragraphs:
        if p.startswith("## "):
            story.append(Spacer(1, 10))
            heading = Paragraph(f"<b><font size='13' color='#2B6CB0'>{p.replace('## ', '')}</font></b>", styles['Heading2'])
            story.append(heading)
            story.append(Spacer(1, 6))
        else:
            body = Paragraph(p, styles['BodyText'])
            story.append(body)
            story.append(Spacer(1, 8))
            
    doc.build(story)

# Core financial data summaries for major Nifty companies
PRELOAD_DATA = {
    "TCS": {
        "title": "Tata Consultancy Services Limited (TCS) - FY25 Annual Highlights",
        "paragraphs": [
            "## 1. Corporate Executive Summary",
            "Tata Consultancy Services is the leading global IT services, consulting, and business solutions organization. During FY25, the company consolidated its market leadership through large-scale strategic delivery contracts and deep expertise in digital transformation platforms.",
            "## 2. Financial Highlights and Performance",
            "TCS reported a record consolidated revenue of INR 2,40,893 Crore for FY25, representing a steady year-on-year growth of 6.8%. The net profit rose to INR 46,230 Crore. Operating margins remained resilient at 24.6%, driven by internal cost optimizations, automation frameworks, and workforce alignment initiatives.",
            "## 3. Strategic Growth Drivers",
            "The company's primary growth drivers include the rapid expansion of GenAI business units (AI.Cloud), large cloud migration contracts with global enterprises, and massive public sector modernization deals in the UK and Europe. Cloud services and cyber security products showed double-digit expansion.",
            "## 4. Key Risk Factors and Mitigation",
            "TCS identified three major risk areas: talent inflation and attrition pressures in high-skill areas, high concentration of revenue from the US banking and financial services (BFS) market, and potential macro-economic spending delays in enterprise software budgets. Mitigations include global talent hubs and diversified geographic focus."
        ]
    },
    "INFY": {
        "title": "Infosys Limited - FY25 Annual Highlights",
        "paragraphs": [
            "## 1. Corporate Executive Summary",
            "Infosys Limited is a global leader in next-generation digital services and business consulting. The company is committed to enabling enterprises to navigate their digital transformation journeys through cloud-first and AI-driven platforms.",
            "## 2. Financial Highlights and Performance",
            "For FY25, Infosys recorded annual revenues of INR 1,53,670 Crore, registering a year-on-year growth of 4.2%. Net profit stood at INR 26,240 Crore. Operating margins settled at 20.8%, within the company's guided range, supported by efficiency improvements under the Project Maximize framework.",
            "## 3. Strategic Growth Drivers",
            "Key strategic drivers include the expansion of the 'Infosys Topaz' AI offering, which integrates generative AI capabilities across client delivery systems. The company also saw high demand for cloud suite solutions (Infosys Cobalt) and enterprise cyber security platforms.",
            "## 4. Key Risk Factors and Mitigation",
            "Major risk factors include pricing pressures in traditional IT outsourcing services, client concentration in global financial verticals, and geopolitical uncertainties. Infosys is mitigating these risks by increasing high-margin digital advisory engagements and growing its footprint in APAC."
        ]
    },
    "RELIANCE": {
        "title": "Reliance Industries Limited (RIL) - FY25 Annual Highlights",
        "paragraphs": [
            "## 1. Corporate Executive Summary",
            "Reliance Industries Limited is India's largest private sector conglomerate, spanning oil-to-chemicals (O2C), telecom (Jio), retail, and new energy verticals. RIL continues to build scalable consumer platforms backed by deep technological infrastructure.",
            "## 2. Financial Highlights and Performance",
            "RIL reported consolidated annual revenues of INR 10,00,320 Crore ($120 Billion) in FY25, growing 9.4% year-on-year. Net profit reached INR 79,020 Crore. Growth was led by double-digit retail footprint expansion and higher average revenue per user (ARPU) in telecom (Jio).",
            "## 3. Strategic Growth Drivers",
            "Telecom growth was driven by nationwide 5G monetization and enterprise fiber rollouts. Retail growth was powered by digital commerce integrations. New Energy strategic growth is led by the construction of Gigafactories in Jamnagar for solar panels, green hydrogen, and advanced energy storage systems.",
            "## 4. Key Risk Factors and Mitigation",
            "Major risks include volatility in global oil refining margins (GRMs), high capital expenditure requirements for 5G network expansion, and potential regulatory shifts. RIL mitigates O2C risks through deep downstream chemical integration and O2C product diversification."
        ]
    },
    "HDFCBANK": {
        "title": "HDFC Bank Limited - FY25 Annual Highlights",
        "paragraphs": [
            "## 1. Corporate Executive Summary",
            "HDFC Bank is India's largest private sector bank, delivering comprehensive banking, loan, and financial services to retail, MSME, and corporate clients across the nation. The bank completed its major integration milestones following the merger.",
            "## 2. Financial Highlights and Performance",
            "HDFC Bank reported annual Net Interest Income of INR 1,12,450 Crore for FY25. Net profit rose to INR 64,060 Crore. The bank maintained a healthy Net Interest Margin (NIM) of 3.8% post-merger, and gross non-performing assets (GNPA) remained stable at 1.24%.",
            "## 3. Strategic Growth Drivers",
            "Strategic growth is driven by accelerated branch expansion in semi-urban and rural regions, digital retail banking adoption (Payzapp and SmartBuy), and cross-selling loan and mortgage products to the pre-merger retail base. Deposit franchise growth remained robust.",
            "## 4. Key Risk Factors and Mitigation",
            "The bank faces potential NIM compression risks due to high competition for deposits, credit cost management within MSME portfolios, and systemic liquidity constraints. Mitigations include building deep retail deposit connections and using AI-based credit profiling."
        ]
    }
}

def preload_corporate_reports():
    """Generate and index major company reports in ChromaDB if not already present."""
    from app.agents.rag_agent import has_corporate_reports
    from app.rag.pdf_loader import load_pdf
    from app.rag.chunker import split_documents
    from app.rag.vector_store import add_documents_to_store
    
    # Setup path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preloaded_dir = os.path.join(base_dir, "data", "preloaded")
    os.makedirs(preloaded_dir, exist_ok=True)
    
    logger.info("Checking preloaded corporate reports for RAG...")
    
    for ticker, data in PRELOAD_DATA.items():
        # Check if already has documents
        if has_corporate_reports(ticker):
            logger.info(f"RAG vector index already exists for ticker: {ticker}. Skipping preloading.")
            continue
            
        pdf_path = os.path.join(preloaded_dir, f"{ticker}_Highlights.pdf")
        logger.info(f"Generating preloaded PDF report for {ticker} at: {pdf_path}")
        
        try:
            # 1. Generate the PDF highlights report
            create_report_pdf(pdf_path, data["title"], data["paragraphs"])
            
            # 2. Parse the PDF pages
            documents = load_pdf(pdf_path)
            
            # 3. Split into overlapping character chunks
            chunks = split_documents(documents)
            
            # 4. Save chunks into ChromaDB indexed by the ticker
            logger.info(f"Indexing {len(chunks)} chunks into ChromaDB for preloaded ticker: {ticker}...")
            add_documents_to_store(chunks, ticker)
            
            # 5. Remove the preloaded temp file to keep directory clean
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
            logger.info(f"Successfully preloaded and indexed report for ticker: {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to preload report for {ticker}: {e}")
            # Ensure cleanup
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except:
                    pass
