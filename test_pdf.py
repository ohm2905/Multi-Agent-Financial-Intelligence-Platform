import os
import requests
import sys

def test_pdf_endpoint():
    print("--- Testing PDF Report Export Endpoint ---")
    
    # We will test for TCS.NS or INFy since they are already analyzed and in the DB.
    # The server runs on port 8000
    base_url = "http://localhost:8000"
    
    # Check health first
    try:
        health_resp = requests.get(f"{base_url}/api/health")
        if health_resp.status_code != 200:
            print(f"❌ Server is not healthy or running: {health_resp.status_code}")
            sys.exit(1)
        print("✔ Connected to FastAPI server successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to FastAPI server at {base_url}: {e}")
        sys.exit(1)
        
    # Let's check for recent analyses to find a ticker that has been analyzed
    recent_resp = requests.get(f"{base_url}/api/analyses/recent")
    if recent_resp.status_code != 200:
        print("❌ Failed to fetch recent analyses.")
        sys.exit(1)
        
    analyses = recent_resp.json()
    if not analyses:
        print("❌ No recent analyses found in database. Run an analysis search first (e.g. TCS).")
        sys.exit(1)
        
    # Pick the first analysis
    analysis = analyses[0]
    ticker = analysis["ticker"]
    clean_ticker = ticker.replace(".NS", "")
    print(f"Testing PDF generation for ticker: {clean_ticker} (from database record for {ticker})...")
    
    pdf_url = f"{base_url}/api/analyses/pdf/{clean_ticker}"
    try:
        pdf_resp = requests.get(pdf_url)
        if pdf_resp.status_code == 200:
            print("✔ PDF generation endpoint returned 200 OK.")
            content_type = pdf_resp.headers.get("content-type")
            content_disposition = pdf_resp.headers.get("content-disposition")
            pdf_bytes = pdf_resp.content
            
            print(f"✔ Content-Type: {content_type}")
            print(f"✔ Content-Disposition: {content_disposition}")
            print(f"✔ PDF Size: {len(pdf_bytes)} bytes")
            
            if content_type != "application/pdf":
                print(f"❌ Failed: Content-Type is not application/pdf (got {content_type})")
                sys.exit(1)
                
            if len(pdf_bytes) < 1000:
                print("❌ Failed: Generated PDF is too small (might be empty or corrupt).")
                sys.exit(1)
                
            # Write to a test file
            output_path = f"test_report_{clean_ticker}.pdf"
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"✔ Saved test PDF to {output_path} (File size: {os.path.getsize(output_path)} bytes)")
            print("\n✔ PDF EXPORT TEST SUCCESSFUL!")
        else:
            print(f"❌ PDF endpoint returned status code {pdf_resp.status_code}: {pdf_resp.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error requesting PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pdf_endpoint()
