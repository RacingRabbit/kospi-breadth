import os, json, pandas as pd, numpy as np, FinanceDataReader as fdr
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

# CONFIG
MARKET = "KOSPI"
START_DATE = "2023-01-01" 
DISPLAY_START = "2024-01-01"
OUTPUT_DIR = "docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DATA FETCH
print("Downloading Market Data...")
tickers = fdr.StockListing(MARKET)["Code"].tolist()
prices_dict = {}
for t in tqdm(tickers, desc="KOSPI Download"):
    try:
        df = fdr.DataReader(t, START_DATE)
        if not df.empty: prices_dict[t] = df["Close"]
    except: continue

prices = pd.DataFrame(prices_dict).ffill()
# Keep index as string for JSON compatibility
prices.index = pd.to_datetime(prices.index)
prices_ytd = prices.loc[prices.index >= DISPLAY_START]

payload = {"dates": prices_ytd.index.strftime('%Y-%m-%d').tolist()}

# BREADTH
for p in [20, 50, 200]:
    sma = prices.rolling(p).mean()
    pct = (prices > sma).sum(axis=1) / sma.count(axis=1) * 100
    payload[f"breadth_{p}"] = pct.loc[pct.index >= DISPLAY_START].round(2).tolist()

# HIGH/LOW
hl = (prices == prices.rolling(252).max()).sum(axis=1) - (prices == prices.rolling(252).min()).sum(axis=1)
payload["high_low"] = hl.loc[hl.index >= DISPLAY_START].astype(int).tolist()

# AD LINE
ad = (prices.diff() > 0).sum(axis=1) - (prices.diff() < 0).sum(axis=1)
ad_line = ad.cumsum()
ad_line_ytd = (ad_line - ad_line.loc[ad_line.index >= DISPLAY_START].iloc[0])
payload["ad_line"] = ad_line_ytd.loc[ad_line_ytd.index >= DISPLAY_START].astype(int).tolist()

# SAVE JSON
with open(f"{OUTPUT_DIR}/data.json", "w") as f:
    json.dump(payload, f)

# AI SUMMARY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
summary_prompt = f"""
Analyze KOSPI health: {payload['breadth_20'][-1]}% above 20D SMA, {payload['breadth_200'][-1]}% above 200D SMA.
Provide a professional report in English and Korean (한국어). Use HTML tags (<b>, <p>).
"""
ai_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}])
with open(f"{OUTPUT_DIR}/ai_summary.html", "w", encoding="utf-8") as f:
    f.write(ai_response.choices[0].message.content)

print("Data ingredients prepared for GitHub Pages.")