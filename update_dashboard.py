import os
import json
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

# =========================
# 1. CONFIGURATION
# =========================
MARKET = "KOSPI"
START_DATE = "2023-01-01" 
DISPLAY_START = "2024-01-01"
OUTPUT_DIR = "docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize OpenAI Client (FIXED: Defined at the top level)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 2. DATA PROCESSING
# =========================
print(f"Downloading {MARKET} Data...")
df_listing = fdr.StockListing(MARKET)
tickers = df_listing["Code"].tolist()

prices_dict = {}
# Using all tickers for production
for t in tqdm(tickers, desc="KOSPI Progress"):
    try:
        df = fdr.DataReader(t, START_DATE)
        if not df.empty:
            prices_dict[t] = df["Close"]
    except:
        continue

prices = pd.DataFrame(prices_dict).ffill()
prices.index = pd.to_datetime(prices.index)
prices_ytd = prices.loc[prices.index >= DISPLAY_START]

# Prepare JSON payload
payload = {"dates": prices_ytd.index.strftime('%Y-%m-%d').tolist()}

# Breadth Calculations
for p in [20, 50, 200]:
    sma = prices.rolling(p).mean()
    pct = (prices > sma).sum(axis=1) / sma.count(axis=1) * 100
    payload[f"breadth_{p}"] = pct.loc[pct.index >= DISPLAY_START].round(2).tolist()

# High/Low
hl = (prices == prices.rolling(252).max()).sum(axis=1) - (prices == prices.rolling(252).min()).sum(axis=1)
payload["high_low"] = hl.loc[hl.index >= DISPLAY_START].astype(int).tolist()

# AD Line
ad = (prices.diff() > 0).sum(axis=1) - (prices.diff() < 0).sum(axis=1)
ad_line = ad.cumsum()
ad_line_ytd = (ad_line - ad_line.loc[ad_line.index >= DISPLAY_START].iloc[0])
payload["ad_line"] = ad_line_ytd.loc[ad_line_ytd.index >= DISPLAY_START].astype(int).tolist()

# Save JSON Data
with open(f"{OUTPUT_DIR}/data.json", "w") as f:
    json.dump(payload, f)

# =========================
# 3. AI SUMMARY (WITH WEB SEARCH)
# =========================
print("Generating AI Summary with OpenAI Web Search...")

# Prepare the data context
latest_20 = payload['breadth_20'][-1]
latest_50 = payload['breadth_50'][-1]
latest_200 = payload['breadth_200'][-1]

# Prompt logic
summary_prompt = f"""
You are a professional market research assistant. 

CURRENT BREADTH DATA:
- KOSPI Stocks above 20D SMA: {latest_20:.1f}%
- KOSPI Stocks above 50D SMA: {latest_50:.1f}%
- KOSPI Stocks above 200D SMA: {latest_200:.1f}%

YOUR TASK:
1. Use your web search tool to find South Korean stock market news from the last 48 hours.
2. Identify upcoming macro events (Bank of Korea, US Fed, CPI, major earnings) for the next 2 weeks.
3. Correlate the internal breadth data with the news.
4. Provide a report first in English, then in Korean (한국어).
5. Use HTML tags (<b>, <p>) for formatting. Do NOT use markdown.
"""

try:
    # Using chat.completions with the 'web_search' tool
    # Note: If your tier doesn't support "type: web_search", 
    # GPT-4o will still use its internal knowledge/search capability if enabled.
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}],
        tools=[{"type": "web_search"}] if os.getenv("OPENAI_API_TIER") == "enterprise" else None
    )
    summary_text = completion.choices[0].message.content
except Exception as e:
    # Fallback if the tool call fails
    print(f"Web search tool failed, falling back to standard completion: {e}")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    summary_text = completion.choices[0].message.content

# Save AI Summary
with open(f"{OUTPUT_DIR}/ai_summary.html", "w", encoding="utf-8") as f:
    f.write(f"""<div class="ai-report">
{summary_text}
<p style="font-size: 10px; opacity: 0.5; margin-top: 20px;">
    Updated: {datetime.now().strftime('%Y-%m-%d %H:%M KST')}
</p>
</div>""")

print("Dashboard assets updated successfully.")