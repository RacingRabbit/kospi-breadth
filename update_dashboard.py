import os
import json
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

# =========================
# CONFIGURATION
# =========================
MARKET = "KOSPI"
START_DATE = "2023-01-01" 
DISPLAY_START = "2024-01-01"
OUTPUT_DIR = "docs"
DATA_DIR = "docs/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# 1. DATA DOWNLOAD
# =========================
print(f"Fetching {MARKET} Tickers...")
df_listing = fdr.StockListing(MARKET)
tickers = df_listing["Code"].tolist()

prices_dict = {}
for t in tqdm(tickers, desc="Downloading Prices"):
    try:
        df = fdr.DataReader(t, START_DATE)
        if not df.empty:
            prices_dict[t] = df["Close"]
    except:
        continue

prices = pd.DataFrame(prices_dict).ffill()
prices.index = pd.to_datetime(prices.index)

# =========================
# 2. CALCULATIONS
# =========================
print("Calculating Market Internals...")

# Breadth (SMA)
breadth_data = {}
for p in [20, 60, 120, 200]:
    sma = prices.rolling(p).mean()
    pct = (prices > sma).sum(axis=1) / sma.count(axis=1) * 100
    breadth_data[f"above_{p}"] = pct

breadth_df = pd.DataFrame(breadth_data)
breadth_df.to_csv(f"{DATA_DIR}/breadth_sma.csv")

# 52-Week Highs - Lows
high_52w = prices == prices.rolling(252).max()
low_52w = prices == prices.rolling(252).min()
hl_df = pd.DataFrame({
    "net": high_52w.sum(axis=1) - low_52w.sum(axis=1)
})
hl_df.to_csv(f"{DATA_DIR}/high_low_52w.csv")

# Advance-Decline Line
returns = prices.diff()
net_adv = (returns > 0).sum(axis=1) - (returns < 0).sum(axis=1)
ad_line = net_adv.cumsum()
# Anchor AD Line to DISPLAY_START for the dashboard
ad_line_display = ad_line - ad_line.loc[ad_line.index >= DISPLAY_START].iloc[0]

ad_df = pd.DataFrame({"ad_line": ad_line_display})
ad_df.to_csv(f"{DATA_DIR}/advance_decline.csv")

# =========================
# 3. SAVE JSON FOR DASHBOARD
# =========================
# This part ensures your index.html charts continue to work
prices_ytd = prices.loc[prices.index >= DISPLAY_START]
payload = {
    "dates": prices_ytd.index.strftime('%Y-%m-%d').tolist(),
    "breadth_20": breadth_df.loc[breadth_df.index >= DISPLAY_START, "above_20"].round(2).tolist(),
    "breadth_50": breadth_df.loc[breadth_df.index >= DISPLAY_START, "above_60"].round(2).tolist(), # Using 60 as 50 proxy
    "breadth_200": breadth_df.loc[breadth_df.index >= DISPLAY_START, "above_200"].round(2).tolist(),
    "high_low": hl_df.loc[hl_df.index >= DISPLAY_START, "net"].astype(int).tolist(),
    "ad_line": ad_df.loc[ad_df.index >= DISPLAY_START, "ad_line"].astype(int).tolist()
}
with open(f"{OUTPUT_DIR}/data.json", "w") as f:
    json.dump(payload, f)


# ========================================================
# 4. AI SUMMARY (BREADTH + NEWS) - YOUR SPECIFIC BLOCK
# ========================================================
print("Starting AI Analysis with Web Search...")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load latest CSV data ---

# Breadth (SMA)
breadth_df_csv = pd.read_csv(
    f"{DATA_DIR}/breadth_sma.csv",
    index_col=0,
    parse_dates=True
)

latest = breadth_df_csv.iloc[-1]

breadth_20 = latest["above_20"]
breadth_60 = latest["above_60"]
breadth_120 = latest["above_120"]
breadth_200 = latest["above_200"]

# 52-week highs / lows
high_low = pd.read_csv(
    f"{DATA_DIR}/high_low_52w.csv",
    index_col=0,
    parse_dates=True
).iloc[-1]

# Advance–decline
ad_line_csv = pd.read_csv(
    f"{DATA_DIR}/advance_decline.csv",
    index_col=0,
    parse_dates=True
).iloc[-1]


breadth_summary = f"""
Percent above moving averages:
20D: {breadth_20:.2f}%
60D: {breadth_60:.2f}%
120D: {breadth_120:.2f}%
200D: {breadth_200:.2f}%

52-week highs minus lows: {high_low['net']}
Advance–Decline line (latest): {ad_line_csv['ad_line']}
"""


# --- Web search helper ---
def get_market_news(query):
    # Using the beta responses tool as requested
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search"}],
        input=f"""
Summarize the most important stock market news from the last 24 hours.
Focus only on macro, earnings, policy, rates, or major risk events.
Be factual. No opinions.

Query: {query}
"""
    )
    return response.output_text.strip()

# --- Fetch news ---
kr_news = get_market_news("Korean stock market news last 24 hours")

# --- Final combined AI summary ---
final_prompt = f"""
You are a professional market research assistant.

You are given:
1) Quantitative market breadth indicators
2) Current market news

Your task:
- Explain how the news context supports, contradicts, or explains the market internals
- Do NOT predict prices
- Do NOT give trading advice
- Be concise, factual, and neutral
- Provide an outlook for the coming weeks
- Provide the text first in English and afterwards in Korean

MARKET BREADTH DATA:
{breadth_summary}

KOREA MARKET NEWS:
{kr_news}
"""

final_response = client.responses.create(
    model="gpt-4.1-mini",
    input=final_prompt
)

summary_text = final_response.output_text.strip()

# --- Convert to HTML-safe format ---
summary_html = summary_text.replace("\n", "<br>")

# --- Write HTML file ---
with open("docs/ai_summary.html", "w", encoding="utf-8") as f:
    f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: system-ui; line-height:1.6; padding:14px;">
<h2>Daily AI Market Breadth Summary</h2>
{summary_html}
</body>
</html>""")

print("AI summary with market news generated successfully.")