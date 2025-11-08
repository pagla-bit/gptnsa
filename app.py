import streamlit as st
import pandas as pd
import aiohttp
import asyncio
import re
from datetime import datetime
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dateutil import parser as date_parser
import html

# -----------------------------
# Load sentiment models
# -----------------------------
@st.cache_resource
def load_models():
    finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    vader = SentimentIntensityAnalyzer()
    return finbert_tokenizer, finbert_model, vader

finbert_tokenizer, finbert_model, vader = load_models()

# -----------------------------
# Helper: Async scraping
# -----------------------------
async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as r:
            return await r.text()
    except Exception:
        return ""

async def get_finviz_news(session, ticker, max_items):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    html_data = await fetch(session, url)
    soup = BeautifulSoup(html_data, "html.parser")
    rows = soup.select("#news-table tr")
    headlines = []
    for r in rows[:max_items]:
        a_tag = r.find("a")
        if a_tag:
            title = a_tag.text.strip()
            link = a_tag.get("href")
            date_text = r.find("td").text.strip() if r.find("td") else ""
            try:
                ts = date_parser.parse(date_text, fuzzy=True)
                timestamp = ts.strftime("%Y-%m-%d %H:%M")
            except Exception:
                timestamp = ""
            headlines.append({"title": title, "link": link, "timestamp": timestamp, "source": "Finviz"})
    return headlines

async def get_yahoo_news(session, ticker, max_items):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    html_data = await fetch(session, url)
    soup = BeautifulSoup(html_data, "html.parser")
    articles = soup.find_all("li", {"class": re.compile("js-stream-content")})
    headlines = []
    for a in articles[:max_items]:
        link_tag = a.find("a")
        if link_tag:
            title = link_tag.text.strip()
            link = "https://finance.yahoo.com" + link_tag.get("href", "")
            date_el = a.find("time")
            ts = ""
            if date_el and date_el.has_attr("datetime"):
                try:
                    ts = date_parser.parse(date_el["datetime"]).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    ts = ""
            headlines.append({"title": title, "link": link, "timestamp": ts, "source": "Yahoo"})
    return headlines

async def get_google_news(session, ticker, max_items):
    query = f"https://news.google.com/rss/search?q={ticker}+stock"
    xml_data = await fetch(session, query)
    soup = BeautifulSoup(xml_data, "xml")
    items = soup.find_all("item")[:max_items]
    headlines = []
    for i in items:
        title = i.title.text if i.title else ""
        link = i.link.text if i.link else ""
        date_text = i.pubDate.text if i.pubDate else ""
        ts = ""
        try:
            ts = date_parser.parse(date_text).strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts = ""
        headlines.append({"title": title, "link": link, "timestamp": ts, "source": "Google"})
    return headlines

# -----------------------------
# Sentiment analysis functions
# -----------------------------
def vader_sentiment(text):
    score = vader.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "positive"
    elif score["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

def finbert_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = finbert_model(**inputs).logits
    label = torch.argmax(logits, dim=1).item()
    return ["negative", "neutral", "positive"][label]

# -----------------------------
# Main Streamlit App
# -----------------------------
st.title("📈 Stock News Sentiment Analyzer")

tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL, TSLA")
uploaded = st.file_uploader("Or upload Excel with tickers", type=["xlsx"])
max_headlines = st.slider("Max headlines per site", 1, 10, 5)
selected_sites = st.multiselect("Select news sites", ["Finviz", "Yahoo", "Google"], ["Finviz", "Yahoo", "Google"])
selected_models = st.multiselect("Select sentiment models", ["VADER", "FinBERT"], ["VADER", "FinBERT"])

if uploaded:
    df = pd.read_excel(uploaded)
    tickers = df.iloc[:, 0].dropna().astype(str).tolist()
else:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Analyze"):
    async def gather_all():
        async with aiohttp.ClientSession() as session:
            results = {}
            for t in tickers:
                tasks = []
                if "Finviz" in selected_sites:
                    tasks.append(get_finviz_news(session, t, max_headlines))
                if "Yahoo" in selected_sites:
                    tasks.append(get_yahoo_news(session, t, max_headlines))
                if "Google" in selected_sites:
                    tasks.append(get_google_news(session, t, max_headlines))
                all_results = await asyncio.gather(*tasks)
                merged = [item for sublist in all_results for item in sublist]
                results[t] = merged
            return results

    results = asyncio.run(gather_all())

    # -----------------------------
    # Analyze sentiments
    # -----------------------------
    summary = []
    details = []

    for t, headlines in results.items():
        vader_counts = {"positive": 0, "negative": 0, "neutral": 0}
        finbert_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for h in headlines:
            text = h["title"]
            if "VADER" in selected_models:
                h["vader"] = vader_sentiment(text)
                vader_counts[h["vader"]] += 1
            if "FinBERT" in selected_models:
                h["finbert"] = finbert_sentiment(text)
                finbert_counts[h["finbert"]] += 1

        n = len(headlines) if len(headlines) > 0 else 1
        summary.append({
            "ticker": t,
            "v_pos": vader_counts["positive"],
            "v_neg": vader_counts["negative"],
            "v_neu": vader_counts["neutral"],
            "f_pos": finbert_counts["positive"],
            "f_neg": finbert_counts["negative"],
            "f_neu": finbert_counts["neutral"],
            "v_pos_pct": round(vader_counts["positive"]/n*100, 1),
            "v_neg_pct": round(vader_counts["negative"]/n*100, 1),
            "v_neu_pct": round(vader_counts["neutral"]/n*100, 1),
            "f_pos_pct": round(finbert_counts["positive"]/n*100, 1),
            "f_neg_pct": round(finbert_counts["negative"]/n*100, 1),
            "f_neu_pct": round(finbert_counts["neutral"]/n*100, 1)
        })
        for h in headlines:
            details.append({
                "ticker": t,
                "timestamp": h.get("timestamp", ""),
                "title": h.get("title", ""),
                "link": h.get("link", ""),
                "vader": h.get("vader", ""),
                "finbert": h.get("finbert", "")
            })

    # -----------------------------
    # Summary Table Display
    # -----------------------------
    st.subheader("📊 Sentiment Summary by Ticker")

    def ticker_color(t, vp, vn, fp, fn):
        if vp + fp > vn + fn:
            return f"<span style='color:green;font-weight:bold'>{t}</span>"
        elif vn + fn > vp + fp:
            return f"<span style='color:red;font-weight:bold'>{t}</span>"
        else:
            return f"<span style='color:gray;font-weight:bold'>{t}</span>"

    summary_html = "<table style='border-collapse: collapse; width:100%'><tr><th>Ticker</th><th>VADER</th><th>FinBERT</th></tr>"
    for r in summary:
        ticker = ticker_color(r["ticker"], r["v_pos"], r["v_neg"], r["f_pos"], r["f_neg"])
        vader_block = f"🟢 {r['v_pos']} ({r['v_pos_pct']}%) 🔴 {r['v_neg']} ({r['v_neg_pct']}%) 🟡 {r['v_neu']} ({r['v_neu_pct']}%)"
        finbert_block = f"🟢 {r['f_pos']} ({r['f_pos_pct']}%) 🔴 {r['f_neg']} ({r['f_neg_pct']}%) 🟡 {r['f_neu']} ({r['f_neu_pct']}%)"
        summary_html += f"<tr><td>{ticker}</td><td>{vader_block}</td><td>{finbert_block}</td></tr>"
    summary_html += "</table>"

    st.markdown(summary_html, unsafe_allow_html=True)

    # -----------------------------
    # Detailed Table Display
    # -----------------------------
    st.subheader("📰 Detailed Headlines")

    def sort_timestamp(ts):
        try:
            return date_parser.parse(ts)
        except Exception:
            return datetime(1970, 1, 1)

    details.sort(key=lambda x: (x["ticker"], sort_timestamp(x["timestamp"])), reverse=True)

    detail_html = "<table style='border-collapse: collapse; width:100%'><tr><th>Ticker</th><th>Timestamp</th><th>Headline</th><th>VADER</th><th>FinBERT</th></tr>"
    for r in details:
        link = html.escape(r["link"])
        title = html.escape(r["title"])
        headline = f"<a href='{link}' target='_blank'>{title}</a>" if link else title
        vader_ball = "🟢" if r["vader"] == "positive" else "🔴" if r["vader"] == "negative" else "🟡" if r["vader"] == "neutral" else ""
        finbert_ball = "🟢" if r["finbert"] == "positive" else "🔴" if r["finbert"] == "negative" else "🟡" if r["finbert"] == "neutral" else ""
        detail_html += f"<tr><td>{r['ticker']}</td><td>{r['timestamp']}</td><td>{headline}</td><td>{vader_ball}</td><td>{finbert_ball}</td></tr>"
    detail_html += "</table>"

    st.markdown(detail_html, unsafe_allow_html=True)
