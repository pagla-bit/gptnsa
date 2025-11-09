import streamlit as st
import pandas as pd
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")

# ------------------------------
# Sentiment Models
# ------------------------------
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

finbert_tokenizer, finbert_model = load_finbert()
vader_analyzer = SentimentIntensityAnalyzer()

# ------------------------------
# Helper: FinBERT Sentiment
# ------------------------------
def analyze_finbert_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = finbert_model(**inputs).logits
    preds = torch.nn.functional.softmax(logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    return labels[torch.argmax(preds)], preds.max().item()

# ------------------------------
# Helper: VADER Sentiment
# ------------------------------
def analyze_vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "positive"
    elif scores["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# ------------------------------
# Finviz Screener Parsing
# ------------------------------
def get_finviz_tickers_from_screener(url):
    tickers = set()
    page = 1
    while True:
        paged_url = f"{url}&r={(page-1)*20+1}"
        res = requests.get(paged_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select('a.screener-link-primary')
        if not links:
            break
        new_tickers = {a.text.strip() for a in links}
        if not new_tickers - tickers:
            break
        tickers.update(new_tickers)
        page += 1
    return list(tickers)

# ------------------------------
# Async News Scraping
# ------------------------------
async def fetch_finviz_news(session, ticker, n_headlines):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
            text = await resp.text()
        soup = BeautifulSoup(text, "html.parser")
        table = soup.find("table", class_="fullview-news-outer")
        rows = table.find_all("tr")[:n_headlines] if table else []
        news = []
        for r in rows:
            cols = r.find_all("td")
            if len(cols) >= 2:
                date_text = cols[0].text.strip()
                link_tag = cols[1].find("a")
                if link_tag:
                    link = link_tag["href"]
                    title = link_tag.text.strip()
                    dt = date_text.split(" ")[0]
                    news.append({"ticker": ticker, "title": title, "link": link, "datetime": dt})
        return news
    except Exception:
        return []

async def fetch_google_news(session, ticker, n_headlines):
    try:
        url = f"https://news.google.com/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
            text = await resp.text()
        soup = BeautifulSoup(text, "html.parser")
        articles = soup.find_all("article")[:n_headlines]
        news = []
        for a in articles:
            link_tag = a.find("a", href=True)
            if link_tag:
                title = link_tag.text.strip()
                link = "https://news.google.com" + link_tag["href"][1:]
                dt_tag = a.find("time")
                dt = dt_tag["datetime"] if dt_tag else ""
                news.append({"ticker": ticker, "title": title, "link": link, "datetime": dt})
        return news
    except Exception:
        return []

async def gather_all_news(tickers, n_headlines):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for t in tickers:
            tasks.append(fetch_finviz_news(session, t, n_headlines))
            tasks.append(fetch_google_news(session, t, n_headlines))
        all_results = await asyncio.gather(*tasks)
    news = [n for sub in all_results for n in sub]
    return news

# ------------------------------
# Sentiment Analysis
# ------------------------------
def process_sentiments(news_data):
    results = []
    for n in news_data:
        vader_label = analyze_vader_sentiment(n["title"])
        finbert_label, _ = analyze_finbert_sentiment(n["title"])
        dt = n["datetime"]
        try:
            parsed_dt = datetime.strptime(dt.split("T")[0], "%Y-%m-%d")
        except:
            parsed_dt = datetime.now()
        results.append({
            "Ticker": n["ticker"],
            "Date": parsed_dt.strftime("%Y-%m-%d"),
            "Headline": n["title"],
            "Link": n["link"],
            "VADER": vader_label,
            "FinBERT": finbert_label,
            "_dt": parsed_dt
        })
    return pd.DataFrame(results)

# ------------------------------
# Streamlit UI
# ------------------------------
st.sidebar.title("⚙️ Settings")

input_method = st.sidebar.radio(
    "Choose Input Method:",
    ["Manual Entry", "Upload Excel", "Finviz Screener Link"]
)

if input_method == "Manual Entry":
    ticker_input = st.sidebar.text_area("Enter tickers separated by commas:")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

elif input_method == "Upload Excel":
    file = st.sidebar.file_uploader("Upload Excel file with a 'Ticker' column", type=["xlsx"])
    tickers = []
    if file:
        df = pd.read_excel(file)
        if "Ticker" in df.columns:
            tickers = df["Ticker"].dropna().astype(str).str.upper().tolist()

else:
    screener_url = st.sidebar.text_input("Paste Finviz Screener Link:")
    tickers = get_finviz_tickers_from_screener(screener_url) if screener_url else []

n_headlines = st.sidebar.slider("Number of Headlines per Site:", 1, 10, 3)

if tickers:
    st.success(f"Analyzing {len(tickers)} tickers...")

    news_data = asyncio.run(gather_all_news(tickers, n_headlines))
    df = process_sentiments(news_data)

    # ------------------------------
    # Table 1: Sentiment Summary
    # ------------------------------
    summary = (
        df.groupby("Ticker")["VADER", "FinBERT"]
        .apply(lambda x: pd.Series({
            "VADER 🟢": sum(x["VADER"] == "positive"),
            "VADER 🔴": sum(x["VADER"] == "negative"),
            "VADER 🟡": sum(x["VADER"] == "neutral"),
            "FinBERT 🟢": sum(x["FinBERT"] == "positive"),
            "FinBERT 🔴": sum(x["FinBERT"] == "negative"),
            "FinBERT 🟡": sum(x["FinBERT"] == "neutral"),
        }))
        .reset_index()
    )

    # Add % columns
    for colset in [("VADER 🟢", "VADER 🔴", "VADER 🟡"), ("FinBERT 🟢", "FinBERT 🔴", "FinBERT 🟡")]:
        total = summary[list(colset)].sum(axis=1)
        for c in colset:
            summary[f"{c} %"] = (summary[c] / total * 100).round(1).fillna(0)

    # Color tickers
    def color_ticker(row):
        pos = row["VADER 🟢"] + row["FinBERT 🟢"]
        neg = row["VADER 🔴"] + row["FinBERT 🔴"]
        color = "green" if pos >= neg else "red"
        return f"<span style='color:{color};font-weight:bold'>{row['Ticker']}</span>"

    summary["Ticker"] = summary.apply(color_ticker, axis=1)

    st.subheader("📊 Sentiment Summary")
    st.markdown(summary.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ------------------------------
    # Table 1.5: Top Movers
    # ------------------------------
    summary['pos_total'] = summary["VADER 🟢"] + summary["FinBERT 🟢"]
    summary['neg_total'] = summary["VADER 🔴"] + summary["FinBERT 🔴"]

    top_pos = summary.nlargest(3, "pos_total")[["Ticker", "pos_total", "neg_total"]]
    top_neg = summary.nlargest(3, "neg_total")[["Ticker", "pos_total", "neg_total"]]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 3 Positive")
        st.markdown(top_pos.to_html(escape=False, index=False), unsafe_allow_html=True)
    with col2:
        st.subheader("⚠️ Top 3 Negative")
        st.markdown(top_neg.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ------------------------------
    # Table 2: Detailed News (with ticker filter)
    # ------------------------------
    st.subheader("📰 News Details")
    ticker_options = ["All"] + sorted(df["Ticker"].unique().tolist())
    selected_ticker = st.selectbox(
        "🔍 Select Ticker to View News",
        options=ticker_options,
        index=0,
        placeholder="Type to search ticker..."
    )

    if selected_ticker != "All":
        df_filtered = df[df["Ticker"] == selected_ticker]
    else:
        df_filtered = df

    df_filtered = df_filtered.sort_values(by="_dt", ascending=False)
    df_filtered["Headline"] = df_filtered.apply(lambda x: f"<a href='{x['Link']}' target='_blank'>{x['Headline']}</a>", axis=1)

    def color_sentiment(val):
        if val == "positive":
            return "🟢"
        elif val == "negative":
            return "🔴"
        else:
            return "🟡"

    df_filtered["VADER"] = df_filtered["VADER"].apply(color_sentiment)
    df_filtered["FinBERT"] = df_filtered["FinBERT"].apply(color_sentiment)

    st.markdown(
        df_filtered[["Ticker", "Date", "Headline", "VADER", "FinBERT"]]
        .to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
else:
    st.info("Please input or upload tickers to start analysis.")
