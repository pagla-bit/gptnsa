import streamlit as st
import pandas as pd
import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Stock News Sentiment", layout="wide")
st.title("📈 Stock News Sentiment Dashboard")

# Initialize analyzers once
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

vader = SentimentIntensityAnalyzer()
finbert_tokenizer, finbert_model = load_finbert()

# -----------------------------
# Helper functions
# -----------------------------
async def fetch(session, url):
    try:
        async with session.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}) as response:
            return await response.text()
    except:
        return None

async def get_finviz_news(ticker, n_headlines):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="fullview-news-outer")
    if not table:
        return []

    rows = table.find_all("tr")[:n_headlines]
    results = []
    for row in rows:
        a_tag = row.find("a")
        if a_tag:
            link = a_tag["href"]
            headline = a_tag.text.strip()
            td = row.find("td", align="right")
            timestamp = td.text.strip() if td else ""
            results.append({
                "ticker": ticker,
                "headline": headline,
                "link": link,
                "source": "Finviz",
                "timestamp": timestamp
            })
    return results

async def get_google_news(ticker, n_headlines):
    query = f"{ticker} stock"
    url = f"https://news.google.com/rss/search?q={query}"
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, url)
    if not html:
        return []

    soup = BeautifulSoup(html, "xml")
    items = soup.find_all("item")[:n_headlines]
    results = []
    for item in items:
        results.append({
            "ticker": ticker,
            "headline": item.title.text,
            "link": item.link.text,
            "source": "Google News",
            "timestamp": item.pubDate.text if item.pubDate else ""
        })
    return results

def vader_sentiment(text):
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def finbert_sentiment(text):
    tokens = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = finbert_model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=-1)
    label = torch.argmax(scores, dim=1).item()
    return ["Neutral", "Positive", "Negative"][label]

def analyze_sentiments(news_list, use_vader, use_finbert):
    for news in news_list:
        if use_vader:
            news["vader"] = vader_sentiment(news["headline"])
        if use_finbert:
            news["finbert"] = finbert_sentiment(news["headline"])
    return news_list

def format_ball(sentiment):
    if sentiment == "Positive":
        return "🟢"
    elif sentiment == "Negative":
        return "🔴"
    else:
        return "🟡"

def parse_timestamp(ts):
    try:
        return datetime.strptime(ts[:16], "%b-%d-%y %I:%M%p")
    except:
        try:
            return datetime.strptime(ts[:16], "%a, %d %b %Y %H:%M")
        except:
            return datetime.min

# -----------------------------
# Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    input_mode = st.radio(
        "Input method",
        ("Manual (text)", "Upload Excel (single-column)", "Finviz Screener URL"),
        key="input_mode_radio"
    )

    tickers = []
    if input_mode == "Manual (text)":
        user_input = st.text_area("Enter tickers (comma-separated):", key="ticker_input")
        if user_input:
            tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    elif input_mode == "Upload Excel (single-column)":
        file = st.file_uploader("Upload Excel file", type=["xlsx"], key="file_upload")
        if file:
            df = pd.read_excel(file)
            tickers = df.iloc[:, 0].dropna().astype(str).str.upper().tolist()

    elif input_mode == "Finviz Screener URL":
        finviz_url = st.text_input("Paste Finviz Screener URL:", key="finviz_url_input")
        if finviz_url:
            try:
                page = requests.get(finviz_url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(page.text, "html.parser")
                tds = soup.find_all("a", class_="screener-link-primary")
                tickers = [td.text.strip().upper() for td in tds]
            except Exception:
                st.warning("Could not parse Finviz screener URL.")

    n_headlines = st.slider("Number of headlines per site:", 1, 10, 3, key="num_headlines_slider")
    use_vader = st.checkbox("Use VADER Sentiment", value=True, key="use_vader_checkbox")
    use_finbert = st.checkbox("Use FinBERT Sentiment", value=True, key="use_finbert_checkbox")

    analyze = st.button("🔍 Analyze", key="analyze_button")

# -----------------------------
# Main Logic
# -----------------------------
if analyze and tickers:
    st.info(f"Fetching latest {n_headlines} news articles per site...")

    async def gather_all():
        tasks = []
        for t in tickers:
            tasks.append(get_finviz_news(t, n_headlines))
            tasks.append(get_google_news(t, n_headlines))
        results = await asyncio.gather(*tasks)
        merged = [item for sublist in results for item in sublist]
        return merged

    news_data = asyncio.run(gather_all())

    if not news_data:
        st.warning("No news found for the given tickers.")
        st.stop()

    news_data = analyze_sentiments(news_data, use_vader, use_finbert)

    df = pd.DataFrame(news_data)

    # Group summary
    summary = []
    for t in sorted(df["ticker"].unique()):
        row = {"Ticker": t}
        for model in ["vader", "finbert"]:
            if model in df.columns:
                counts = df[df["ticker"] == t][model].value_counts()
                total = counts.sum()
                pos = counts.get("Positive", 0)
                neg = counts.get("Negative", 0)
                neu = counts.get("Neutral", 0)
                row[f"{model}_pos"] = pos
                row[f"{model}_neg"] = neg
                row[f"{model}_neu"] = neu
                row[f"{model}_pos_pct"] = round(100 * pos / total, 1) if total else 0
                row[f"{model}_neg_pct"] = round(100 * neg / total, 1) if total else 0
        summary.append(row)

    df_sum = pd.DataFrame(summary)

    # -----------------------------
    # Sentiment Summary Table
    # -----------------------------
    st.subheader("📊 Sentiment Summary")

    # Prepare styled DataFrame
    def color_ticker(row):
        total_pos = row.get("vader_pos", 0) + row.get("finbert_pos", 0)
        total_neg = row.get("vader_neg", 0) + row.get("finbert_neg", 0)
        if total_pos > total_neg:
            return "color: green; font-weight: bold;"
        elif total_neg > total_pos:
            return "color: red; font-weight: bold;"
        return "color: black;"

    st.dataframe(
        df_sum.style.apply(lambda r: [color_ticker(r)] + [""] * (len(r) - 1), axis=1),
        use_container_width=True
    )

    # -----------------------------
    # Top 3 Positive / Negative Table
    # -----------------------------
    st.subheader("🏆 Top Sentiment Movers")

    if use_vader or use_finbert:
        df_sum["total_pos"] = df_sum.get("vader_pos", 0) + df_sum.get("finbert_pos", 0)
        df_sum["total_neg"] = df_sum.get("vader_neg", 0) + df_sum.get("finbert_neg", 0)
        top_pos = df_sum.nlargest(3, "total_pos")[["Ticker", "total_pos"]]
        top_neg = df_sum.nlargest(3, "total_neg")[["Ticker", "total_neg"]]
        top_pos.rename(columns={"total_pos": "Positive Count"}, inplace=True)
        top_neg.rename(columns={"total_neg": "Negative Count"}, inplace=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 3 Positive**")
            st.table(top_pos)
        with col2:
            st.markdown("**Top 3 Negative**")
            st.table(top_neg)

    # -----------------------------
    # Detailed News Table
    # -----------------------------
    st.subheader("📰 Detailed News")

    # Sort by ticker then timestamp desc
    df["_dt"] = df["timestamp"].apply(parse_timestamp)
    df = df.sort_values(by=["ticker", "_dt"], ascending=[True, False])

    ticker_options = ["All"] + sorted(df["ticker"].unique().tolist())
    selected_ticker = st.selectbox("Select ticker to view:", ticker_options, key="ticker_dropdown")

    if selected_ticker != "All":
        df_filtered = df[df["ticker"] == selected_ticker]
    else:
        df_filtered = df

    df_filtered["VADER"] = df_filtered["vader"].apply(format_ball) if use_vader else ""
    df_filtered["FinBERT"] = df_filtered["finbert"].apply(format_ball) if use_finbert else ""
    df_filtered["Headline"] = df_filtered.apply(lambda x: f"[{x['headline']}]({x['link']})", axis=1)

    st.markdown(df_filtered[["ticker", "Headline", "source", "VADER", "FinBERT", "timestamp"]].to_markdown(index=False), unsafe_allow_html=True)
else:
    st.info("👈 Enter your inputs in the sidebar and click **Analyze** to begin.")
