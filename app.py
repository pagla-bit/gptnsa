import streamlit as st
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import pandas as pd
import urllib.parse
from typing import List, Dict, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import time

# -----------------------
# Config
# -----------------------
NEWS_PER_SITE = 10
REQUEST_TIMEOUT = 10  # seconds
MAX_TICKERS = 30  # safety limit for UI (you can change)

st.set_page_config(page_title="Ticker News Sentiment", layout="wide")

# -----------------------
# Caching: heavy resources
# -----------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert_pipeline():
    # Use a FinBERT tone model that returns positive/negative/neutral labels
    # This will download the model the first time it's run.
    # Model choice: 'yiyanghkust/finbert-tone' often used for finance tone classification
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", device=-1)

# -----------------------
# Utilities: scraping functions
# -----------------------
def fetch_url(url: str, headers=None, params=None) -> requests.Response:
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp

def parse_finviz(ticker: str, max_items: int = NEWS_PER_SITE) -> List[str]:
    """
    Finviz has a news section for each ticker: https://finviz.com/quote.ashx?t=AAPL
    We'll parse the news table and extract headlines (text).
    """
    url = f"https://finviz.com/quote.ashx"
    params = {"t": ticker}
    try:
        resp = fetch_url(url, params=params)
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    # news table rows appear in the 'news-table' id/class
    news_table = soup.find("table", class_="fullview-news-outer")
    if not news_table:
        return []
    headlines = []
    for a in news_table.find_all("a"):
        txt = a.get_text(strip=True)
        if txt:
            headlines.append(txt)
        if len(headlines) >= max_items:
            break
    return headlines

def parse_yahoo(ticker: str, max_items: int = NEWS_PER_SITE) -> List[str]:
    """
    Scrape Yahoo Finance news list for a ticker.
    Example: https://finance.yahoo.com/quote/AAPL?p=AAPL
    Headlines are in <h3> tags with data-test-locator or links.
    """
    url = f"https://finance.yahoo.com/quote/{urllib.parse.quote(ticker)}?p={urllib.parse.quote(ticker)}"
    try:
        resp = fetch_url(url)
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    headlines = []
    # Yahoo uses many structures. Try some fallbacks.
    for h in soup.find_all(["h3", "h2"]):
        txt = h.get_text(strip=True)
        if txt and len(txt) > 5:
            headlines.append(txt)
        if len(headlines) >= max_items:
            break
    # If not enough, try the newsfeed area
    if len(headlines) < max_items:
        for a in soup.find_all("a"):
            txt = a.get_text(strip=True)
            if txt and len(txt) > 10 and txt not in headlines:
                headlines.append(txt)
            if len(headlines) >= max_items:
                break
    return headlines[:max_items]

def parse_google_news(ticker: str, max_items: int = NEWS_PER_SITE) -> List[str]:
    """
    Use Google News RSS: https://news.google.com/rss/search?q={ticker}
    This is simple and stable.
    """
    query = urllib.parse.quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={query}"
    try:
        resp = fetch_url(url)
    except Exception:
        return []
    soup = BeautifulSoup(resp.content, "xml")
    items = soup.find_all("item")
    headlines = []
    for item in items[:max_items]:
        title = item.title.get_text(strip=True) if item.title else ""
        if title:
            headlines.append(title)
    return headlines

# -----------------------
# Sentiment helpers
# -----------------------
def analyze_vader(vader, text: str) -> str:
    vs = vader.polarity_scores(text)
    c = vs["compound"]
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_finbert(finbert_pipe, text: str) -> str:
    # finbert pipeline returns label and score; label typically 'positive','negative','neutral'
    try:
        out = finbert_pipe(text[:512])  # truncate to 512 tokens roughly to speed up
    except Exception:
        return "neutral"
    if not out:
        return "neutral"
    label = out[0].get("label", "").lower()
    # Some versions return 'POSITIVE' etc. Normalize:
    if "pos" in label:
        return "positive"
    if "neg" in label:
        return "negative"
    return "neutral"

# -----------------------
# High-level processing
# -----------------------
def fetch_all_sites_for_ticker(ticker: str, sites: List[str]) -> List[str]:
    """
    Fetch headlines from selected sites for the ticker, up to NEWS_PER_SITE per site.
    """
    headlines = []
    # We'll call each parser function; wrap each in try/except to be robust.
    if "finviz" in sites:
        headlines += parse_finviz(ticker, NEWS_PER_SITE)
    if "yahoo" in sites:
        headlines += parse_yahoo(ticker, NEWS_PER_SITE)
    if "google" in sites:
        headlines += parse_google_news(ticker, NEWS_PER_SITE)
    # deduplicate while preserving order
    seen = set()
    deduped = []
    for h in headlines:
        if h not in seen:
            deduped.append(h)
            seen.add(h)
    return deduped

@st.cache_data(show_spinner=False)
def fetch_cached_all_sites_for_ticker(ticker: str, sites_tuple: Tuple[str, ...]) -> List[str]:
    # st.cache_data requires hashable args; pass sites as tuple
    return fetch_all_sites_for_ticker(ticker, list(sites_tuple))

def process_ticker(ticker: str, sites: List[str], run_vader: bool, run_finbert: bool) -> Dict:
    """
    Fetch headlines and run sentiment analyzers; return counts:
    {
      'ticker': ticker,
      'vader_pos': int, 'vader_neg': int,
      'finbert_pos': int, 'finbert_neg': int
    }
    """
    ticker = ticker.strip().upper()
    headlines = fetch_cached_all_sites_for_ticker(ticker, tuple(sites))
    vader = load_vader() if run_vader else None
    finbert = load_finbert_pipeline() if run_finbert else None

    vpos = vneg = 0
    fpos = fneg = 0

    for h in headlines:
        if run_vader and vader:
            res = analyze_vader(vader, h)
            if res == "positive":
                vpos += 1
            elif res == "negative":
                vneg += 1
        if run_finbert and finbert:
            res = analyze_finbert(finbert, h)
            if res == "positive":
                fpos += 1
            elif res == "negative":
                fneg += 1

    return {
        "ticker": ticker,
        "vader_pos": vpos,
        "vader_neg": vneg,
        "finbert_pos": fpos,
        "finbert_neg": fneg,
        "n_headlines": len(headlines)
    }

# -----------------------
# Streamlit UI
# -----------------------
st.title("Stock Ticker News Sentiment (VADER + FinBERT)")

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input method", ("Manual (text box)", "Upload CSV/XLSX"))
    sites_multiselect = st.multiselect(
        "News sources to scrape (order doesn't matter)",
        options=["finviz", "yahoo", "google"],
        default=["finviz", "yahoo", "google"],
        help="Finviz, Yahoo Finance, Google News (RSS)."
    )
    st.write("Sentiment analyzers")
    col1, col2 = st.columns(2)
    with col1:
        run_vader = st.checkbox("VADER (fast)", value=True)
    with col2:
        run_finbert = st.checkbox("FinBERT (slower)", value=True)
    max_news = st.number_input("Max headlines per site", min_value=1, max_value=20, value=NEWS_PER_SITE)
    st.write("Performance options")
    max_workers = st.slider("Concurrent workers (fetching)", 2, 20, 8)
    st.write("Tip: reduce FinBERT runs or max headlines to speed up on Streamlit Free tier.")

# sync NEWS_PER_SITE with user
NEWS_PER_SITE = int(max_news)

tickers = []
uploaded_file = None

if input_mode == "Manual (text box)":
    raw = st.text_area("Enter tickers (comma, space or newline separated). Max {} tickers.".format(MAX_TICKERS),
                       placeholder="AAPL, MSFT, TSLA")
    if raw:
        # split on comma/whitespace/lines
        candidates = [t.strip().upper() for t in raw.replace(",", " ").split()]
        tickers = [t for t in candidates if t]
else:
    uploaded_file = st.file_uploader("Upload CSV/XLSX (single column of tickers or file with a column named 'ticker')",
                                     type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # try to extract tickers
            if "ticker" in (c.lower() for c in df.columns):
                # find the column with name 'ticker' case-insensitive
                col = next(c for c in df.columns if c.lower() == "ticker")
                tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
            else:
                # fallback: if single column, take that
                if df.shape[1] == 1:
                    col = df.columns[0]
                    tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
                else:
                    st.warning("Uploaded file has multiple columns and no 'ticker' column. Please upload a single-column file or include a 'ticker' column.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

if len(tickers) > MAX_TICKERS:
    st.warning(f"You provided {len(tickers)} tickers; limiting to first {MAX_TICKERS}.")
    tickers = tickers[:MAX_TICKERS]

run_button = st.button("Run analysis")

# show quick preview
if tickers:
    st.markdown(f"**Tickers to analyze ({len(tickers)}):** {', '.join(tickers[:40])}")

if run_button:
    if not tickers:
        st.error("No tickers provided.")
    elif not sites_multiselect:
        st.error("Select at least one news source.")
    elif not (run_vader or run_finbert):
        st.error("Select at least one sentiment analyzer.")
    else:
        t0 = time.time()
        st.info("Fetching headlines and analyzing... (this may take some seconds — FinBERT is slower)")
        # Use ThreadPoolExecutor to process tickers in parallel
        results = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total = len(tickers)
        completed = 0

        # Bind the model loads early (load into cache so not repeatedly downloaded during executor runs)
        if run_vader:
            _ = load_vader()
        if run_finbert:
            _ = load_finbert_pipeline()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(process_ticker, t, sites_multiselect, run_vader, run_finbert): t for t in tickers
            }
            for fut in concurrent.futures.as_completed(future_to_ticker):
                tkr = future_to_ticker[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "ticker": tkr,
                        "vader_pos": 0,
                        "vader_neg": 0,
                        "finbert_pos": 0,
                        "finbert_neg": 0,
                        "n_headlines": 0
                    }
                    st.warning(f"Error processing {tkr}: {e}")
                results.append(res)
                completed += 1
                progress_bar.progress(completed / total)
                progress_text.text(f"Processed {completed}/{total} tickers")
        progress_bar.empty()
        progress_text.empty()

        # Build DataFrame
        df_res = pd.DataFrame(results).set_index("ticker")
        df_res = df_res[["vader_pos", "vader_neg", "finbert_pos", "finbert_neg", "n_headlines"]]

        # Color the ticker labels based on counts: mostly positive => green, mostly negative => red
        def color_row(ticker, row):
            # Decide based on sum of positive vs negative from selected analyzers
            pos = 0
            neg = 0
            if run_vader:
                pos += row["vader_pos"]
                neg += row["vader_neg"]
            if run_finbert:
                pos += row["finbert_pos"]
                neg += row["finbert_neg"]
            if pos > neg:
                return f"color: green; font-weight: bold"
            elif neg > pos:
                return f"color: red; font-weight: bold"
            else:
                return f"color: black;"

        # For display, we'll show colored ticker names in a separate column
        display_df = df_res.reset_index()
        display_df["ticker_display"] = display_df["ticker"]  # placeholder

        # Use st.table with markdown-like coloring via HTML in st.markdown per-row
        st.subheader("Sentiment counts summary")
        # Build a simple HTML table so we can color the ticker name cell
        def make_html_table(df):
            header_cells = "".join(f"<th style='padding:8px;text-align:left'>{c}</th>" for c in df.columns if c != "ticker_display")
            header = f"<tr><th style='padding:8px;text-align:left'>Ticker</th>{header_cells}</tr>"
            rows = []
            for _, r in df.iterrows():
                # compute color
                pos = 0
                neg = 0
                if run_vader:
                    pos += r["vader_pos"]
                    neg += r["vader_neg"]
                if run_finbert:
                    pos += r["finbert_pos"]
                    neg += r["finbert_neg"]
                if pos > neg:
                    color = "green"
                elif neg > pos:
                    color = "red"
                else:
                    color = "black"
                ticker_html = f"<span style='color:{color};font-weight:bold'>{r['ticker']}</span>"
                other_cells = "".join(f"<td style='padding:6px'>{r[c]}</td>" for c in df.columns if c != "ticker_display")
                rows.append(f"<tr><td style='padding:6px'>{ticker_html}</td>{other_cells}</tr>")
            table = f"<table style='border-collapse: collapse;'>{header}{''.join(rows)}</table>"
            return table

        # Reformat display columns order
        display_df = display_df[["ticker", "vader_pos", "vader_neg", "finbert_pos", "finbert_neg", "n_headlines"]]
        html_table = make_html_table(display_df)
        st.markdown(html_table, unsafe_allow_html=True)

        # Also provide a normal dataframe (downloadable)
        st.subheader("Raw results (downloadable CSV)")
        st.dataframe(df_res)

        csv = df_res.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(label="Download results as CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")

        t_elapsed = time.time() - t0
        st.success(f"Done — processed {len(results)} tickers in {t_elapsed:.1f} s. Headlines per ticker (deduped): see 'n_headlines' column.")
