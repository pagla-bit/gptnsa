# app.py
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
import html
from dateutil import parser as date_parser

# -----------------------
# Config & globals
# -----------------------
st.set_page_config(page_title="Ticker News Sentiment", layout="wide")
REQUEST_TIMEOUT = 10
MAX_TICKERS = 200  # safety cap for screener extraction

# -----------------------
# Cached model loaders
# -----------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert_pipeline():
    # Hugging Face FinBERT tone model (may download once)
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", device=-1)

# -----------------------
# Networking helper
# -----------------------
def fetch_url(url: str, headers=None, params=None) -> requests.Response:
    headers = headers or {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp

# -----------------------
# Finviz screener parser (follows pagination to collect tickers)
# -----------------------
def extract_tickers_from_finviz_screener(screener_url: str, max_total: int = 500) -> List[str]:
    """
    Given a Finviz screener URL (e.g. https://finviz.com/screener.ashx?v=111&f=...),
    follow pagination and extract all tickers. Pagination uses 'r=' parameter (1,21,41,...).
    Stops when no new tickers found or max_total reached.
    """
    if not screener_url:
        return []
    parsed = urllib.parse.urlparse(screener_url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    query = urllib.parse.parse_qs(parsed.query)
    # Keep existing query params except 'r'
    base_params = {k: v[0] for k, v in query.items() if k != "r"}
    tickers = []
    seen = set()
    start = 1
    page_size = 20  # Finviz shows 20 tickers per page typically
    while True:
        params = dict(base_params)
        params["r"] = str(start)
        try:
            resp = fetch_url(base, params=params)
        except Exception:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        # Find ticker links - usually anchors with href like 'quote.ashx?t=AAPL'
        anchors = soup.find_all("a", href=True)
        page_tickers = []
        for a in anchors:
            href = a["href"]
            if "quote.ashx?t=" in href:
                # extract ticker param
                q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                t = q.get("t")
                if t:
                    ticker = t[0].strip().upper()
                    if ticker and ticker not in seen:
                        page_tickers.append(ticker)
                        seen.add(ticker)
                        tickers.append(ticker)
                        if len(tickers) >= max_total:
                            break
        # If no tickers found on this page, stop
        if not page_tickers:
            break
        # If we reached fewer than page_size found, might be last page - but continue to check next page until no new found
        if len(tickers) >= max_total:
            break
        start += page_size
        # safety break
        if start > 2000:
            break
        # small loop continue to next page
    return tickers

# -----------------------
# Parsers (Finviz and Google news)
# -----------------------
def parse_finviz(ticker: str, max_items: int = 10) -> List[Dict]:
    """
    Parse Finviz news table. Returns list of {"title","link","source":"finviz","timestamp": "..."}.
    """
    url = "https://finviz.com/quote.ashx"
    params = {"t": ticker}
    try:
        resp = fetch_url(url, headers={"User-Agent": "Mozilla/5.0"}, params=params)
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    results = []
    if not news_table:
        # fallback: find anchors with quote links or headlines
        for a in soup.find_all("a"):
            txt = a.get_text(strip=True)
            if txt and len(txt) > 10:
                href = a.get("href")
                link = href if href and href.startswith("http") else ("https://finviz.com/" + href.lstrip("/") if href else "")
                results.append({"title": txt, "link": link, "source": "finviz", "timestamp": ""})
                if len(results) >= max_items:
                    break
        return results[:max_items]

    for tr in news_table.find_all("tr"):
        tds = tr.find_all("td")
        timestamp = ""
        title = ""
        link = ""
        if len(tds) >= 2:
            ts_text = tds[0].get_text(strip=True)
            timestamp = ts_text
            a = tds[1].find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href")
                if href and href.startswith("http"):
                    link = href
                elif href:
                    link = "https://finviz.com/" + href.lstrip("/")
        else:
            a = tr.find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href")
                if href and href.startswith("http"):
                    link = href
                elif href:
                    link = "https://finviz.com/" + href.lstrip("/")
        if title:
            results.append({"title": title, "link": link, "source": "finviz", "timestamp": timestamp})
            if len(results) >= max_items:
                break
    return results[:max_items]

def parse_google_news(ticker: str, max_items: int = 10) -> List[Dict]:
    """
    Use Google News RSS for the search query.
    """
    query = urllib.parse.quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={query}"
    try:
        resp = fetch_url(url)
    except Exception:
        return []
    try:
        soup = BeautifulSoup(resp.content, "xml")
    except Exception:
        soup = BeautifulSoup(resp.content, "html.parser")
    items = soup.find_all("item")
    results = []
    for item in items[:max_items]:
        title = item.title.get_text(strip=True) if item.title else ""
        link = item.link.get_text(strip=True) if item.link else ""
        ts = ""
        if item.pubDate:
            try:
                ts = item.pubDate.get_text(strip=True)
            except Exception:
                ts = ""
        results.append({"title": title, "link": link, "source": "google", "timestamp": ts})
    return results

# -----------------------
# Sentiment analysis functions
# -----------------------
def analyze_vader(text: str, vader) -> str:
    if not text:
        return "neutral"
    scores = vader.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        return "positive"
    elif comp <= -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_finbert_batch(texts: List[str], finbert) -> List[str]:
    """
    Batch predict with FinBERT. Returns a list of labels: "positive", "negative", "neutral".
    """
    if not texts:
        return []
    results = finbert(texts, top_k=1)
    return [r[0]["label"].lower() for r in results]

# -----------------------
# Main processing function (per ticker)
# -----------------------
def process_ticker(ticker: str, vader, finbert, news_per_ticker: int = 10) -> Dict:
    """
    For one ticker: fetch from Finviz + Google, analyze sentiment, return summary:
    {
      "ticker": str,
      "vader_pos": int,
      "vader_neg": int,
      "vader_neu": int,
      "finbert_pos": int,
      "finbert_neg": int,
      "finbert_neu": int,
      "headlines": list of {title,link,timestamp,vader,finbert}
    }
    """
    finviz_news = parse_finviz(ticker, max_items=news_per_ticker)
    google_news = parse_google_news(ticker, max_items=news_per_ticker)
    all_news = finviz_news + google_news

    if not all_news:
        return {
            "ticker": ticker,
            "vader_pos": 0,
            "vader_neg": 0,
            "vader_neu": 0,
            "finbert_pos": 0,
            "finbert_neg": 0,
            "finbert_neu": 0,
            "headlines": []
        }

    # dedupe by title (case-insensitive)
    seen = set()
    unique_news = []
    for n in all_news:
        t = n["title"].strip().lower()
        if t and t not in seen:
            seen.add(t)
            unique_news.append(n)

    # split text
    titles = [n["title"] for n in unique_news]

    # VADER
    vader_labels = [analyze_vader(t, vader) for t in titles]

    # FinBERT (batch)
    finbert_labels = analyze_finbert_batch(titles, finbert)

    # counts
    vader_pos = vader_labels.count("positive")
    vader_neg = vader_labels.count("negative")
    vader_neu = vader_labels.count("neutral")
    finbert_pos = finbert_labels.count("positive")
    finbert_neg = finbert_labels.count("negative")
    finbert_neu = finbert_labels.count("neutral")

    # build headlines array
    headlines = []
    for n, vl, fl in zip(unique_news, vader_labels, finbert_labels):
        headlines.append({
            "title": n["title"],
            "link": n["link"],
            "timestamp": n["timestamp"],
            "vader": vl,
            "finbert": fl
        })

    return {
        "ticker": ticker,
        "vader_pos": vader_pos,
        "vader_neg": vader_neg,
        "vader_neu": vader_neu,
        "finbert_pos": finbert_pos,
        "finbert_neg": finbert_neg,
        "finbert_neu": finbert_neu,
        "headlines": headlines
    }

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.title("📊 Multi-Ticker News Sentiment Analyzer")
    st.markdown("Fetch and analyze news sentiment for multiple tickers (Finviz + Google News) using VADER & FinBERT.")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        st.markdown("### Input Method")
        input_choice = st.radio(
            "Choose input method:",
            options=["Enter Tickers Manually", "Use Finviz Screener URL"],
            index=0
        )
        
        tickers_to_process = []
        
        if input_choice == "Enter Tickers Manually":
            tickers_input = st.text_area(
                "Enter tickers (comma-separated or one per line)",
                value="AAPL,GOOGL,MSFT,TSLA,NVDA",
                height=100
            )
            if tickers_input.strip():
                raw = tickers_input.replace(",", " ").replace("\n", " ").strip()
                tickers_to_process = [t.strip().upper() for t in raw.split() if t.strip()]
        else:
            screener_url = st.text_input(
                "Finviz Screener URL",
                value="",
                placeholder="https://finviz.com/screener.ashx?v=111&f=..."
            )
            if screener_url.strip():
                with st.spinner("Extracting tickers from screener..."):
                    tickers_to_process = extract_tickers_from_finviz_screener(screener_url, max_total=MAX_TICKERS)
                if tickers_to_process:
                    st.success(f"Extracted {len(tickers_to_process)} tickers from screener")
                    with st.expander("Show extracted tickers"):
                        st.write(", ".join(tickers_to_process))
                else:
                    st.error("No tickers found. Check the URL.")

        st.markdown("---")
        news_per_ticker = st.slider(
            "Max news items per ticker (per source)",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        
        max_workers = st.slider(
            "Concurrent workers",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of parallel threads for fetching news"
        )

    # Main area
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if not tickers_to_process:
            st.warning("Please enter tickers or provide a screener URL.")
            return

        st.info(f"Processing {len(tickers_to_process)} tickers with {max_workers} workers...")
        t0 = time.time()

        # load models
        vader = load_vader()
        finbert = load_finbert_pipeline()

        # parallel fetch
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(process_ticker, tkr, vader, finbert, news_per_ticker): tkr for tkr in tickers_to_process}
            for future in concurrent.futures.as_completed(future_map):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    tkr = future_map[future]
                    st.warning(f"Error processing {tkr}: {e}")

        if not results:
            st.error("No results to display.")
            return

        # Create display dataframe for summary
        display_df = pd.DataFrame(results).set_index("ticker")

        st.subheader("Sentiment counts summary")
        def ticker_color_style(ticker, row):
            pos = row.get("vader_pos", 0) + row.get("finbert_pos", 0)
            neg = row.get("vader_neg", 0) + row.get("finbert_neg", 0)
            if pos > neg:
                return f"<span style='color:green;font-weight:bold'>{html.escape(ticker)}</span>"
            elif neg > pos:
                return f"<span style='color:red;font-weight:bold'>{html.escape(ticker)}</span>"
            else:
                return f"<span style='color:gray;font-weight:bold'>{html.escape(ticker)}</span>"

        header = (
            "<tr>"
            "<th style='padding:8px;text-align:left'>Ticker</th>"
            "<th style='padding:8px;text-align:center'>VADER</th>"
            "<th style='padding:8px;text-align:center'>FINBERT</th>"
            "</tr>"
        )
        rows_html = []
        for ticker, row in display_df.reset_index().iterrows():
            tv = ticker_color_style(row["ticker"], row)
            vader_cell = f"🟢 {row['vader_pos']} &nbsp;&nbsp; 🔴 {row['vader_neg']} &nbsp;&nbsp; 🟡 {row['vader_neu']}"
            finbert_cell = f"🟢 {row['finbert_pos']} &nbsp;&nbsp; 🔴 {row['finbert_neg']} &nbsp;&nbsp; 🟡 {row['finbert_neu']}"
            rows_html.append(f"<tr><td style='padding:6px'>{tv}</td>"
                             f"<td style='padding:6px;text-align:center'>{vader_cell}</td>"
                             f"<td style='padding:6px;text-align:center'>{finbert_cell}</td></tr>")
        summary_table = f"<table style='border-collapse: collapse; width:100%'>{header}{''.join(rows_html)}</table>"
        st.markdown(summary_table, unsafe_allow_html=True)

        # -----------------------
        # Top Movers (side-by-side): top 3 positive and top 3 negative
        # -----------------------
        st.subheader("Top Movers (by total counts)")

        # compute scores
        movers = []
        for r in results:
            pos_total = r.get("vader_pos", 0) + r.get("finbert_pos", 0)
            neg_total = r.get("vader_neg", 0) + r.get("finbert_neg", 0)
            movers.append({"ticker": r["ticker"], "pos_total": pos_total, "neg_total": neg_total})

        movers_df = pd.DataFrame(movers)

        top_pos = movers_df.sort_values(by="pos_total", ascending=False).head(3).reset_index(drop=True) if not movers_df.empty else pd.DataFrame()
        top_neg = movers_df.sort_values(by="neg_total", ascending=False).head(3).reset_index(drop=True) if not movers_df.empty else pd.DataFrame()

        # build two small html tables side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 3 Positive**")
            table_html = "<table style='border-collapse: collapse; width:100%'><tr><th>Rank</th><th>Ticker</th><th>Total Positive</th></tr>"
            for i in range(3):
                if i < len(top_pos):
                    row = top_pos.loc[i]
                    ticker_html = ticker_color_style(row["ticker"], {"vader_pos": 0, "vader_neg": 0, "finbert_pos": 0, "finbert_neg": 0})
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'>{ticker_html}</td><td style='padding:6px'>{int(row['pos_total'])}</td></tr>"
                else:
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'></td><td style='padding:6px'></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

        with col2:
            st.markdown("**Top 3 Negative**")
            table_html = "<table style='border-collapse: collapse; width:100%'><tr><th>Rank</th><th>Ticker</th><th>Total Negative</th></tr>"
            for i in range(3):
                if i < len(top_neg):
                    row = top_neg.loc[i]
                    ticker_html = ticker_color_style(row["ticker"], {"vader_pos": 0, "vader_neg": 0, "finbert_pos": 0, "finbert_neg": 0})
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'>{ticker_html}</td><td style='padding:6px'>{int(row['neg_total'])}</td></tr>"
                else:
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'></td><td style='padding:6px'></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

        # -----------------------
        # Detailed headlines table with TICKER FILTER DROPDOWN
        # -----------------------
        st.subheader("All headlines (detailed) — grouped by ticker, newest first")

        # Collect all detail rows
        detail_rows = []
        for r in results:
            tkr = r["ticker"]
            for h in r.get("headlines", []):
                detail_rows.append({
                    "ticker": tkr,
                    "timestamp_raw": h.get("timestamp", "") or "",
                    "title": h.get("title", ""),
                    "link": h.get("link", ""),
                    "vader": h.get("vader", "n/a"),
                    "finbert": h.get("finbert", "n/a")
                })

        # Create ticker filter dropdown
        all_tickers = sorted(list(set([r["ticker"] for r in detail_rows])))
        ticker_options = ["All Tickers"] + all_tickers
        
        selected_ticker = st.selectbox(
            "Filter by ticker:",
            options=ticker_options,
            index=0,
            key="ticker_filter"
        )

        # Filter rows based on selection
        if selected_ticker != "All Tickers":
            filtered_rows = [r for r in detail_rows if r["ticker"] == selected_ticker]
        else:
            filtered_rows = detail_rows

        # Parse and normalize timestamp into datetime if possible
        def parse_ts(ts_str):
            if not ts_str:
                return None
            try:
                dt = date_parser.parse(ts_str, fuzzy=True)
                return dt
            except Exception:
                return None

        # Group by ticker alphabetically, and within each ticker sort by timestamp descending (with parsed times first)
        grouped = {}
        for r in filtered_rows:
            grouped.setdefault(r["ticker"], []).append(r)

        grouped_sorted_rows = []
        for ticker in sorted(grouped.keys()):
            items = grouped[ticker]
            # attach parsed dt
            for it in items:
                it["_dt"] = parse_ts(it["timestamp_raw"])
            # split lists
            with_dt = [x for x in items if x.get("_dt") is not None]
            without_dt = [x for x in items if x.get("_dt") is None]
            try:
                with_dt_sorted = sorted(with_dt, key=lambda x: x["_dt"], reverse=True)
            except Exception:
                with_dt_sorted = with_dt
            grouped_sorted_rows.extend(with_dt_sorted + without_dt)

        # Display count
        st.caption(f"Showing {len(grouped_sorted_rows)} headlines{' for ' + selected_ticker if selected_ticker != 'All Tickers' else ''}")

        # Build HTML table for details; use emoji balls (no text)
        def ball_for(label):
            if label == "positive":
                return "🟢"
            if label == "negative":
                return "🔴"
            if label == "neutral":
                return "🟡"
            return ""

        detail_header = ("<tr>"
                         "<th style='padding:8px;text-align:left'>Ticker</th>"
                         "<th style='padding:8px;text-align:left'>Timestamp</th>"
                         "<th style='padding:8px;text-align:left'>Source</th>"
                         "<th style='padding:8px;text-align:left'>Headline</th>"
                         "<th style='padding:8px;text-align:center'>VADER</th>"
                         "<th style='padding:8px;text-align:center'>FINBERT</th>"
                         "</tr>")
        detail_rows_html = []
        for it in grouped_sorted_rows:
            ticker = html.escape(it["ticker"])
            ts_formatted = ""
            if it.get("_dt") is not None:
                ts_formatted = it["_dt"].strftime("%Y-%m-%d %H:%M")
            else:
                ts_formatted = html.escape(it.get("timestamp_raw", ""))
            link = it.get("link") or ""
            source_host = ""
            try:
                if link:
                    parsed = urllib.parse.urlparse(link)
                    source_host = parsed.netloc.replace("www.", "")
            except Exception:
                source_host = ""
            source_host = html.escape(source_host)
            title_escaped = html.escape(it.get("title", ""))
            if link:
                safe_link = html.escape(link, quote=True)
                headline_cell = f"<a href='{safe_link}' target='_blank' rel='noopener noreferrer'>{title_escaped}</a>"
            else:
                headline_cell = title_escaped
            vader_ball = ball_for(it.get("vader", ""))
            finbert_ball = ball_for(it.get("finbert", ""))
            detail_rows_html.append(
                f"<tr>"
                f"<td style='padding:6px'>{ticker}</td>"
                f"<td style='padding:6px'>{ts_formatted}</td>"
                f"<td style='padding:6px'>{source_host}</td>"
                f"<td style='padding:6px'>{headline_cell}</td>"
                f"<td style='padding:6px;text-align:center'>{vader_ball}</td>"
                f"<td style='padding:6px;text-align:center'>{finbert_ball}</td>"
                f"</tr>"
            )

        detail_table = f"<table style='border-collapse: collapse; width:100%'>{detail_header}{''.join(detail_rows_html)}</table>"
        st.markdown(detail_table, unsafe_allow_html=True)

        t_elapsed = time.time() - t0
        st.success(f"Done — processed {len(results)} tickers in {t_elapsed:.1f} s.")

if __name__ == "__main__":
    main()
