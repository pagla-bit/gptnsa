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
MAX_TICKERS = 40

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
# Parsers (return list of dicts: title, link, source, timestamp)
# -----------------------
def parse_finviz(ticker: str, max_items: int = 10) -> List[Dict]:
    """
    Parse Finviz news table. Use headers to avoid being blocked.
    """
    url = "https://finviz.com/quote.ashx"
    params = {"t": ticker}
    try:
        resp = fetch_url(url, headers={"User-Agent": "Mozilla/5.0"}, params=params)
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")

    # Finviz news table is often in a table with class 'fullview-news-outer'
    news_table = soup.find("table", class_="fullview-news-outer")
    results = []
    if not news_table:
        # fallback: try to locate anchors in the page
        for a in soup.find_all("a"):
            txt = a.get_text(strip=True)
            href = a.get("href")
            if txt and len(txt) > 10:
                link = href if href and href.startswith("http") else ("https://finviz.com/" + href.lstrip("/") if href else "")
                results.append({"title": txt, "link": link, "source": "finviz", "timestamp": ""})
                if len(results) >= max_items:
                    break
        return results[:max_items]

    # Finviz uses rows: <tr><td>date/time</td><td><a>headline</a></td></tr>
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
    Use Google News RSS for search query. Parses pubDate where available.
    """
    query = urllib.parse.quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={query}"
    try:
        resp = fetch_url(url)
    except Exception:
        return []
    # preferred xml parser (lxml). fallback to html parser if necessary.
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
    return results[:max_items]

# -----------------------
# High-level fetch per ticker (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_cached_all_sites_for_ticker(ticker: str, sites_tuple: Tuple[str, ...], max_items: int) -> List[Dict]:
    sites = list(sites_tuple)
    all_items = []
    for s in sites:
        if s == "finviz":
            all_items.extend(parse_finviz(ticker, max_items))
        elif s == "google":
            all_items.extend(parse_google_news(ticker, max_items))
    # dedupe by title preserving order
    seen = set()
    deduped = []
    for it in all_items:
        key = it.get("title", "")
        if key and key not in seen:
            deduped.append(it)
            seen.add(key)
    return deduped

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
    try:
        out = finbert_pipe(text[:512])
    except Exception:
        return "neutral"
    if not out:
        return "neutral"
    label = out[0].get("label", "").lower()
    if "pos" in label:
        return "positive"
    if "neg" in label:
        return "negative"
    return "neutral"

# -----------------------
# Process ticker: returns summary counts and detailed headlines
# -----------------------
def process_ticker(ticker: str, sites: List[str], run_vader: bool, run_finbert: bool, max_per_site: int) -> Dict:
    ticker = ticker.strip().upper()
    headlines = fetch_cached_all_sites_for_ticker(ticker, tuple(sites), int(max_per_site))
    vader = load_vader() if run_vader else None
    finbert_pipe = load_finbert_pipeline() if run_finbert else None

    vpos = vneg = vneu = 0
    fpos = fneg = fneu = 0
    detailed = []

    for h in headlines:
        title = h.get("title", "")
        link = h.get("link", "")
        source = h.get("source", "")
        timestamp = h.get("timestamp", "") or ""

        v_label = "n/a"
        f_label = "n/a"
        if run_vader and vader:
            v_label = analyze_vader(vader, title)
            if v_label == "positive":
                vpos += 1
            elif v_label == "negative":
                vneg += 1
            else:
                vneu += 1
        if run_finbert and finbert_pipe:
            f_label = analyze_finbert(finbert_pipe, title)
            if f_label == "positive":
                fpos += 1
            elif f_label == "negative":
                fneg += 1
            else:
                fneu += 1

        detailed.append({
            "title": title,
            "link": link,
            "source": source,
            "timestamp": timestamp,
            "vader": v_label,
            "finbert": f_label
        })

    return {
        "ticker": ticker,
        "vader_pos": vpos,
        "vader_neg": vneg,
        "vader_neu": vneu,
        "finbert_pos": fpos,
        "finbert_neg": fneg,
        "finbert_neu": fneu,
        "n_headlines": len(headlines),
        "headlines": detailed
    }

# -----------------------
# UI (sidebar & main)
# -----------------------
st.title("Stock Ticker News Sentiment (VADER + FinBERT)")
with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input method", ("Manual (text box)", "Upload Excel (single-column)"))
    run_vader = st.checkbox("VADER (fast)", value=True)
    run_finbert = st.checkbox("FinBERT (slower)", value=True)
    sites_multiselect = st.multiselect("News sources", ["finviz", "google"], default=["finviz", "google"])
    max_news = st.number_input("Max headlines per site", min_value=1, max_value=20, value=3)
    max_workers = st.slider("Concurrent workers", 2, 20, 8)
    st.write("---")
    st.write("Input tickers")
    tickers = []
    uploaded_file = None
    if input_mode == "Manual (text box)":
        raw = st.text_area("Tickers (comma, newline or space separated)", placeholder="AAPL, MSFT, TSLA")
        if raw:
            candidates = [t.strip().upper() for t in raw.replace(",", " ").split()]
            tickers = [t for t in candidates if t]
    else:
        uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if "ticker" in (c.lower() for c in df.columns):
                    col = next(c for c in df.columns if c.lower() == "ticker")
                    tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
                else:
                    # if single column, use that
                    if df.shape[1] == 1:
                        col = df.columns[0]
                        tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
                    else:
                        st.warning("Uploaded file has multiple columns and no 'ticker' column. Please upload a single-column file or include 'ticker' column named 'ticker'.")
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
    if len(tickers) > MAX_TICKERS:
        st.warning(f"Too many tickers provided; limiting to first {MAX_TICKERS}.")
        tickers = tickers[:MAX_TICKERS]

    run_button = st.button("Run analysis")

# preview tickers in main
if tickers:
    st.markdown(f"**Tickers to analyze ({len(tickers)}):** {', '.join(tickers[:40])}")

# When button pressed
if run_button:
    if not tickers:
        st.error("No tickers provided.")
    elif not sites_multiselect:
        st.error("Select at least one news source.")
    elif not (run_vader or run_finbert):
        st.error("Select at least one sentiment analyzer.")
    else:
        t0 = time.time()
        st.info("Fetching headlines and analyzing... (FinBERT may be slow)")

        # Pre-load models so caching works for threads
        if run_vader:
            _ = load_vader()
        if run_finbert:
            _ = load_finbert_pipeline()

        results = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total = len(tickers)
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(process_ticker, t, sites_multiselect, run_vader, run_finbert, int(max_news)): t for t in tickers
            }
            for fut in concurrent.futures.as_completed(future_to_ticker):
                tkr = future_to_ticker[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    st.warning(f"Error processing {tkr}: {e}")
                    res = {
                        "ticker": tkr,
                        "vader_pos": 0, "vader_neg": 0, "vader_neu": 0,
                        "finbert_pos": 0, "finbert_neg": 0, "finbert_neu": 0,
                        "n_headlines": 0,
                        "headlines": []
                    }
                results.append(res)
                completed += 1
                progress_bar.progress(completed / total)
                progress_text.text(f"Processed {completed}/{total} tickers")
        progress_bar.empty()
        progress_text.empty()

        # -----------------------
        # Build summary table (no percentages)
        # Column groups: VADER and FINBERT (each cell shows three emoji-balls with counts)
        # -----------------------
        df_res = pd.DataFrame(results).set_index("ticker")
        # We no longer show percentages per your confirmation.
        display_df = df_res[[
            "vader_pos", "vader_neg", "vader_neu",
            "finbert_pos", "finbert_neg", "finbert_neu",
        ]]

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

        # Build HTML table with grouped columns
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
        # Detailed headlines table: grouped by ticker and sorted newest -> oldest
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

        # Parse and normalize timestamp to YYYY-MM-DD HH:MM when possible; empty if not
        def parse_ts(ts_str):
            if not ts_str:
                return None
            try:
                dt = date_parser.parse(ts_str, fuzzy=True)
                # return both dt and formatted string
                return dt
            except Exception:
                return None

        # Group by ticker alphabetically, and within each ticker sort by timestamp descending (with parsed times first)
        grouped = {}
        for r in detail_rows:
            grouped.setdefault(r["ticker"], []).append(r)

        grouped_sorted_rows = []
        for ticker in sorted(grouped.keys()):
            items = grouped[ticker]
            # attach parsed dt safely
            for it in items:
                it["_dt"] = parse_ts(it.get("timestamp_raw", ""))

            # split into parsed and unparsed
            with_dt = [x for x in items if isinstance(x.get("_dt"), (pd.Timestamp, type(date_parser.parse('2020-01-01'))))]
            without_dt = [x for x in items if not isinstance(x.get("_dt"), (pd.Timestamp, type(date_parser.parse('2020-01-01'))))]

            # safely sort with_dt
            try:
                with_dt_sorted = sorted(with_dt, key=lambda x: x["_dt"], reverse=True)
            except Exception:
                with_dt_sorted = with_dt  # fallback to original order if anything weird happens

            grouped_sorted_rows.extend(with_dt_sorted + without_dt)


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
            # formatted timestamp if parsed, else raw (or empty)
            ts_formatted = ""
            if it.get("_dt") is not None:
                ts_formatted = it["_dt"].strftime("%Y-%m-%d %H:%M")
            else:
                ts_formatted = html.escape(it.get("timestamp_raw", ""))
            # try to detect source from link or leave blank (we have source not stored here; but headline link may include source)
            # To show source column we can examine the link domain
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
