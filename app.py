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
import re

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Ticker News Sentiment", layout="wide")
REQUEST_TIMEOUT = 10  # seconds
MAX_TICKERS = 30

# -----------------------
# Caching: heavy resources
# -----------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert_pipeline():
    # FinBERT / finance tone classifier (may download first time)
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", device=-1)

# -----------------------
# Networking helper
# -----------------------
def fetch_url(url: str, headers=None, params=None) -> requests.Response:
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp

# -----------------------
# Parsers (return dicts with title, link, source, timestamp)
# -----------------------
def parse_finviz(ticker: str, max_items: int = 10) -> List[Dict]:
    """
    Parse finviz news table. Returns list of {"title","link","source":"finviz","timestamp": "..."}.
    Finviz usually lists rows where the date/time may appear in the first td and the anchor in the second.
    """
    url = "https://finviz.com/quote.ashx"
    params = {"t": ticker}
    try:
        resp = fetch_url(url, params=params)
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    if not news_table:
        return []
    results = []
    # Finviz places news in rows: <tr><td>DATE/TIME</td><td><a>headline</a></td></tr>
    for tr in news_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        # sometimes date/time is in first td, headline anchor in second
        timestamp = ""
        title = ""
        link = ""
        if len(tds) == 2:
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
            # fallback: find first anchor
            a = tr.find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href")
                link = href if href and href.startswith("http") else ("https://finviz.com/" + href.lstrip("/") if href else "")
        if title:
            results.append({"title": title, "link": link, "source": "finviz", "timestamp": timestamp})
            if len(results) >= max_items:
                break
    return results[:max_items]

def parse_yahoo(ticker: str, max_items: int = 10) -> List[Dict]:
    """
    Scrape Yahoo Finance news listing for a ticker.
    Attempts to extract headline and time where available.
    """
    results = []
    url = f"https://finance.yahoo.com/quote/{urllib.parse.quote(ticker)}/news?p={urllib.parse.quote(ticker)}"
    try:
        resp = fetch_url(url)
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")

    # Attempt to find news card blocks
    # Yahoo structures change; try a few heuristics.
    items = soup.select("li.js-stream-content, div[data-test-locator='mega'] a")  # broad selector
    # If above didn't find good items, fall back to anchors
    if not items:
        anchors = soup.find_all("a")
        for a in anchors:
            txt = a.get_text(strip=True)
            href = a.get("href")
            if not txt or len(txt) < 10:
                continue
            if href and ("news" in href or href.startswith("/") or href.startswith("http")):
                if href.startswith("/"):
                    link = urllib.parse.urljoin("https://finance.yahoo.com", href)
                else:
                    link = href
                # attempt to find a timestamp near the anchor
                ts = ""
                parent = a.parent
                # look for time tag or span with 'time' or 'ago'
                time_tag = parent.find("time") if parent else None
                if time_tag:
                    ts = time_tag.get_text(strip=True)
                else:
                    # look for sibling spans
                    sib = a.find_next_sibling("span") or (parent.find("span") if parent else None)
                    if sib:
                        ts_text = sib.get_text(strip=True)
                        # quick filter to avoid nav labels
                        if re.search(r'\d', ts_text):
                            ts = ts_text
                results.append({"title": txt, "link": link, "source": "yahoo", "timestamp": ts})
                if len(results) >= max_items:
                    break
    else:
        for item in items:
            a = item.find("a")
            if not a:
                continue
            txt = a.get_text(strip=True)
            href = a.get("href")
            if not txt or len(txt) < 10:
                continue
            # link absolute
            if href and href.startswith("/"):
                link = urllib.parse.urljoin("https://finance.yahoo.com", href)
            else:
                link = href or ""
            # search for time within the item
            ts = ""
            t = item.find("time")
            if t:
                ts = t.get_text(strip=True)
            else:
                # sometimes there's a small tag or span with time info
                possible = item.find_all(["span", "small"])
                for p in possible:
                    txtp = p.get_text(strip=True)
                    if txtp and re.search(r'\d', txtp):
                        ts = txtp
                        break
            results.append({"title": txt, "link": link, "source": "yahoo", "timestamp": ts})
            if len(results) >= max_items:
                break

    # final fallback: parse RSS via Yahoo's news endpoint is not straightforward; use what we found
    return results[:max_items]

def parse_google_news(ticker: str, max_items: int = 10) -> List[Dict]:
    """
    Use Google News RSS for the search query.
    Returns link, title, and pubDate if present.
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
            ts = item.pubDate.get_text(strip=True)
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
        elif s == "yahoo":
            all_items.extend(parse_yahoo(ticker, max_items))
        elif s == "google":
            all_items.extend(parse_google_news(ticker, max_items))
    # Deduplicate by title while preserving order
    seen = set()
    deduped = []
    for item in all_items:
        key = item.get("title", "")
        if key and key not in seen:
            deduped.append(item)
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
        out = finbert_pipe(str(text)[:512])
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
# Process ticker: returns summary counts AND detailed headline-level rows
# -----------------------
def process_ticker(ticker: str, sites: List[str], run_vader: bool, run_finbert: bool, max_per_site: int) -> Dict:
    ticker = ticker.strip().upper()
    headlines = fetch_cached_all_sites_for_ticker(ticker, tuple(sites), int(max_per_site))
    vader = load_vader() if run_vader else None
    finbert = load_finbert_pipeline() if run_finbert else None

    vpos = vneg = vneu = 0
    fpos = fneg = fneu = 0
    detailed = []

    for h in headlines:
        title = h.get("title", "")
        link = h.get("link", "")
        source = h.get("source", "")
        timestamp = h.get("timestamp", "") or ""  # blank if not present

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
        if run_finbert and finbert:
            f_label = analyze_finbert(finbert, title)
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
    max_news = st.number_input("Max headlines per site", min_value=1, max_value=20, value=3)
    st.write("Performance options")
    max_workers = st.slider("Concurrent workers (fetching)", 2, 20, 8)
    st.write("Tip: reduce FinBERT runs or max headlines to speed up on Streamlit Free tier.")

# Input tickers
tickers = []
uploaded_file = None
if input_mode == "Manual (text box)":
    raw = st.text_area("Enter tickers (comma, space or newline separated). Max {} tickers.".format(MAX_TICKERS),
                       placeholder="AAPL, MSFT, TSLA")
    if raw:
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
            if "ticker" in (c.lower() for c in df.columns):
                col = next(c for c in df.columns if c.lower() == "ticker")
                tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
            else:
                if df.shape[1] == 1:
                    col = df.columns[0]
                    tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
                else:
                    st.warning("Uploaded file has multiple columns and no 'ticker' column. Please upload single-column file or include 'ticker' column.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

if len(tickers) > MAX_TICKERS:
    st.warning(f"You provided {len(tickers)} tickers; limiting to first {MAX_TICKERS}.")
    tickers = tickers[:MAX_TICKERS]

run_button = st.button("Run analysis")

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
        results = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total = len(tickers)
        completed = 0

        # Pre-load models
        if run_vader:
            _ = load_vader()
        if run_finbert:
            _ = load_finbert_pipeline()

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
        # Summary table with percentages
        # -----------------------
        df_res = pd.DataFrame(results).set_index("ticker")
        df_res = df_res[
            ["vader_pos", "vader_neg", "vader_neu",
             "finbert_pos", "finbert_neg", "finbert_neu",
             "n_headlines"]
        ]

        # add percentage columns (rounded to 1 decimal)
        def pct(part, total):
            try:
                return round((part / total) * 100.0, 1) if total and total > 0 else 0.0
            except Exception:
                return 0.0

        extra_cols = []
        for idx, row in df_res.reset_index().iterrows():
            n = row["n_headlines"]
            df_res.loc[row["ticker"], "vader_pos_%"] = pct(row["vader_pos"], n)
            df_res.loc[row["ticker"], "vader_neg_%"] = pct(row["vader_neg"], n)
            df_res.loc[row["ticker"], "finbert_pos_%"] = pct(row["finbert_pos"], n)
            df_res.loc[row["ticker"], "finbert_neg_%"] = pct(row["finbert_neg"], n)

        # Reorder columns for display
        display_cols = [
            "vader_pos", "vader_neg", "vader_neu",
            "finbert_pos", "finbert_neg", "finbert_neu",
            "n_headlines",
            "vader_pos_%", "vader_neg_%", "finbert_pos_%", "finbert_neg_%"
        ]
        df_display = df_res[display_cols]

        st.subheader("Sentiment counts summary")
        # Build HTML table so we color ticker name
        def make_summary_html(df):
            header_cells = "".join(f"<th style='padding:8px;text-align:left'>{html.escape(c)}</th>" for c in df.columns)
            header = f"<tr><th style='padding:8px;text-align:left'>Ticker</th>{header_cells}</tr>"
            rows = []
            for idx, r in df.reset_index().iterrows():
                pos = 0; neg = 0
                if run_vader:
                    pos += r["vader_pos"]; neg += r["vader_neg"]
                if run_finbert:
                    pos += r["finbert_pos"]; neg += r["finbert_neg"]
                if pos > neg:
                    color = "green"
                elif neg > pos:
                    color = "red"
                else:
                    color = "black"
                ticker_html = f"<span style='color:{color};font-weight:bold'>{html.escape(r['ticker'])}</span>"
                other_cells = "".join(f"<td style='padding:6px'>{html.escape(str(r[c]))}</td>" for c in df.columns)
                rows.append(f"<tr><td style='padding:6px'>{ticker_html}</td>{other_cells}</tr>")
            table = f"<table style='border-collapse: collapse;'>{header}{''.join(rows)}</table>"
            return table

        st.markdown(make_summary_html(df_display), unsafe_allow_html=True)

        # -----------------------
        # Detailed headlines table (with timestamp and colored sentiments)
        # -----------------------
        st.subheader("All headlines (detailed) — color-coded sentiments")
        def sentiment_color_label(label: str) -> str:
            if label == "positive":
                return "<span style='color:green;font-weight:bold'>positive</span>"
            elif label == "negative":
                return "<span style='color:red;font-weight:bold'>negative</span>"
            elif label == "neutral":
                return "<span style='color:orange;font-weight:bold'>neutral</span>"
            else:
                return "<span style='color:gray'>n/a</span>"

        detail_rows = []
        for r in results:
            tkr = r["ticker"]
            for h in r.get("headlines", []):
                title = h.get("title", "")
                link = h.get("link", "")
                source = h.get("source", "")
                timestamp = h.get("timestamp", "") or ""
                vader_label = h.get("vader", "n/a")
                fin_label = h.get("finbert", "n/a")
                detail_rows.append({
                    "ticker": tkr,
                    "source": source,
                    "title": title,
                    "link": link,
                    "timestamp": timestamp,
                    "vader": vader_label,
                    "finbert": fin_label
                })

                # -----------------------
        # Detailed headlines table (with timestamp and colored sentiments)
        # -----------------------
        st.subheader("All headlines (detailed) — grouped by ticker and sorted by time")

        def sentiment_color_label(label: str) -> str:
            if label == "positive":
                return "<span style='color:green;font-weight:bold'>positive</span>"
            elif label == "negative":
                return "<span style='color:red;font-weight:bold'>negative</span>"
            elif label == "neutral":
                return "<span style='color:orange;font-weight:bold'>neutral</span>"
            else:
                return "<span style='color:gray'>n/a</span>"

        # Build all detailed rows
        detail_rows = []
        for r in results:
            tkr = r["ticker"]
            for h in r.get("headlines", []):
                detail_rows.append({
                    "ticker": tkr,
                    "source": h.get("source", ""),
                    "title": h.get("title", ""),
                    "link": h.get("link", ""),
                    "timestamp": h.get("timestamp", "") or "",
                    "vader": h.get("vader", "n/a"),
                    "finbert": h.get("finbert", "n/a")
                })

        # Group by ticker, then sort each group by timestamp (descending if possible)
        def timestamp_key(ts: str):
            # Simple heuristic: prioritize filled timestamps, keep string order otherwise
            if not ts:
                return (0, "")
            return (1, ts)

        grouped_rows = []
        for ticker in sorted({r["ticker"] for r in detail_rows}):
            subset = [r for r in detail_rows if r["ticker"] == ticker]
            subset_sorted = sorted(subset, key=lambda x: timestamp_key(x["timestamp"]), reverse=True)
            grouped_rows.extend(subset_sorted)

        # Build HTML table for grouped detail view
        def make_detail_html(rows):
            header = (
                "<tr>"
                "<th style='padding:8px;text-align:left'>Ticker</th>"
                "<th style='padding:8px;text-align:left'>Timestamp</th>"
                "<th style='padding:8px;text-align:left'>Source</th>"
                "<th style='padding:8px;text-align:left'>Headline</th>"
                "<th style='padding:8px;text-align:left'>VADER</th>"
                "<th style='padding:8px;text-align:left'>FinBERT</th>"
                "</tr>"
            )
            html_rows = []
            for r in rows:
                ticker = html.escape(r["ticker"])
                ts = html.escape(r.get("timestamp", ""))
                source = html.escape(r.get("source", ""))
                title = html.escape(r.get("title", ""))
                link = r.get("link") or ""
                if link:
                    safe_link = html.escape(link, quote=True)
                    headline_cell = f"<a href='{safe_link}' target='_blank' rel='noopener noreferrer'>{title}</a>"
                else:
                    headline_cell = title
                vader_col = sentiment_color_label(r.get("vader", "n/a"))
                fin_col = sentiment_color_label(r.get("finbert", "n/a"))
                html_rows.append(
                    f"<tr>"
                    f"<td style='padding:6px'>{ticker}</td>"
                    f"<td style='padding:6px'>{ts}</td>"
                    f"<td style='padding:6px'>{source}</td>"
                    f"<td style='padding:6px'>{headline_cell}</td>"
                    f"<td style='padding:6px'>{vader_col}</td>"
                    f"<td style='padding:6px'>{fin_col}</td>"
                    f"</tr>"
                )
            table = f"<table style='border-collapse: collapse; width:100%'>{header}{''.join(html_rows)}</table>"
            return table

        st.markdown(make_detail_html(grouped_rows), unsafe_allow_html=True)
