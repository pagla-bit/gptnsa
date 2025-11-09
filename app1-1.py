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
# Config
# -----------------------
st.set_page_config(page_title="Ticker News Sentiment", layout="wide")
REQUEST_TIMEOUT = 10  # seconds
MAX_TICKERS_LIMIT = 300

# -----------------------
# Cached model loaders
# -----------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert_pipeline():
    # FinBERT-like finance tone classifier
    # This will download the model on first run; consider swapping to a hosted API if needed
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", device=-1)

# -----------------------
# Networking helper
# -----------------------
def fetch_url_text(url: str, params: dict = None, headers: dict = None) -> str:
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text

def fetch_url_content(url: str, params: dict = None, headers: dict = None) -> bytes:
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content

# -----------------------
# Finviz Screener parsing (follow pagination)
# -----------------------
def extract_tickers_from_finviz_screener(screener_url: str, max_total: int = 200) -> List[str]:
    """
    Follows pagination and extracts all tickers from the provided Finviz screener URL.
    Pagination parameter 'r' moves 1,21,41,...
    """
    try:
        parsed = urllib.parse.urlparse(screener_url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query = urllib.parse.parse_qs(parsed.query)
        # keep other params
        base_params = {k: v[0] for k, v in query.items() if k != "r"}
    except Exception:
        return []

    tickers = []
    seen = set()
    start = 1
    page_size = 20
    while True:
        params = dict(base_params)
        params["r"] = str(start)
        try:
            text = fetch_url_text(base, params=params, headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            break
        soup = BeautifulSoup(text, "html.parser")
        # anchors with screener tickers use class 'screener-link-primary' or href 'quote.ashx?t='
        anchors = soup.find_all("a", href=True)
        page_tickers = []
        for a in anchors:
            href = a["href"]
            if "quote.ashx?t=" in href:
                # sometimes anchor text is ticker
                t = a.get_text(strip=True)
                if t and t not in seen:
                    seen.add(t)
                    page_tickers.append(t)
                    tickers.append(t)
                    if len(tickers) >= max_total:
                        break
        if not page_tickers:
            break
        # next page
        start += page_size
        if len(tickers) >= max_total:
            break
        # safety
        if start > 5000:
            break
    return tickers

# -----------------------
# Parsers that return list of dicts {title, link, source, timestamp}
# -----------------------
def parse_finviz(ticker: str, max_items: int = 10) -> List[Dict]:
    url = "https://finviz.com/quote.ashx"
    params = {"t": ticker}
    try:
        text = fetch_url_text(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
    except Exception:
        return []
    soup = BeautifulSoup(text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    results = []
    if not news_table:
        # fallback: find anchors and take headline-like texts
        for a in soup.find_all("a"):
            txt = a.get_text(strip=True)
            if txt and len(txt) > 10:
                href = a.get("href") or ""
                link = href if href.startswith("http") else ("https://finviz.com/" + href.lstrip("/")) if href else ""
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
            timestamp = tds[0].get_text(strip=True)
            a = tds[1].find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href") or ""
                if href.startswith("http"):
                    link = href
                elif href:
                    link = "https://finviz.com/" + href.lstrip("/")
        else:
            a = tr.find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href") or ""
                if href.startswith("http"):
                    link = href
                elif href:
                    link = "https://finviz.com/" + href.lstrip("/")
        if title:
            results.append({"title": title, "link": link, "source": "finviz", "timestamp": timestamp})
            if len(results) >= max_items:
                break
    return results[:max_items]

def parse_google_news(ticker: str, max_items: int = 10) -> List[Dict]:
    # Use Google News RSS to avoid html parsing brittleness
    query = urllib.parse.quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={query}"
    try:
        content = fetch_url_content(url)
    except Exception:
        return []
    try:
        soup = BeautifulSoup(content, "xml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")
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
# Cached fetch per ticker
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_cached_all_sites_for_ticker(ticker: str, sites_tuple: Tuple[str, ...], max_items: int = 10) -> List[Dict]:
    sites = list(sites_tuple)
    all_items = []
    for s in sites:
        if s == "finviz":
            all_items.extend(parse_finviz(ticker, max_items))
        elif s == "google":
            all_items.extend(parse_google_news(ticker, max_items))
    # dedupe by title while preserving order
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
# Process ticker function (fetch + analyze)
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
# UI: Sidebar (Option A layout)
# -----------------------
st.title("Stock Ticker News Sentiment (VADER + FinBERT)")

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input method", ("Manual (text)", "Upload Excel (single-column)", "Finviz Screener URL"))
    st.write("Sentiment analyzers")
    col1, col2 = st.columns(2)
    with col1:
        run_vader = st.checkbox("VADER (fast)", value=True)
    with col2:
        run_finbert = st.checkbox("FinBERT (slower)", value=True)

    st.write("---")
    sites_multiselect = st.multiselect("News sources", options=["finviz", "google"], default=["finviz", "google"])
    max_news = st.number_input("Max headlines per site", min_value=1, max_value=20, value=3)
    max_workers = st.slider("Concurrent workers (fetching)", 2, 20, 8)
    st.write("---")
    st.write("Input tickers / screener")
    tickers = []
    screener_url = ""
    uploaded_file = None

    if input_mode == "Manual (text)":
        raw = st.text_area("Tickers (comma, newline or space separated)", placeholder="AAPL, MSFT, TSLA")
        if raw:
            candidates = [t.strip().upper() for t in raw.replace(",", " ").split()]
            tickers = [t for t in candidates if t]
    elif input_mode == "Upload Excel (single-column)":
        uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            try:
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
    else:
        screener_url = st.text_input("Paste Finviz screener URL (will fetch all pages):", placeholder="https://finviz.com/screener.ashx?v=111&f=...")
        if screener_url:
            with st.spinner("Fetching tickers from screener (may take a few seconds)..."):
                try:
                    tickers = extract_tickers_from_finviz_screener(screener_url, max_total=MAX_TICKERS_LIMIT)
                    if not tickers:
                        st.warning("No tickers found on the provided screener URL.")
                except Exception as e:
                    st.error(f"Failed to extract tickers from screener URL: {e}")

    if len(tickers) > MAX_TICKERS_LIMIT:
        st.warning(f"Too many tickers provided; limiting to first {MAX_TICKERS_LIMIT}.")
        tickers = tickers[:MAX_TICKERS_LIMIT]

    run_button = st.button("Run analysis")

# quick preview in main area
if tickers:
    st.markdown(f"**Tickers to analyze ({len(tickers)}):** {', '.join(tickers[:40])}")

# -----------------------
# When analyze button pressed
# -----------------------
if run_button:
    if not tickers:
        st.error("No tickers provided.")
    elif not sites_multiselect:
        st.error("Select at least one news source.")
    elif not (run_vader or run_finbert):
        st.error("Select at least one sentiment analyzer.")
    else:
        t0 = time.time()
        st.info("Fetching headlines and analyzing... (FinBERT may be slower)")

        # pre-load model resources (cached)
        if run_vader:
            _ = load_vader()
        if run_finbert:
            _ = load_finbert_pipeline()

        # concurrent processing of tickers
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
        # Summary table (grouped Vader & FinBERT counts, no percentages)
        # -----------------------
        df_res = pd.DataFrame(results).set_index("ticker")
        display_df = df_res[[
            "vader_pos", "vader_neg", "vader_neu",
            "finbert_pos", "finbert_neg", "finbert_neu",
        ]]

        st.subheader("Sentiment counts summary")
        def ticker_color_html(ticker, row):
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
            ticker_html = ticker_color_html(row["ticker"], row)
            vader_cell = f"🟢 {row['vader_pos']} &nbsp;&nbsp; 🔴 {row['vader_neg']} &nbsp;&nbsp; 🟡 {row['vader_neu']}"
            finbert_cell = f"🟢 {row['finbert_pos']} &nbsp;&nbsp; 🔴 {row['finbert_neg']} &nbsp;&nbsp; 🟡 {row['finbert_neu']}"
            rows_html.append(
                f"<tr><td style='padding:6px'>{ticker_html}</td>"
                f"<td style='padding:6px;text-align:center'>{vader_cell}</td>"
                f"<td style='padding:6px;text-align:center'>{finbert_cell}</td></tr>"
            )
        summary_table = f"<table style='border-collapse: collapse; width:100%'>{header}{''.join(rows_html)}</table>"
        st.markdown(summary_table, unsafe_allow_html=True)

        # -----------------------
        # Top Movers (side-by-side top 3 positive and top 3 negative)
        # -----------------------
        st.subheader("Top Movers (by total counts)")
        movers = []
        for r in results:
            pos_total = r.get("vader_pos", 0) + r.get("finbert_pos", 0)
            neg_total = r.get("vader_neg", 0) + r.get("finbert_neg", 0)
            movers.append({"ticker": r["ticker"], "pos_total": pos_total, "neg_total": neg_total})
        movers_df = pd.DataFrame(movers)
        if movers_df.empty:
            top_pos = pd.DataFrame()
            top_neg = pd.DataFrame()
        else:
            top_pos = movers_df.sort_values(by="pos_total", ascending=False).head(3).reset_index(drop=True)
            top_neg = movers_df.sort_values(by="neg_total", ascending=False).head(3).reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 3 Positive**")
            table_html = "<table style='border-collapse: collapse; width:100%'><tr><th>Rank</th><th>Ticker</th><th>Total Positive</th></tr>"
            for i in range(3):
                if i < len(top_pos):
                    row = top_pos.loc[i]
                    ticker_html = ticker_color_html(row["ticker"], {"vader_pos": 0, "vader_neg": 0, "finbert_pos": 0, "finbert_neg": 0})
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
                    ticker_html = ticker_color_html(row["ticker"], {"vader_pos": 0, "vader_neg": 0, "finbert_pos": 0, "finbert_neg": 0})
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'>{ticker_html}</td><td style='padding:6px'>{int(row['neg_total'])}</td></tr>"
                else:
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'></td><td style='padding:6px'></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

        # -----------------------
        # Detailed headlines table with ticker dropdown filter (above the table)
        # -----------------------
        st.subheader("All headlines (detailed) — grouped by ticker, newest first")
        # build detail rows
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
                    "finbert": h.get("finbert", "n/a"),
                    "source": h.get("source", "")
                })

        # parse timestamps into datetime where possible
        def parse_ts_or_none(ts):
            if not ts:
                return None
            try:
                dt = date_parser.parse(ts, fuzzy=True)
                return dt
            except Exception:
                return None

        # attach parsed _dt
        for it in detail_rows:
            it["_dt"] = parse_ts_or_none(it["timestamp_raw"])

        # ticker filter dropdown (searchable)
        ticker_options = ["All"] + sorted({r["ticker"] for r in detail_rows})
        selected_ticker = st.selectbox("🔍 Select Ticker to View News", options=ticker_options, index=0)

        # filter rows
        if selected_ticker != "All":
            rows_to_show = [r for r in detail_rows if r["ticker"] == selected_ticker]
        else:
            rows_to_show = detail_rows

        # group by ticker alphabetically and sort within group by _dt desc (with parsed dt first)
        grouped = {}
        for r in rows_to_show:
            grouped.setdefault(r["ticker"], []).append(r)

        grouped_sorted_rows = []
        for ticker in sorted(grouped.keys()):
            items = grouped[ticker]
            with_dt = [x for x in items if x.get("_dt") is not None]
            without_dt = [x for x in items if x.get("_dt") is None]
            try:
                with_dt_sorted = sorted(with_dt, key=lambda x: x["_dt"], reverse=True)
            except Exception:
                with_dt_sorted = with_dt
            grouped_sorted_rows.extend(with_dt_sorted + without_dt)

        # render detail HTML table
        def ball(label):
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
                         "<th style='padding:8px;text-align:center'>FinBERT</th>"
                         "</tr>")
        detail_rows_html = []
        for it in grouped_sorted_rows:
            ticker_html = html.escape(it["ticker"])
            tsf = ""
            if it.get("_dt") is not None:
                tsf = it["_dt"].strftime("%Y-%m-%d %H:%M")
            else:
                tsf = html.escape(it.get("timestamp_raw", ""))
            link = it.get("link") or ""
            source_host = ""
            try:
                if link:
                    parsed = urllib.parse.urlparse(link)
                    source_host = parsed.netloc.replace("www.", "")
            except Exception:
                source_host = it.get("source", "")
            source_host = html.escape(source_host)
            title = html.escape(it.get("title", ""))
            if link:
                safe_link = html.escape(link, quote=True)
                headline_cell = f"<a href='{safe_link}' target='_blank' rel='noopener noreferrer'>{title}</a>"
            else:
                headline_cell = title
            vball = ball(it.get("vader", ""))
            fball = ball(it.get("finbert", ""))
            detail_rows_html.append(
                f"<tr>"
                f"<td style='padding:6px'>{ticker_html}</td>"
                f"<td style='padding:6px'>{tsf}</td>"
                f"<td style='padding:6px'>{source_host}</td>"
                f"<td style='padding:6px'>{headline_cell}</td>"
                f"<td style='padding:6px;text-align:center'>{vball}</td>"
                f"<td style='padding:6px;text-align:center'>{fball}</td>"
                f"</tr>"
            )

        detail_table = f"<table style='border-collapse: collapse; width:100%'>{detail_header}{''.join(detail_rows_html)}</table>"
        st.markdown(detail_table, unsafe_allow_html=True)

        t_elapsed = time.time() - t0
        st.success(f"Done — processed {len(results)} tickers in {t_elapsed:.1f} s.")
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
# Config
# -----------------------
st.set_page_config(page_title="Ticker News Sentiment", layout="wide")
REQUEST_TIMEOUT = 10  # seconds
MAX_TICKERS_LIMIT = 300

# -----------------------
# Cached model loaders
# -----------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert_pipeline():
    # FinBERT-like finance tone classifier
    # This will download the model on first run; consider swapping to a hosted API if needed
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", device=-1)

# -----------------------
# Networking helper
# -----------------------
def fetch_url_text(url: str, params: dict = None, headers: dict = None) -> str:
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text

def fetch_url_content(url: str, params: dict = None, headers: dict = None) -> bytes:
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content

# -----------------------
# Finviz Screener parsing (follow pagination)
# -----------------------
def extract_tickers_from_finviz_screener(screener_url: str, max_total: int = 200) -> List[str]:
    """
    Follows pagination and extracts all tickers from the provided Finviz screener URL.
    Pagination parameter 'r' moves 1,21,41,...
    """
    try:
        parsed = urllib.parse.urlparse(screener_url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query = urllib.parse.parse_qs(parsed.query)
        # keep other params
        base_params = {k: v[0] for k, v in query.items() if k != "r"}
    except Exception:
        return []

    tickers = []
    seen = set()
    start = 1
    page_size = 20
    while True:
        params = dict(base_params)
        params["r"] = str(start)
        try:
            text = fetch_url_text(base, params=params, headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            break
        soup = BeautifulSoup(text, "html.parser")
        # anchors with screener tickers use class 'screener-link-primary' or href 'quote.ashx?t='
        anchors = soup.find_all("a", href=True)
        page_tickers = []
        for a in anchors:
            href = a["href"]
            if "quote.ashx?t=" in href:
                # sometimes anchor text is ticker
                t = a.get_text(strip=True)
                if t and t not in seen:
                    seen.add(t)
                    page_tickers.append(t)
                    tickers.append(t)
                    if len(tickers) >= max_total:
                        break
        if not page_tickers:
            break
        # next page
        start += page_size
        if len(tickers) >= max_total:
            break
        # safety
        if start > 5000:
            break
    return tickers

# -----------------------
# Parsers that return list of dicts {title, link, source, timestamp}
# -----------------------
def parse_finviz(ticker: str, max_items: int = 10) -> List[Dict]:
    url = "https://finviz.com/quote.ashx"
    params = {"t": ticker}
    try:
        text = fetch_url_text(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
    except Exception:
        return []
    soup = BeautifulSoup(text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    results = []
    if not news_table:
        # fallback: find anchors and take headline-like texts
        for a in soup.find_all("a"):
            txt = a.get_text(strip=True)
            if txt and len(txt) > 10:
                href = a.get("href") or ""
                link = href if href.startswith("http") else ("https://finviz.com/" + href.lstrip("/")) if href else ""
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
            timestamp = tds[0].get_text(strip=True)
            a = tds[1].find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href") or ""
                if href.startswith("http"):
                    link = href
                elif href:
                    link = "https://finviz.com/" + href.lstrip("/")
        else:
            a = tr.find("a")
            if a:
                title = a.get_text(strip=True)
                href = a.get("href") or ""
                if href.startswith("http"):
                    link = href
                elif href:
                    link = "https://finviz.com/" + href.lstrip("/")
        if title:
            results.append({"title": title, "link": link, "source": "finviz", "timestamp": timestamp})
            if len(results) >= max_items:
                break
    return results[:max_items]

def parse_google_news(ticker: str, max_items: int = 10) -> List[Dict]:
    # Use Google News RSS to avoid html parsing brittleness
    query = urllib.parse.quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={query}"
    try:
        content = fetch_url_content(url)
    except Exception:
        return []
    try:
        soup = BeautifulSoup(content, "xml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")
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
# Cached fetch per ticker
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_cached_all_sites_for_ticker(ticker: str, sites_tuple: Tuple[str, ...], max_items: int = 10) -> List[Dict]:
    sites = list(sites_tuple)
    all_items = []
    for s in sites:
        if s == "finviz":
            all_items.extend(parse_finviz(ticker, max_items))
        elif s == "google":
            all_items.extend(parse_google_news(ticker, max_items))
    # dedupe by title while preserving order
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
# Process ticker function (fetch + analyze)
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
# UI: Sidebar (Option A layout)
# -----------------------
st.title("Stock Ticker News Sentiment (VADER + FinBERT)")

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input method", ("Manual (text)", "Upload Excel (single-column)", "Finviz Screener URL"))
    st.write("Sentiment analyzers")
    col1, col2 = st.columns(2)
    with col1:
        run_vader = st.checkbox("VADER (fast)", value=True)
    with col2:
        run_finbert = st.checkbox("FinBERT (slower)", value=True)

    st.write("---")
    sites_multiselect = st.multiselect("News sources", options=["finviz", "google"], default=["finviz", "google"])
    max_news = st.number_input("Max headlines per site", min_value=1, max_value=20, value=3)
    max_workers = st.slider("Concurrent workers (fetching)", 2, 20, 8)
    st.write("---")
    st.write("Input tickers / screener")
    tickers = []
    screener_url = ""
    uploaded_file = None

    if input_mode == "Manual (text)":
        raw = st.text_area("Tickers (comma, newline or space separated)", placeholder="AAPL, MSFT, TSLA")
        if raw:
            candidates = [t.strip().upper() for t in raw.replace(",", " ").split()]
            tickers = [t for t in candidates if t]
    elif input_mode == "Upload Excel (single-column)":
        uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            try:
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
    else:
        screener_url = st.text_input("Paste Finviz screener URL (will fetch all pages):", placeholder="https://finviz.com/screener.ashx?v=111&f=...")
        if screener_url:
            with st.spinner("Fetching tickers from screener (may take a few seconds)..."):
                try:
                    tickers = extract_tickers_from_finviz_screener(screener_url, max_total=MAX_TICKERS_LIMIT)
                    if not tickers:
                        st.warning("No tickers found on the provided screener URL.")
                except Exception as e:
                    st.error(f"Failed to extract tickers from screener URL: {e}")

    if len(tickers) > MAX_TICKERS_LIMIT:
        st.warning(f"Too many tickers provided; limiting to first {MAX_TICKERS_LIMIT}.")
        tickers = tickers[:MAX_TICKERS_LIMIT]

    run_button = st.button("Run analysis")

# quick preview in main area
if tickers:
    st.markdown(f"**Tickers to analyze ({len(tickers)}):** {', '.join(tickers[:40])}")

# -----------------------
# When analyze button pressed
# -----------------------
if run_button:
    if not tickers:
        st.error("No tickers provided.")
    elif not sites_multiselect:
        st.error("Select at least one news source.")
    elif not (run_vader or run_finbert):
        st.error("Select at least one sentiment analyzer.")
    else:
        t0 = time.time()
        st.info("Fetching headlines and analyzing... (FinBERT may be slower)")

        # pre-load model resources (cached)
        if run_vader:
            _ = load_vader()
        if run_finbert:
            _ = load_finbert_pipeline()

        # concurrent processing of tickers
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
        # Summary table (grouped Vader & FinBERT counts, no percentages)
        # -----------------------
        df_res = pd.DataFrame(results).set_index("ticker")
        display_df = df_res[[
            "vader_pos", "vader_neg", "vader_neu",
            "finbert_pos", "finbert_neg", "finbert_neu",
        ]]

        st.subheader("Sentiment counts summary")
        def ticker_color_html(ticker, row):
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
            ticker_html = ticker_color_html(row["ticker"], row)
            vader_cell = f"🟢 {row['vader_pos']} &nbsp;&nbsp; 🔴 {row['vader_neg']} &nbsp;&nbsp; 🟡 {row['vader_neu']}"
            finbert_cell = f"🟢 {row['finbert_pos']} &nbsp;&nbsp; 🔴 {row['finbert_neg']} &nbsp;&nbsp; 🟡 {row['finbert_neu']}"
            rows_html.append(
                f"<tr><td style='padding:6px'>{ticker_html}</td>"
                f"<td style='padding:6px;text-align:center'>{vader_cell}</td>"
                f"<td style='padding:6px;text-align:center'>{finbert_cell}</td></tr>"
            )
        summary_table = f"<table style='border-collapse: collapse; width:100%'>{header}{''.join(rows_html)}</table>"
        st.markdown(summary_table, unsafe_allow_html=True)

        # -----------------------
        # Top Movers (side-by-side top 3 positive and top 3 negative)
        # -----------------------
        st.subheader("Top Movers (by total counts)")
        movers = []
        for r in results:
            pos_total = r.get("vader_pos", 0) + r.get("finbert_pos", 0)
            neg_total = r.get("vader_neg", 0) + r.get("finbert_neg", 0)
            movers.append({"ticker": r["ticker"], "pos_total": pos_total, "neg_total": neg_total})
        movers_df = pd.DataFrame(movers)
        if movers_df.empty:
            top_pos = pd.DataFrame()
            top_neg = pd.DataFrame()
        else:
            top_pos = movers_df.sort_values(by="pos_total", ascending=False).head(3).reset_index(drop=True)
            top_neg = movers_df.sort_values(by="neg_total", ascending=False).head(3).reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 3 Positive**")
            table_html = "<table style='border-collapse: collapse; width:100%'><tr><th>Rank</th><th>Ticker</th><th>Total Positive</th></tr>"
            for i in range(3):
                if i < len(top_pos):
                    row = top_pos.loc[i]
                    ticker_html = ticker_color_html(row["ticker"], {"vader_pos": 0, "vader_neg": 0, "finbert_pos": 0, "finbert_neg": 0})
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
                    ticker_html = ticker_color_html(row["ticker"], {"vader_pos": 0, "vader_neg": 0, "finbert_pos": 0, "finbert_neg": 0})
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'>{ticker_html}</td><td style='padding:6px'>{int(row['neg_total'])}</td></tr>"
                else:
                    table_html += f"<tr><td style='padding:6px'>{i+1}</td><td style='padding:6px'></td><td style='padding:6px'></td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

        # -----------------------
        # Detailed headlines table with ticker dropdown filter (above the table)
        # -----------------------
        st.subheader("All headlines (detailed) — grouped by ticker, newest first")
        # build detail rows
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
                    "finbert": h.get("finbert", "n/a"),
                    "source": h.get("source", "")
                })

        # parse timestamps into datetime where possible
        def parse_ts_or_none(ts):
            if not ts:
                return None
            try:
                dt = date_parser.parse(ts, fuzzy=True)
                return dt
            except Exception:
                return None

        # attach parsed _dt
        for it in detail_rows:
            it["_dt"] = parse_ts_or_none(it["timestamp_raw"])

        # ticker filter dropdown (searchable)
        ticker_options = ["All"] + sorted({r["ticker"] for r in detail_rows})
        selected_ticker = st.selectbox("🔍 Select Ticker to View News", options=ticker_options, index=0)

        # filter rows
        if selected_ticker != "All":
            rows_to_show = [r for r in detail_rows if r["ticker"] == selected_ticker]
        else:
            rows_to_show = detail_rows

        # group by ticker alphabetically and sort within group by _dt desc (with parsed dt first)
        grouped = {}
        for r in rows_to_show:
            grouped.setdefault(r["ticker"], []).append(r)

        grouped_sorted_rows = []
        for ticker in sorted(grouped.keys()):
            items = grouped[ticker]
            with_dt = [x for x in items if x.get("_dt") is not None]
            without_dt = [x for x in items if x.get("_dt") is None]
            try:
                with_dt_sorted = sorted(with_dt, key=lambda x: x["_dt"], reverse=True)
            except Exception:
                with_dt_sorted = with_dt
            grouped_sorted_rows.extend(with_dt_sorted + without_dt)

        # render detail HTML table
        def ball(label):
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
                         "<th style='padding:8px;text-align:center'>FinBERT</th>"
                         "</tr>")
        detail_rows_html = []
        for it in grouped_sorted_rows:
            ticker_html = html.escape(it["ticker"])
            tsf = ""
            if it.get("_dt") is not None:
                tsf = it["_dt"].strftime("%Y-%m-%d %H:%M")
            else:
                tsf = html.escape(it.get("timestamp_raw", ""))
            link = it.get("link") or ""
            source_host = ""
            try:
                if link:
                    parsed = urllib.parse.urlparse(link)
                    source_host = parsed.netloc.replace("www.", "")
            except Exception:
                source_host = it.get("source", "")
            source_host = html.escape(source_host)
            title = html.escape(it.get("title", ""))
            if link:
                safe_link = html.escape(link, quote=True)
                headline_cell = f"<a href='{safe_link}' target='_blank' rel='noopener noreferrer'>{title}</a>"
            else:
                headline_cell = title
            vball = ball(it.get("vader", ""))
            fball = ball(it.get("finbert", ""))
            detail_rows_html.append(
                f"<tr>"
                f"<td style='padding:6px'>{ticker_html}</td>"
                f"<td style='padding:6px'>{tsf}</td>"
                f"<td style='padding:6px'>{source_host}</td>"
                f"<td style='padding:6px'>{headline_cell}</td>"
                f"<td style='padding:6px;text-align:center'>{vball}</td>"
                f"<td style='padding:6px;text-align:center'>{fball}</td>"
                f"</tr>"
            )

        detail_table = f"<table style='border-collapse: collapse; width:100%'>{detail_header}{''.join(detail_rows_html)}</table>"
        st.markdown(detail_table, unsafe_allow_html=True)

        t_elapsed = time.time() - t0
        st.success(f"Done — processed {len(results)} tickers in {t_elapsed:.1f} s.")
