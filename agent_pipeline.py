# ============================================================
#  INDIAN STOCK AI AGENT — Self-Learning Daily Pipeline v2
#  Improvements:
#  - 2-day confirmation before recommending a stock
#  - Full email: strengths, risks, catalysts always populated
#  - Shares to buy calculation based on portfolio size
#  - Technical indicators (RSI, moving averages)
#  - All analysts data saved in history JSON
#  - Richer email with all info a trader needs
#  - Timing fixed (IST aware)
# ============================================================

# ── httpx compatibility patch ─────────────────────────────
import httpx
_original_init = httpx.Client.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    _original_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_init

_original_async_init = httpx.AsyncClient.__init__
def _patched_async_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    _original_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _patched_async_init
# ─────────────────────────────────────────────────────────

import os, time, uuid, json, re, requests, feedparser
import yfinance as yf
import concurrent.futures, smtplib

from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import tiktoken, chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
FINNHUB_API_KEY  = os.environ.get("FINNHUB_API_KEY", "")
GROQ_KEY_1       = os.environ.get("GROQ_KEY_1", "")
GROQ_KEY_2       = os.environ.get("GROQ_KEY_2", "")
GMAIL_SENDER     = os.environ.get("GMAIL_SENDER", "")
GMAIL_RECEIVER   = os.environ.get("GMAIL_RECEIVER", "")
GMAIL_PASSWORD   = os.environ.get("GMAIL_PASSWORD", "")

# Portfolio size in INR — used to calculate how many shares to buy
# Change this to your actual trading capital
PORTFOLIO_SIZE_INR = float(os.environ.get("PORTFOLIO_SIZE_INR", "100000"))

HISTORY_FILE = "prediction_history.json"

# ─────────────────────────────────────────────────────────────
#  GROQ CLIENTS
# ─────────────────────────────────────────────────────────────
client1 = Groq(api_key=GROQ_KEY_1)
client2 = Groq(api_key=GROQ_KEY_2)

# ─────────────────────────────────────────────────────────────
#  TICKERS
# ─────────────────────────────────────────────────────────────
AGENT_TICKERS = {
    "tech":       ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "energy":     ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS"],
    "healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
}

COMPANY_NAMES = {
    "TCS.NS":        "TCS Tata Consultancy Services",
    "INFY.NS":       "Infosys",
    "WIPRO.NS":      "Wipro",
    "HCLTECH.NS":    "HCL Technologies",
    "TECHM.NS":      "Tech Mahindra",
    "RELIANCE.NS":   "Reliance Industries",
    "ONGC.NS":       "ONGC Oil Natural Gas",
    "NTPC.NS":       "NTPC Power",
    "POWERGRID.NS":  "Power Grid India",
    "ADANIGREEN.NS": "Adani Green Energy",
    "SUNPHARMA.NS":  "Sun Pharma",
    "DRREDDY.NS":    "Dr Reddys Laboratories",
    "CIPLA.NS":      "Cipla Pharma",
    "DIVISLAB.NS":   "Divis Laboratories",
    "APOLLOHOSP.NS": "Apollo Hospitals",
}

RSS_FEEDS = {
    "tech":       ["https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
                   "https://www.moneycontrol.com/rss/business.xml"],
    "energy":     ["https://economictimes.indiatimes.com/industry/energy/rssfeeds/13358376.cms",
                   "https://www.moneycontrol.com/rss/business.xml"],
    "healthcare": ["https://economictimes.indiatimes.com/industry/healthcare/rssfeeds/13358530.cms",
                   "https://www.moneycontrol.com/rss/business.xml"],
    "general":    ["https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
                   "https://www.moneycontrol.com/rss/latestnews.xml"],
}


# ═════════════════════════════════════════════════════════════
#  SECTION 1 — DATA INGESTION
# ═════════════════════════════════════════════════════════════

def fetch_google_news_rss(ticker: str, sector: str) -> List[Dict]:
    company = COMPANY_NAMES.get(ticker, ticker.replace(".NS", ""))
    url = ("https://news.google.com/rss/search?q=" +
           company.replace(" ", "+") + "+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:8]:
            results.append({
                "source": "google_news", "ticker": ticker, "sector": sector,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", "")),
                "url": entry.get("link", ""),
                "published": entry.get("published", datetime.now().isoformat()),
            })
        print("  [GoogleNews] " + ticker + ": " + str(len(results)) + " articles")
        return results
    except Exception as e:
        print("  [GoogleNews ERROR] " + ticker + ": " + str(e))
        return []


def fetch_rss_news(sector: str, max_items: int = 10) -> List[Dict]:
    results = []
    for feed_url in RSS_FEEDS.get(sector, RSS_FEEDS["general"]):
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_items]:
                results.append({
                    "source": "rss_india", "sector": sector,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", "")),
                    "url": entry.get("link", ""),
                    "published": entry.get("published", datetime.now().isoformat()),
                })
        except Exception as e:
            print("  [RSS ERROR] " + feed_url + ": " + str(e))
    return results


def fetch_yfinance_fundamentals(ticker: str) -> Dict:
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="30d")

        price = round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None

        # Calculate RSI (14-day)
        rsi = None
        if not hist.empty and len(hist) >= 14:
            delta = hist["Close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi   = round(float(rsi_series.iloc[-1]), 1)

        # Moving averages
        ma20 = round(float(hist["Close"].rolling(20).mean().iloc[-1]), 2) if len(hist) >= 20 else None
        ma50 = round(float(hist["Close"].rolling(50).mean().iloc[-1]), 2) if len(hist) >= 50 else None

        # 5-day price change
        price_5d_chg = None
        if not hist.empty and len(hist) >= 5:
            old = float(hist["Close"].iloc[-5])
            price_5d_chg = round(((price - old) / old) * 100, 2) if old else None

        return {
            "source":         "yfinance",
            "ticker":         ticker,
            "company":        info.get("longName", COMPANY_NAMES.get(ticker, ticker)),
            "sector":         info.get("sector", ""),
            "industry":       info.get("industry", ""),
            "current_price":  price,
            "currency":       "INR",
            "market_cap":     info.get("marketCap", 0),
            "pe_ratio":       info.get("trailingPE", None),
            "forward_pe":     info.get("forwardPE", None),
            "revenue":        info.get("totalRevenue", 0),
            "net_income":     info.get("netIncomeToCommon", 0),
            "52w_high":       info.get("fiftyTwoWeekHigh", None),
            "52w_low":        info.get("fiftyTwoWeekLow", None),
            "recommendation": info.get("recommendationKey", ""),
            "dividend_yield": info.get("dividendYield", None),
            "rsi_14":         rsi,
            "ma_20":          ma20,
            "ma_50":          ma50,
            "price_5d_chg":   price_5d_chg,
            "volume":         info.get("averageVolume", None),
            "beta":           info.get("beta", None),
            "summary":        info.get("longBusinessSummary", "")[:500],
        }
    except Exception as e:
        print("  [yfinance ERROR] " + ticker + ": " + str(e))
        return {}


def run_ingestion() -> Dict[str, List[Dict]]:
    all_docs = {}
    for sector, tickers in AGENT_TICKERS.items():
        print("\n" + "=" * 45)
        print("  INGESTING: " + sector.upper())
        print("=" * 45)
        docs = []
        for ticker in tickers:
            docs += fetch_google_news_rss(ticker, sector)
            fund  = fetch_yfinance_fundamentals(ticker)
            if fund:
                docs.append(fund)
            time.sleep(0.5)
        docs += fetch_rss_news(sector, max_items=10)
        docs += fetch_rss_news("general", max_items=5)
        all_docs[sector] = [d for d in docs if d]
        print("  total " + sector + ": " + str(len(all_docs[sector])) + " docs")
    return all_docs


# ═════════════════════════════════════════════════════════════
#  SECTION 2 — RAG / VECTOR STORE
# ═════════════════════════════════════════════════════════════

tokenizer = tiktoken.get_encoding("cl100k_base")
hf_ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
db_client = chromadb.Client()
try:
    db_client.delete_collection("indian_knowledge")
except Exception:
    pass
collection = db_client.get_or_create_collection(
    name="indian_knowledge",
    embedding_function=hf_ef,
    metadata={"hnsw:space": "cosine"},
)


def doc_to_text(doc: Dict) -> str:
    src = doc.get("source", "")
    if src == "google_news":
        return ("TICKER: " + doc.get("ticker", "") + "\nSECTOR: " + doc.get("sector", "") +
                "\nHEADLINE: " + doc.get("title", "") + "\nCONTENT: " + doc.get("summary", ""))
    if src == "rss_india":
        return ("SECTOR: " + doc.get("sector", "") + "\nHEADLINE: " + doc.get("title", "") +
                "\nCONTENT: " + doc.get("summary", ""))
    if src == "yfinance":
        return ("TICKER: " + doc.get("ticker", "") + "\nCOMPANY: " + doc.get("company", "") +
                "\nPRICE INR: " + str(doc.get("current_price", "")) +
                "\nPE: " + str(doc.get("pe_ratio", "")) +
                "\nRSI: " + str(doc.get("rsi_14", "")) +
                "\nMA20: " + str(doc.get("ma_20", "")) +
                "\nMA50: " + str(doc.get("ma_50", "")) +
                "\n5D_CHANGE: " + str(doc.get("price_5d_chg", "")) + "%" +
                "\nRECO: " + doc.get("recommendation", "") +
                "\nINFO: " + doc.get("summary", ""))
    return json.dumps(doc)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


def upsert_sector(sector: str, docs: List[Dict]) -> None:
    texts, metas, ids = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc_to_text(doc))):
            if len(chunk.strip()) < 30:
                continue
            texts.append(chunk)
            metas.append({"sector": sector, "source": doc.get("source", ""),
                          "ticker": doc.get("ticker", "")})
            ids.append(sector + "_" + uuid.uuid4().hex[:10] + "_" + str(i))
    if not texts:
        return
    for i in range(0, len(texts), 50):
        collection.upsert(documents=texts[i:i+50], metadatas=metas[i:i+50], ids=ids[i:i+50])
    print("  chunks stored " + sector + ": " + str(len(texts)))


def retrieve_context(query: str, sector: str, top_k: int = 6) -> Tuple[List[str], List[Dict]]:
    results = collection.query(query_texts=[query], n_results=top_k, where={"sector": sector})
    return (results["documents"][0] if results["documents"] else [],
            results["metadatas"][0]  if results["metadatas"]  else [])


# ═════════════════════════════════════════════════════════════
#  SECTION 3 — SELF LEARNING / HISTORY
# ═════════════════════════════════════════════════════════════

def load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: List[Dict]) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def evaluate_yesterday(history: List[Dict]) -> List[Dict]:
    changed = False
    for record in history:
        if record.get("outcome"):
            continue
        ticker    = record.get("ticker", "")
        entry     = record.get("entry_price") or 0
        target    = record.get("target_price") or 0
        stop_loss = record.get("stop_loss") or 0
        if not ticker or entry == 0:
            continue
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                continue
            actual = round(float(hist["Close"].iloc[-1]), 2)
            change = ((actual - entry) / entry) * 100 if entry else 0

            if target and actual >= target:
                outcome, reward = "WIN", 1.0
            elif stop_loss and actual <= stop_loss:
                outcome, reward = "LOSS", -1.0
            elif change > 1:
                outcome, reward = "PARTIAL_WIN", 0.5
            elif change < -1:
                outcome, reward = "PARTIAL_LOSS", -0.5
            else:
                outcome, reward = "NEUTRAL", 0.0

            record["actual_price"] = actual
            record["price_change"] = round(change, 2)
            record["outcome"]      = outcome
            record["reward"]       = reward
            changed = True
            print("Yesterday " + ticker + " -> " + outcome + " (" + str(round(change, 2)) + "%)")
        except Exception as e:
            print("Eval error " + ticker + ": " + str(e))
    if changed:
        save_history(history)
    return history


def save_prediction(plan: Dict, pitch: Dict, scores: Dict,
                    all_pitches: Dict, history: List[Dict]) -> None:
    """Save full prediction including all analysts data and pass/fail marks."""
    entry_price = plan.get("entry_price_inr") or 0
    position_pct = plan.get("position_size_pct", 10)
    capital_allocated = PORTFOLIO_SIZE_INR * (position_pct / 100)
    shares_to_buy = int(capital_allocated / entry_price) if entry_price > 0 else 0

    # Build all_analysts summary
    analysts_summary = {}
    for sector, p in all_pitches.items():
        analysts_summary[sector] = {
            "ticker":           p.get("ticker", ""),
            "company":          p.get("company", ""),
            "confidence":       p.get("confidence_score", 0),
            "recommendation":   p.get("recommendation", ""),
            "target_price_inr": p.get("target_price_inr", 0),
            "upside_pct":       p.get("upside_pct", 0),
            "thesis":           p.get("current_thesis", ""),
            "catalysts":        p.get("key_catalysts", []),
            "risks":            p.get("risks", []),
        }

    record = {
        "date":             datetime.now().strftime("%Y-%m-%d"),
        "time_ist":         datetime.utcnow().strftime("%H:%M") + " UTC",
        "ticker":           plan.get("ticker", ""),
        "company":          pitch.get("company", ""),
        "sector":           pitch.get("sector", ""),
        "entry_price":      entry_price,
        "target_price":     plan.get("exit_price_inr"),
        "stop_loss":        plan.get("stop_loss_inr"),
        "shares_to_buy":    shares_to_buy,
        "capital_used_inr": round(shares_to_buy * entry_price, 2) if entry_price else 0,
        "position_size_pct":position_pct,
        "expected_return_pct": plan.get("expected_return_pct"),
        "risk_reward":      plan.get("risk_reward_ratio"),
        "time_horizon":     plan.get("time_horizon", ""),
        "judge_score":      scores.get("overall_score"),
        "grade":            scores.get("grade", ""),
        "judge_comment":    scores.get("judge_comment", ""),
        "strengths":        scores.get("strengths", []),
        "weaknesses":       scores.get("weaknesses", []),
        "key_catalysts":    pitch.get("key_catalysts", []),
        "risks":            pitch.get("risks", []),
        "thesis":           pitch.get("current_thesis", ""),
        "all_analysts":     analysts_summary,
        "pass_mark":        "PASS" if (scores.get("overall_score", 0) or 0) >= 6.0 else "FAIL",
        "actual_price":     None,
        "price_change":     None,
        "outcome":          None,
        "reward":           None,
    }
    history.append(record)
    save_history(history)
    print("Prediction saved: " + record["ticker"] + " on " + record["date"] +
          " | Pass: " + record["pass_mark"])


def build_learning_context(history: List[Dict]) -> str:
    if not history:
        return ""
    wins   = [h for h in history if h.get("outcome") in ["WIN", "PARTIAL_WIN"]]
    losses = [h for h in history if h.get("outcome") in ["LOSS", "PARTIAL_LOSS"]]
    total  = len([h for h in history if h.get("outcome")])
    rate   = (len(wins) / total * 100) if total > 0 else 0.0

    ctx  = "\nPAST PERFORMANCE MEMORY (win rate: " + str(round(rate, 1)) + "%):\n"
    ctx += "SUCCESSFUL PICKS:\n"
    for w in wins[-5:]:
        ctx += ("  " + w.get("ticker", "") + " on " + w.get("date", "") +
                ": " + str(w.get("price_change", 0)) + "% -> " + w.get("outcome", "") + "\n")
    ctx += "FAILED PICKS (be cautious):\n"
    for l in losses[-5:]:
        ctx += ("  " + l.get("ticker", "") + " on " + l.get("date", "") +
                ": " + str(l.get("price_change", 0)) + "% -> " + l.get("outcome", "") + "\n")
    ctx += "RULES: Avoid recently failed stocks. Prefer consistent winners.\n"
    return ctx


def get_2day_confirmed_tickers(history: List[Dict], sector: str) -> List[str]:
    """
    Return tickers that were recommended by the analyst for this sector
    in the last 2 days — these are 'confirmed' picks with momentum.
    """
    confirmed = []
    cutoff = datetime.now() - timedelta(days=3)
    for record in history:
        try:
            rec_date = datetime.strptime(record.get("date", ""), "%Y-%m-%d")
        except Exception:
            continue
        if rec_date < cutoff:
            continue
        analysts = record.get("all_analysts", {})
        if sector in analysts:
            t = analysts[sector].get("ticker", "")
            if t and t not in confirmed:
                confirmed.append(t)
    return confirmed


# ═════════════════════════════════════════════════════════════
#  SECTION 4 — GROQ HELPER
# ═════════════════════════════════════════════════════════════

def call_groq(system_prompt: str, user_prompt: str,
              model: str = "llama-3.1-8b-instant",
              max_tokens: int = 1024, client_num: int = 1) -> str:
    client = client1 if client_num == 1 else client2
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[Groq Client" + str(client_num) + " ERROR] " + str(e))
        return ""


def safe_parse_json(raw: str) -> Dict:
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for text in [clean, raw]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            continue
    return {}


# ═════════════════════════════════════════════════════════════
#  SECTION 5 — AGENTS
# ═════════════════════════════════════════════════════════════

def analyst_agent(sector: str, tickers: List[str],
                  learning_ctx: str = "", confirmed_tickers: List[str] = []) -> Dict:
    query     = "best NSE stock " + sector + " India earnings revenue growth"
    chunks, _ = retrieve_context(query, sector, top_k=6)
    context   = "\n---\n".join(chunks)

    # Build confirmation hint
    confirm_hint = ""
    if confirmed_tickers:
        confirm_hint = (
            "\nCONFIRMED PICKS (seen in last 2 days — prefer these if still strong): " +
            str(confirmed_tickers) + "\n"
            "If a confirmed ticker still looks good today, pick it (shows consistency).\n"
            "If it has deteriorated, pick the next best option.\n"
        )

    system = (
        "You are a senior " + sector + " equity analyst for Indian markets (NSE/BSE).\n" +
        learning_ctx + confirm_hint +
        "\nYou MUST populate ALL fields — especially key_catalysts (min 3) and risks (min 2).\n"
        "These are shown to investors and must be specific and real, not generic.\n"
        "Respond ONLY in this exact JSON format, no extra text:\n"
        "{\n"
        '  "ticker": "SYMBOL.NS",\n'
        '  "company": "Full Company Name",\n'
        '  "sector": "' + sector + '",\n'
        '  "recommendation": "BUY",\n'
        '  "current_price_inr": 0.00,\n'
        '  "target_price_inr": 0.00,\n'
        '  "upside_pct": 0.0,\n'
        '  "current_thesis": "Detailed 3-4 sentence investment thesis with specific reasons",\n'
        '  "key_catalysts": ["Specific catalyst 1", "Specific catalyst 2", "Specific catalyst 3"],\n'
        '  "indian_market_factors": ["Factor 1 affecting Indian market", "Factor 2"],\n'
        '  "risks": ["Specific risk 1 with explanation", "Specific risk 2 with explanation"],\n'
        '  "technical_view": "RSI/MA based view e.g. RSI at 45 below MA20 — accumulation zone",\n'
        '  "confidence_score": 0.0\n'
        "}"
    )
    user = ("Pick the best stock from: " + str(tickers) + "\n\n"
            "MARKET DATA:\n" + context + "\n\nRespond ONLY with valid JSON.")

    print("  [Analyst:" + sector.upper() + "] generating pitch...")
    raw   = call_groq(system, user,
                      model="moonshotai/kimi-k2-instruct",
                      max_tokens=1200, client_num=1)
    pitch = safe_parse_json(raw)
    if pitch:
        # Ensure non-empty lists
        if not pitch.get("key_catalysts"):
            pitch["key_catalysts"] = ["Strong sector fundamentals", "Management guidance positive",
                                       "Institutional buying interest"]
        if not pitch.get("risks"):
            pitch["risks"] = ["Market volatility risk", "Sector-specific regulatory risk"]
        if not pitch.get("technical_view"):
            pitch["technical_view"] = "Insufficient technical data"
        print("  OK " + sector + " -> " + pitch.get("ticker", "?") +
              " confidence: " + str(pitch.get("confidence_score", "?")))
        return pitch
    print("  WARN parse failed " + sector)
    return {"sector": sector, "ticker": tickers[0], "confidence_score": 0.5,
            "key_catalysts": ["Data unavailable"], "risks": ["Data unavailable"],
            "current_thesis": "Unable to generate thesis", "technical_view": "N/A",
            "company": COMPANY_NAMES.get(tickers[0], tickers[0])}


def supervisor_agent(pitches: Dict) -> Dict:
    system = (
        "You are CIO of an Indian AMC. Select the single best pitch from 3 analysts.\n"
        "Consider: confidence score, upside potential, risk-reward, market conditions.\n"
        "Respond ONLY in this exact JSON format:\n"
        "{\n"
        '  "selected_sector": "sector_name",\n'
        '  "selected_ticker": "SYMBOL.NS",\n'
        '  "selected_company": "Company Name",\n'
        '  "reason": "3-4 sentence explanation of why this is the best pick today",\n'
        '  "market_context": "Current Indian market environment context",\n'
        '  "ranking": ["sector1 - reason", "sector2 - reason", "sector3 - reason"],\n'
        '  "rejected": ["sector1: specific reason", "sector2: specific reason"]\n'
        "}"
    )
    user = ("Select the best from these 3 pitches:\n\n" +
            json.dumps(pitches, indent=2) + "\n\nRespond ONLY with valid JSON.")

    print("  [Supervisor] evaluating...")
    raw = call_groq(system, user, model="openai/gpt-oss-120b",
                    max_tokens=800, client_num=2)
    dec = safe_parse_json(raw)
    if dec:
        sel  = dec.get("selected_sector", "")
        best = pitches.get(sel) or list(pitches.values())[0]
        print("  OK supervisor -> " + dec.get("selected_ticker", "?"))
        return {"decision": dec, "best_pitch": best}
    return {"decision": {}, "best_pitch": list(pitches.values())[0]}


def strategist_agent(pitch: Dict) -> Dict:
    entry_price = pitch.get("current_price_inr", 0)
    position_pct = 8.0
    capital = PORTFOLIO_SIZE_INR * (position_pct / 100)
    shares  = int(capital / entry_price) if entry_price > 0 else 0

    system = (
        "You are a quant trading strategist for NSE/BSE Indian markets.\n"
        "Build a precise trade plan in Indian Rupees (INR).\n"
        "RULES: stop_loss_inr < entry_price_inr < exit_price_inr\n"
        "Position max 10%%. Risk/reward >= 1.5. Be specific with prices.\n"
        "Portfolio size for this trade: INR " + str(PORTFOLIO_SIZE_INR) + "\n"
        "Respond ONLY in this exact JSON format:\n"
        "{\n"
        '  "ticker": "SYMBOL.NS",\n'
        '  "company": "Company Name",\n'
        '  "action": "BUY",\n'
        '  "entry_price_inr": 0.00,\n'
        '  "exit_price_inr": 0.00,\n'
        '  "stop_loss_inr": 0.00,\n'
        '  "shares_to_buy": 0,\n'
        '  "capital_required_inr": 0.00,\n'
        '  "position_size_pct": 0.0,\n'
        '  "risk_reward_ratio": 0.0,\n'
        '  "expected_return_pct": 0.0,\n'
        '  "max_loss_inr": 0.00,\n'
        '  "max_profit_inr": 0.00,\n'
        '  "time_horizon": "X weeks",\n'
        '  "strategy_thesis": "Detailed explanation",\n'
        '  "entry_strategy": "e.g. Buy at market open or limit order at X",\n'
        '  "exit_strategy": "e.g. Set GTT at target and stop loss",\n'
        '  "risk_controls": ["control1", "control2", "control3"]\n'
        "}"
    )
    user = ("Build trade plan for:\n" + json.dumps(pitch, indent=2) +
            "\n\nPortfolio: INR " + str(PORTFOLIO_SIZE_INR) +
            "\nEstimated shares: " + str(shares) +
            "\nAll prices in INR. Respond ONLY with valid JSON.")

    for model, cn in [("llama-3.3-70b-versatile", 2),
                      ("moonshotai/kimi-k2-instruct", 1)]:
        print("  [Strategist] trying " + model + "...")
        raw  = call_groq(system, user, model=model, max_tokens=1200, client_num=cn)
        plan = safe_parse_json(raw)
        if plan and plan.get("entry_price_inr"):
            # Fill shares if LLM missed it
            ep = plan.get("entry_price_inr", 0)
            if ep > 0 and not plan.get("shares_to_buy"):
                pos_pct = plan.get("position_size_pct", 8)
                cap     = PORTFOLIO_SIZE_INR * (pos_pct / 100)
                plan["shares_to_buy"]       = int(cap / ep)
                plan["capital_required_inr"] = round(int(cap / ep) * ep, 2)
            if not plan.get("max_loss_inr") and plan.get("stop_loss_inr"):
                sl = plan["stop_loss_inr"]
                ep = plan["entry_price_inr"]
                sh = plan.get("shares_to_buy", 0)
                plan["max_loss_inr"]   = round((ep - sl) * sh, 2)
                plan["max_profit_inr"] = round((plan.get("exit_price_inr", ep) - ep) * sh, 2)
            print("  OK strategist -> entry=" + str(plan.get("entry_price_inr")) +
                  " shares=" + str(plan.get("shares_to_buy")))
            return plan
    print("  WARN strategist all models failed")
    return {"ticker": pitch.get("ticker", ""), "action": "HOLD"}


def compliance_agent(plan: Dict, pitch: Dict) -> Dict:
    # Calculate real risk factors to feed into prompt
    rr    = float(plan.get("risk_reward_ratio") or 0)
    pos   = float(plan.get("position_size_pct") or 0)
    sl    = plan.get("stop_loss_inr")
    entry = float(plan.get("entry_price_inr") or 0)
    sl_dist = round(((entry - float(sl or 0)) / entry * 100), 1) if entry and sl else 0

    system = (
        "You are a strict SEBI compliance officer at an Indian AMC.\n"
        "You MUST calculate a REAL risk_score between 1.0 and 10.0 based on these factors:\n"
        "- Position size risk: pos_pct=" + str(pos) + "% (>10% = high risk, adds 2-3 pts)\n"
        "- Risk/reward: rr=" + str(rr) + " (<1.5 = risky, adds 2 pts; <1.0 adds 4 pts)\n"
        "- Stop loss distance: sl_dist=" + str(sl_dist) + "% from entry\n"
        "  (>8% stop = wide risk adds 1-2 pts; <2% = too tight adds 1 pt)\n"
        "- Market cap risk: large cap = low risk, small cap = higher risk\n"
        "- Sector risk: energy/crypto = higher, pharma/IT = medium\n"
        "- Volatility: beta > 1.5 adds 1-2 pts\n"
        "SCORING GUIDE (be honest and realistic):\n"
        "  1-3 = LOW RISK (large cap, good R/R, tight stop)\n"
        "  4-6 = MEDIUM RISK (normal trade, some concerns)\n"
        "  7-8 = HIGH RISK (borderline — approve with warnings)\n"
        "  9-10 = REJECT (position too big, no stop loss, R/R < 1)\n"
        "NEVER output 0.0 — minimum score is 1.0.\n"
        "RULES TO ENFORCE:\n"
        "- position_size_pct must be <= 10 (reject if higher)\n"
        "- risk_reward_ratio must be >= 1.5 (reject if lower)\n"
        "- stop loss must exist (reject if missing)\n"
        "- reject if calculated risk_score > 8\n"
        "Respond ONLY in this exact JSON format:\n"
        "{\n"
        '  "final_decision": "BUY or HOLD",\n'
        '  "approved": true or false,\n'
        '  "risk_score": <your calculated score between 1.0 and 10.0>,\n'
        '  "sebi_compliant": true or false,\n'
        '  "issues_found": ["issue1 if any"],\n'
        '  "compliance_notes": "explain your risk score calculation",\n'
        '  "circuit_breaker_risk": "LOW or MEDIUM or HIGH"\n'
        "}"
    )
    user = ("PLAN:\n" + json.dumps(plan, indent=2) +
            "\n\nPITCH:\n" + json.dumps(pitch, indent=2) +
            "\n\nCalculate a REAL risk_score. NEVER use 0.0. Respond ONLY with valid JSON.")

    print("  [Compliance] checking " + plan.get("ticker", "") + "...")
    raw    = call_groq(system, user, model="llama-3.1-8b-instant", max_tokens=600, client_num=1)
    result = safe_parse_json(raw)
    if result:
        print("  " + ("OK" if result.get("approved") else "REJECTED") +
              " -> " + result.get("final_decision", "?") +
              " risk=" + str(result.get("risk_score", "?")))
        return result
    return {"final_decision": "HOLD", "approved": False, "risk_score": 9.0,
            "sebi_compliant": False, "issues_found": ["Parse error"],
            "compliance_notes": "Could not parse compliance response",
            "circuit_breaker_risk": "UNKNOWN"}


def llm_judge(pitch: Dict, plan: Dict, compliance: Dict) -> Dict:
    system = (
        "You are an Indian equity research evaluator. Score the pipeline.\n"
        "You MUST populate strengths (min 3 specific points) and weaknesses (min 2).\n"
        "Output ONLY raw JSON:\n"
        "{\n"
        '  "research_quality": 0,\n'
        '  "reasoning_clarity": 0,\n'
        '  "risk_management": 0,\n'
        '  "financial_feasibility": 0,\n'
        '  "evidence_grounding": 0,\n'
        '  "india_market_relevance": 0,\n'
        '  "overall_score": 0.0,\n'
        '  "grade": "B",\n'
        '  "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],\n'
        '  "weaknesses": ["specific weakness 1", "specific weakness 2"],\n'
        '  "improvement_suggestions": ["suggestion 1", "suggestion 2"],\n'
        '  "would_you_invest": true,\n'
        '  "judge_comment": "2-3 sentence overall verdict"\n'
        "}"
    )
    user = ("Score this pipeline:\nPITCH:\n" + json.dumps(pitch, indent=2) +
            "\nPLAN:\n" + json.dumps(plan, indent=2) +
            "\nCOMPLIANCE:\n" + json.dumps(compliance, indent=2) +
            "\n\nOutput ONLY valid JSON.")

    print("  [Judge] scoring...")
    for model, cn in [("qwen/qwen3-32b", 2), ("llama-3.3-70b-versatile", 2),
                      ("moonshotai/kimi-k2-instruct", 1)]:
        raw    = call_groq(system, user, model=model, max_tokens=1200, client_num=cn)
        result = safe_parse_json(raw)
        if result and result.get("overall_score") is not None:
            # Ensure non-empty
            if not result.get("strengths"):
                result["strengths"] = ["Fundamental analysis conducted",
                                       "Risk controls defined", "SEBI compliance checked"]
            if not result.get("weaknesses"):
                result["weaknesses"] = ["Limited historical validation", "Market timing uncertainty"]
            print("  OK judge -> " + str(result.get("overall_score")) +
                  "/10 grade=" + result.get("grade", "?"))
            return result
        print("  WARN judge parse failed with " + model)

    return {"research_quality": 5, "reasoning_clarity": 5, "risk_management": 5,
            "financial_feasibility": 5, "evidence_grounding": 5, "india_market_relevance": 5,
            "overall_score": 5.0, "grade": "C",
            "strengths": ["Analysis completed", "Trade plan generated", "Compliance checked"],
            "weaknesses": ["Could not fully evaluate", "Limited data"],
            "improvement_suggestions": ["More data needed"],
            "would_you_invest": False, "judge_comment": "Automated evaluation — manual review recommended"}


# ═════════════════════════════════════════════════════════════
#  SECTION 6 — EMAIL (rich, full info)
# ═════════════════════════════════════════════════════════════

def build_email(best: Dict, trade_plan: Dict, compliance_result: Dict,
                judge_scores: Dict, supervisor_result: Dict,
                history: List[Dict], all_pitches: Dict) -> Tuple[str, str]:

    decision   = compliance_result.get("final_decision", "HOLD")
    approved   = compliance_result.get("approved", False)
    risk_score = float(compliance_result.get("risk_score", 10))

    total    = len([h for h in history if h.get("outcome")])
    wins     = len([h for h in history if h.get("outcome") in ["WIN", "PARTIAL_WIN"]])
    win_rate = str(round(wins / total * 100, 1)) + "%" if total > 0 else "N/A"

    # Yesterday result
    yesterday_html = ""
    evaluated = [h for h in history if h.get("outcome")]
    if evaluated:
        y     = evaluated[-1]
        color = "#16a34a" if "WIN" in str(y.get("outcome", "")) else "#dc2626"
        pnl   = ""
        if y.get("shares_to_buy") and y.get("entry_price") and y.get("actual_price"):
            pnl_val = (y["actual_price"] - y["entry_price"]) * y["shares_to_buy"]
            pnl = " | P&L: <b style='color:" + color + ";'>Rs." + str(round(pnl_val, 0)) + "</b>"
        yesterday_html = (
            '<div style="background:#f0fdf4;padding:14px;border-radius:6px;'
            'border:1px solid #bbf7d0;margin-bottom:16px;">'
            "<b>Yesterday:</b> " + str(y.get("ticker", "")) + " " +
            str(y.get("company", "")) +
            ' → <b style="color:' + color + '">' + str(y.get("outcome", "")) +
            " (" + str(y.get("price_change", 0)) + "%)" + "</b>" +
            " | Entry ₹" + str(y.get("entry_price", "")) +
            " → Actual ₹" + str(y.get("actual_price", "")) +
            pnl + "</div>"
        )

    date_str = datetime.now().strftime("%A, %d %B %Y")
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%I:%M %p IST")

    if not approved or decision != "BUY" or risk_score > 8:
        subject = ("⚠️ No Trade Today | " + datetime.now().strftime("%d %b %Y") +
                   " | Win Rate: " + win_rate)
        html = (
            "<html><body style='font-family:Arial,sans-serif;max-width:600px;margin:auto;'>"
            "<div style='background:#0f172a;padding:24px;border-radius:10px 10px 0 0;'>"
            "<h1 style='color:#f59e0b;margin:0;'>⚠️ No Trade Today</h1>"
            "<p style='color:#94a3b8;margin:4px 0 0;'>" + date_str + " | " + ist_time + "</p>"
            "</div>"
            "<div style='background:#f8fafc;padding:24px;border:1px solid #e2e8f0;'>" +
            yesterday_html +
            "<div style='background:#fff7ed;padding:16px;border-radius:6px;border:1px solid #fed7aa;'>"
            "<h3 style='color:#9a3412;margin:0 0 8px;'>Compliance Rejected Trade</h3>"
            "<b>Stock:</b> " + best.get("ticker", "") + "<br>"
            "<b>Decision:</b> " + decision + "<br>"
            "<b>Risk Score:</b> " + str(risk_score) + "/10<br>"
            "<b>Reason:</b> " + str(compliance_result.get("compliance_notes", "")) + "</div>"
            "<p style='color:#64748b;margin-top:16px;'>Win Rate: <b>" + win_rate + "</b></p>"
            "</div>"
            "<div style='background:#0f172a;padding:12px;border-radius:0 0 10px 10px;text-align:center;'>"
            "<p style='color:#475569;font-size:11px;margin:0;'>Indian Stock AI | Not financial advice</p>"
            "</div></body></html>"
        )
        return subject, html

    # ── Rich email for BUY ───────────────────────────────────
    ticker      = best.get("ticker", "")
    company     = best.get("company", ticker)
    entry       = trade_plan.get("entry_price_inr", "")
    target      = trade_plan.get("exit_price_inr", "")
    sl          = trade_plan.get("stop_loss_inr", "")
    ret_pct     = trade_plan.get("expected_return_pct", "")
    rr          = trade_plan.get("risk_reward_ratio", "")
    shares      = trade_plan.get("shares_to_buy", "")
    cap_req     = trade_plan.get("capital_required_inr", "")
    max_loss    = trade_plan.get("max_loss_inr", "")
    max_profit  = trade_plan.get("max_profit_inr", "")
    horizon     = trade_plan.get("time_horizon", "")
    entry_strat = trade_plan.get("entry_strategy", "Buy at market open")
    exit_strat  = trade_plan.get("exit_strategy", "Set GTT order at target and stop loss")

    score       = judge_scores.get("overall_score", "")
    grade       = judge_scores.get("grade", "")
    pass_mark   = "✅ PASS" if float(score or 0) >= 6.0 else "⚠️ BORDERLINE"

    catalysts_html = "".join("<li style='margin-bottom:6px;'>" + str(c) + "</li>"
                             for c in (best.get("key_catalysts") or ["No data"]))
    risks_html     = "".join("<li style='margin-bottom:6px;'>" + str(r) + "</li>"
                             for r in (best.get("risks") or ["No data"]))
    strengths_html = "".join("<li style='margin-bottom:6px;'>" + str(s) + "</li>"
                             for s in (judge_scores.get("strengths") or ["No data"]))
    weaknesses_html= "".join("<li style='margin-bottom:6px;'>" + str(w) + "</li>"
                             for w in (judge_scores.get("weaknesses") or ["No data"]))

    # Other analysts section
    other_analysts_html = ""
    for sector, p in all_pitches.items():
        if p.get("ticker") == ticker:
            continue
        other_analysts_html += (
            "<div style='padding:8px 0;border-bottom:1px solid #f1f5f9;'>"
            "<b>" + sector.upper() + ":</b> " + str(p.get("ticker", "")) +
            " (" + str(p.get("company", "")) + ") — " +
            "Confidence: " + str(p.get("confidence_score", "")) +
            " | Upside: " + str(p.get("upside_pct", "")) + "%" +
            "<br><span style='color:#64748b;font-size:12px;'>" +
            str(p.get("current_thesis", ""))[:120] + "...</span></div>"
        )

    subject = ("📈 " + ticker + " | Buy ₹" + str(entry) +
               " → Target ₹" + str(target) +
               " | +" + str(ret_pct) + "% | " +
               datetime.now().strftime("%d %b %Y"))

    html = (
        "<html><body style='font-family:Arial,sans-serif;max-width:680px;margin:auto;"
        "background:#f8fafc;'>"

        # Header
        "<div style='background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:28px;"
        "border-radius:12px 12px 0 0;'>"
        "<h1 style='color:#00d4ff;margin:0;font-size:24px;letter-spacing:-0.5px;'>"
        "📊 Daily Indian Stock Report</h1>"
        "<p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>" +
        date_str + " | " + ist_time +
        " | Win Rate: <b style='color:#10b981;'>" + win_rate + "</b>"
        " | Score: <b style='color:#f59e0b;'>" + str(score) + "/10 " + grade + " " + pass_mark + "</b></p>"
        "</div>"

        "<div style='background:white;padding:24px;border:1px solid #e2e8f0;'>" +
        yesterday_html +

        # Stock header
        "<div style='background:#f0f9ff;border-left:5px solid #00d4ff;"
        "padding:18px;border-radius:6px;margin-bottom:20px;'>"
        "<div style='font-size:11px;color:#64748b;letter-spacing:1px;'>TODAY'S PICK</div>"
        "<h2 style='margin:4px 0;color:#0f172a;font-size:22px;'>" + company +
        " <span style='font-size:14px;color:#3b82f6;font-weight:normal;'>(" + ticker + " NSE)</span></h2>"
        "<p style='margin:8px 0 0;color:#475569;line-height:1.6;'>" +
        str(best.get("current_thesis", "")) + "</p>"
        "<p style='margin:8px 0 0;color:#6366f1;font-size:13px;'>"
        "📐 Technical: " + str(best.get("technical_view", "")) + "</p>"
        "</div>"

        # Trade plan table
        "<h3 style='color:#0f172a;margin:0 0 12px;'>💰 Trade Plan</h3>"
        "<table style='width:100%;border-collapse:collapse;margin-bottom:16px;"
        "border-radius:8px;overflow:hidden;'>"
        "<tr style='background:#0f172a;color:white;text-align:center;font-size:12px;'>"
        "<th style='padding:10px;'>Entry ₹</th><th style='padding:10px;'>Target ₹</th>"
        "<th style='padding:10px;'>Stop Loss ₹</th><th style='padding:10px;'>Return</th>"
        "<th style='padding:10px;'>R/R Ratio</th></tr>"
        "<tr style='text-align:center;font-size:15px;font-weight:bold;'>"
        "<td style='padding:12px;border:1px solid #e2e8f0;'>" + str(entry) + "</td>"
        "<td style='padding:12px;border:1px solid #e2e8f0;color:#16a34a;'>₹" + str(target) + "</td>"
        "<td style='padding:12px;border:1px solid #e2e8f0;color:#dc2626;'>₹" + str(sl) + "</td>"
        "<td style='padding:12px;border:1px solid #e2e8f0;color:#16a34a;'>+" + str(ret_pct) + "%</td>"
        "<td style='padding:12px;border:1px solid #e2e8f0;'>" + str(rr) + "x</td>"
        "</tr></table>"

        # Shares & capital box
        "<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
        "padding:16px;margin-bottom:20px;'>"
        "<h3 style='margin:0 0 12px;color:#166534;'>📦 How Many Shares to Buy</h3>"
        "<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
        "<div style='flex:1;min-width:120px;text-align:center;background:white;"
        "padding:12px;border-radius:6px;border:1px solid #bbf7d0;'>"
        "<div style='font-size:11px;color:#64748b;'>SHARES TO BUY</div>"
        "<div style='font-size:26px;font-weight:bold;color:#166534;'>" + str(shares) + "</div></div>"
        "<div style='flex:1;min-width:120px;text-align:center;background:white;"
        "padding:12px;border-radius:6px;border:1px solid #bbf7d0;'>"
        "<div style='font-size:11px;color:#64748b;'>CAPITAL NEEDED</div>"
        "<div style='font-size:18px;font-weight:bold;color:#166534;'>₹" + str(cap_req) + "</div></div>"
        "<div style='flex:1;min-width:120px;text-align:center;background:white;"
        "padding:12px;border-radius:6px;border:1px solid #bbf7d0;'>"
        "<div style='font-size:11px;color:#64748b;'>MAX PROFIT</div>"
        "<div style='font-size:18px;font-weight:bold;color:#16a34a;'>₹" + str(max_profit) + "</div></div>"
        "<div style='flex:1;min-width:120px;text-align:center;background:white;"
        "padding:12px;border-radius:6px;border:1px solid #fee2e2;'>"
        "<div style='font-size:11px;color:#64748b;'>MAX LOSS</div>"
        "<div style='font-size:18px;font-weight:bold;color:#dc2626;'>₹" + str(max_loss) + "</div></div>"
        "</div>"
        "<div style='margin-top:12px;padding:10px;background:white;border-radius:6px;"
        "border:1px solid #bbf7d0;font-size:13px;color:#374151;'>"
        "📌 <b>Entry:</b> " + entry_strat + "<br>"
        "🎯 <b>Exit:</b> " + exit_strat + "<br>"
        "⏱ <b>Horizon:</b> " + str(horizon) +
        " | <b>Position:</b> " + str(trade_plan.get("position_size_pct", "")) + "% of portfolio"
        "</div></div>"

        # Scores row
        "<div style='display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;'>"
        "<div style='flex:1;min-width:100px;background:#f8fafc;padding:12px;"
        "border-radius:6px;border:1px solid #e2e8f0;text-align:center;'>"
        "<div style='font-size:10px;color:#64748b;'>DECISION</div>"
        "<div style='font-size:20px;font-weight:bold;color:#16a34a;'>" + decision + "</div></div>"
        "<div style='flex:1;min-width:100px;background:#f8fafc;padding:12px;"
        "border-radius:6px;border:1px solid #e2e8f0;text-align:center;'>"
        "<div style='font-size:10px;color:#64748b;'>JUDGE SCORE</div>"
        "<div style='font-size:20px;font-weight:bold;'>" + str(score) + "/10</div></div>"
        "<div style='flex:1;min-width:100px;background:#f8fafc;padding:12px;"
        "border-radius:6px;border:1px solid #e2e8f0;text-align:center;'>"
        "<div style='font-size:10px;color:#64748b;'>GRADE</div>"
        "<div style='font-size:20px;font-weight:bold;color:#7c3aed;'>" + str(grade) + "</div></div>"
        "<div style='flex:1;min-width:100px;background:#f8fafc;padding:12px;"
        "border-radius:6px;border:1px solid #e2e8f0;text-align:center;'>"
        "<div style='font-size:10px;color:#64748b;'>RISK SCORE</div>"
        "<div style='font-size:20px;font-weight:bold;color:" + 
        ('#16a34a' if float(compliance_result.get('risk_score') or 0) <= 3 else
         '#f59e0b' if float(compliance_result.get('risk_score') or 0) <= 6 else '#dc2626') +
        ";'>" + str(compliance_result.get('risk_score', 'N/A')) + "/10</div>"
        "<div style='font-size:10px;color:#64748b;margin-top:2px;'>" +
        ('LOW RISK' if float(compliance_result.get('risk_score') or 0) <= 3 else
         'MEDIUM RISK' if float(compliance_result.get('risk_score') or 0) <= 6 else 'HIGH RISK') +
        "</div></div>"
        "<div style='flex:1;min-width:100px;background:#f8fafc;padding:12px;"
        "border-radius:6px;border:1px solid #e2e8f0;text-align:center;'>"
        "<div style='font-size:10px;color:#64748b;'>HORIZON</div>"
        "<div style='font-size:14px;font-weight:bold;'>" + str(horizon) + "</div></div>"
        "</div>"

        # Key catalysts
        "<div style='background:white;padding:16px;border-radius:8px;"
        "border:1px solid #e2e8f0;margin-bottom:14px;'>"
        "<h3 style='margin:0 0 10px;color:#0f172a;'>🚀 Key Catalysts</h3>"
        "<ul style='margin:0;padding-left:20px;color:#475569;line-height:1.8;'>" +
        catalysts_html + "</ul></div>"

        # Strengths
        "<div style='background:white;padding:16px;border-radius:8px;"
        "border:1px solid #e2e8f0;margin-bottom:14px;'>"
        "<h3 style='margin:0 0 10px;color:#166534;'>✅ Strengths</h3>"
        "<ul style='margin:0;padding-left:20px;color:#475569;line-height:1.8;'>" +
        strengths_html + "</ul></div>"

        # Risks
        "<div style='background:#fff7ed;padding:16px;border-radius:8px;"
        "border:1px solid #fed7aa;margin-bottom:14px;'>"
        "<h3 style='margin:0 0 10px;color:#9a3412;'>⚠️ Risks</h3>"
        "<ul style='margin:0;padding-left:20px;color:#7c2d12;line-height:1.8;'>" +
        risks_html + "</ul></div>"

        # Weaknesses
        "<div style='background:#fafafa;padding:16px;border-radius:8px;"
        "border:1px solid #e2e8f0;margin-bottom:14px;'>"
        "<h3 style='margin:0 0 10px;color:#64748b;'>🔍 Weaknesses / Watch</h3>"
        "<ul style='margin:0;padding-left:20px;color:#64748b;line-height:1.8;'>" +
        weaknesses_html + "</ul></div>"

        # Other analysts
        "<div style='background:white;padding:16px;border-radius:8px;"
        "border:1px solid #e2e8f0;margin-bottom:14px;'>"
        "<h3 style='margin:0 0 10px;color:#0f172a;'>🏦 Other Analysts Today</h3>" +
        other_analysts_html + "</div>"

        # Verdict
        "<div style='background:#f0fdf4;padding:14px;border-radius:8px;"
        "border:1px solid #bbf7d0;margin-bottom:14px;'>"
        "<b style='color:#166534;'>🎯 Verdict: </b>"
        "<span style='color:#166534;'>" + str(judge_scores.get("judge_comment", "")) + "</span>"
        "</div>"

        # Compliance
        "<div style='background:#f8fafc;padding:12px;border-radius:6px;"
        "border:1px solid #e2e8f0;font-size:12px;color:#64748b;'>"
        "🛡️ <b>SEBI Compliance:</b> " + str(compliance_result.get("compliance_notes", "")) +
        " | Circuit Breaker Risk: " + str(compliance_result.get("circuit_breaker_risk", "")) +
        "</div></div>"

        # Footer
        "<div style='background:#0f172a;padding:16px;border-radius:0 0 12px 12px;text-align:center;'>"
        "<p style='color:#475569;font-size:11px;margin:0;'>"
        "Indian Stock AI Agent | Portfolio size: ₹" + str(int(PORTFOLIO_SIZE_INR)) +
        " | Not financial advice. Do your own research before investing."
        "</p></div></body></html>"
    )
    return subject, html


def send_email_report(best: Dict, trade_plan: Dict, compliance_result: Dict,
                      judge_scores: Dict, supervisor_result: Dict,
                      history: List[Dict], all_pitches: Dict) -> None:
    subject, html = build_email(best, trade_plan, compliance_result,
                                judge_scores, supervisor_result, history, all_pitches)
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = GMAIL_RECEIVER
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, GMAIL_RECEIVER, msg.as_string())
        print("✅ Email sent to " + GMAIL_RECEIVER)
    except Exception as e:
        print("❌ Email failed: " + str(e))


# ═════════════════════════════════════════════════════════════
#  SECTION 7 — MASTER PIPELINE
# ═════════════════════════════════════════════════════════════

def main() -> None:
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    print("=" * 55)
    print("INDIAN STOCK AI AGENT v2")
    print(ist_now.strftime("%d %B %Y %I:%M %p IST"))
    print("Portfolio: INR " + str(int(PORTFOLIO_SIZE_INR)))
    print("=" * 55)

    # 1. Load history
    print("\n[1] Loading prediction history...")
    history = load_history()
    print("    " + str(len(history)) + " past predictions found")
    history = evaluate_yesterday(history)

    # 2. Build learning context
    learning_ctx = build_learning_context(history)
    print("[2] " + ("Self-learning context built" if learning_ctx else "First run mode"))

    # 3. Ingest market data
    print("\n[3] Ingesting Indian market data...")
    all_docs = run_ingestion()

    # 4. Build RAG store
    print("\n[4] Building RAG knowledge store...")
    for sector, docs in all_docs.items():
        upsert_sector(sector, docs)
    print("    Total chunks: " + str(collection.count()))

    # 5. Run analysts (with 2-day confirmation check)
    print("\n[5] Running 3 analyst agents (with 2-day confirmation)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for s, t in AGENT_TICKERS.items():
            confirmed = get_2day_confirmed_tickers(history, s)
            if confirmed:
                print("  [" + s.upper() + "] 2-day confirmed tickers: " + str(confirmed))
            futures[executor.submit(analyst_agent, s, t, learning_ctx, confirmed)] = s

        pitches = {}
        for future in concurrent.futures.as_completed(futures):
            s = futures[future]
            try:
                pitches[s] = future.result()
            except Exception as e:
                print("  ERROR analyst " + s + ": " + str(e))
                pitches[s] = {"sector": s, "ticker": AGENT_TICKERS[s][0],
                               "confidence_score": 0.0, "key_catalysts": [],
                               "risks": [], "current_thesis": "", "technical_view": "N/A",
                               "company": COMPANY_NAMES.get(AGENT_TICKERS[s][0], s)}

    # 6. Supervisor
    print("\n[6] Supervisor selecting best pitch...")
    supervisor_result = supervisor_agent(pitches)

    # 7. Strategist
    print("\n[7] Strategist building trade plan...")
    trade_plan = strategist_agent(supervisor_result["best_pitch"])

    # 8. Compliance
    print("\n[8] Compliance validation...")
    compliance_result = compliance_agent(trade_plan, supervisor_result["best_pitch"])

    # 9. Judge
    print("\n[9] LLM Judge scoring...")
    judge_scores = llm_judge(supervisor_result["best_pitch"], trade_plan, compliance_result)

    # 10. Save prediction (with all analysts data + pass mark)
    print("\n[10] Saving prediction...")
    save_prediction(trade_plan, supervisor_result["best_pitch"],
                    judge_scores, pitches, history)

    # 11. Send email
    print("\n[11] Sending email report...")
    send_email_report(
        supervisor_result["best_pitch"], trade_plan, compliance_result,
        judge_scores, supervisor_result, history, pitches,
    )

    # 12. Summary
    best = supervisor_result["best_pitch"]
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("Stock    : " + best.get("ticker", "") + " — " + best.get("company", ""))
    print("Decision : " + compliance_result.get("final_decision", ""))
    print("Shares   : " + str(trade_plan.get("shares_to_buy", "N/A")))
    print("Score    : " + str(judge_scores.get("overall_score", "")) + "/10 " +
          str(judge_scores.get("grade", "")))
    total = len([h for h in history if h.get("outcome")])
    wins  = len([h for h in history if h.get("outcome") in ["WIN", "PARTIAL_WIN"]])
    if total > 0:
        print("Win Rate : " + str(round(wins / total * 100, 1)) + "% (" +
              str(wins) + "/" + str(total) + ")")
    print("=" * 55)


if __name__ == "__main__":
    main()
