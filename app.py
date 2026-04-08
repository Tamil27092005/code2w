"""
Stocky OpenEnv — Indian Stock Market Multi-Agent Environment
FastAPI server implementing the OpenEnv step()/reset()/state() API.
Runs on port 7860 for HuggingFace Spaces compatibility.
"""

import os
import uuid
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ─── Pydantic Models (OpenEnv Spec) ──────────────────────────────────────────

class StockyObservation(BaseModel):
    echoed_message: str                  # Agent's last message echoed back
    market_snapshot: Dict[str, Any]      # Live/mock stock data
    news_headlines: List[str]            # Recent relevant headlines
    agent_feedback: str                  # Pipeline agent response
    task_name: str                       # Active task identifier
    task_description: str                # What the agent needs to do
    step_count: int                      # Current step number
    max_steps: int                       # Maximum allowed steps
    reward_breakdown: Dict[str, float]   # Partial reward components so far
    history_summary: str                 # Win/loss history context
    valid_tickers: List[str]             # Available tickers for this task

class StockyAction(BaseModel):
    message: str                         # Agent's natural language action

class StepResult(BaseModel):
    observation: StockyObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task: Optional[str] = "full_pipeline_recommendation"   # Task name
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    session_id: str
    action: StockyAction

class StateRequest(BaseModel):
    session_id: str

# ─── Task Definitions ────────────────────────────────────────────────────────

TASKS = {
    "identify_top_stock": {
        "name": "identify_top_stock",
        "difficulty": "easy",
        "max_steps": 5,
        "max_total_reward": 5.0,
        "description": (
            "You are given live market data for the TECHNOLOGY sector (NSE India). "
            "Your task: identify the single best stock to BUY today. "
            "Provide: (1) the ticker symbol (e.g., TCS.NS), "
            "(2) your reasoning using RSI, moving averages, and news, "
            "(3) a confidence score 0-10. "
            "The grader will evaluate your reasoning quality and signal alignment."
        ),
        "sector": "tech",
        "tickers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    },
    "create_trade_plan": {
        "name": "create_trade_plan",
        "difficulty": "medium",
        "max_steps": 8,
        "max_total_reward": 8.0,
        "description": (
            "You are given a stock pitch for INFY.NS (Infosys). "
            "Your task: create a SEBI-compliant trade plan. "
            "You MUST specify: entry_price (INR), target_price (INR), "
            "stop_loss (INR), position_size_pct (max 10%%), and time_horizon. "
            "Constraints: stop_loss < entry_price < target_price, "
            "risk_reward_ratio >= 1.5, position_size <= 10%%. "
            "Iterate based on compliance feedback until the plan is approved."
        ),
        "sector": "tech",
        "tickers": ["INFY.NS"],
        "prefilled_pitch": {
            "ticker": "INFY.NS",
            "company": "Infosys Limited",
            "sector": "tech",
            "current_price_inr": 1545.0,
            "recommendation": "BUY",
            "current_thesis": "Infosys Q3 results beat estimates with 7% QoQ revenue growth and 150bps margin expansion. Large deal TCV of $2.1B and guidance upgrade signal strong demand environment.",
            "key_catalysts": [
                "FY25 guidance upgraded to 4.5-5% growth",
                "AI-led deal wins in financial services vertical",
                "Margin expansion from operational efficiency",
            ],
            "risks": [
                "USD/INR currency headwind if rupee strengthens",
                "US banking sector slowdown may impact BFSI vertical",
            ],
            "rsi_14": 52.3,
            "ma_20": 1530.0,
            "ma_50": 1490.0,
            "technical_view": "RSI at 52 — neutral zone, above MA20 and MA50 — bullish structure",
        },
    },
    "full_pipeline_recommendation": {
        "name": "full_pipeline_recommendation",
        "difficulty": "hard",
        "max_steps": 15,
        "max_total_reward": 15.0,
        "description": (
            "You are the orchestrator of a multi-agent Indian stock recommendation system. "
            "Complete the full pipeline across 3 sectors (tech, energy, healthcare): "
            "Step 1-3: Analyze each sector and pick a stock with reasoning. "
            "Step 4-6: Select the best stock across sectors (supervisor role). "
            "Step 7-9: Build a trade plan with entry/target/stop-loss. "
            "Step 10-11: Validate compliance (position ≤10%%, R/R ≥1.5). "
            "Step 12-13: Score the recommendation (judge role, 0-10). "
            "Step 14-15: Write the final recommendation summary. "
            "Partial rewards are given at each stage. Total score is cumulative."
        ),
        "sector": "all",
        "tickers": [
            "TCS.NS", "INFY.NS", "WIPRO.NS",
            "RELIANCE.NS", "ONGC.NS", "NTPC.NS",
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
        ],
    },
}

# ─── Mock Market Data (used when API keys unavailable) ───────────────────────

MOCK_MARKET_DATA = {
    "TCS.NS":        {"price": 3842.0, "rsi": 61.2, "ma20": 3780.0, "ma50": 3650.0, "chg5d": 1.8,  "pe": 28.4, "volume": 1200000},
    "INFY.NS":       {"price": 1545.0, "rsi": 52.3, "ma20": 1530.0, "ma50": 1490.0, "chg5d": 0.9,  "pe": 22.1, "volume": 3400000},
    "WIPRO.NS":      {"price": 458.0,  "rsi": 44.1, "ma20": 462.0,  "ma50": 470.0,  "chg5d": -0.8, "pe": 19.5, "volume": 2100000},
    "HCLTECH.NS":    {"price": 1623.0, "rsi": 58.7, "ma20": 1600.0, "ma50": 1555.0, "chg5d": 1.2,  "pe": 24.3, "volume": 900000},
    "TECHM.NS":      {"price": 1387.0, "rsi": 48.9, "ma20": 1395.0, "ma50": 1420.0, "chg5d": -0.3, "pe": 21.7, "volume": 780000},
    "RELIANCE.NS":   {"price": 2834.0, "rsi": 55.4, "ma20": 2800.0, "ma50": 2750.0, "chg5d": 1.5,  "pe": 27.8, "volume": 4200000},
    "ONGC.NS":       {"price": 247.0,  "rsi": 41.2, "ma20": 252.0,  "ma50": 260.0,  "chg5d": -1.2, "pe": 8.1,  "volume": 5600000},
    "NTPC.NS":       {"price": 342.0,  "rsi": 63.1, "ma20": 335.0,  "ma50": 320.0,  "chg5d": 2.1,  "pe": 14.2, "volume": 3900000},
    "POWERGRID.NS":  {"price": 298.0,  "rsi": 57.3, "ma20": 292.0,  "ma50": 285.0,  "chg5d": 1.0,  "pe": 16.8, "volume": 2300000},
    "ADANIGREEN.NS": {"price": 1124.0, "rsi": 38.4, "ma20": 1150.0, "ma50": 1190.0, "chg5d": -2.5, "pe": 89.2, "volume": 680000},
    "SUNPHARMA.NS":  {"price": 1687.0, "rsi": 66.8, "ma20": 1650.0, "ma50": 1600.0, "chg5d": 2.3,  "pe": 32.1, "volume": 1500000},
    "DRREDDY.NS":    {"price": 1234.0, "rsi": 53.2, "ma20": 1220.0, "ma50": 1195.0, "chg5d": 0.7,  "pe": 18.9, "volume": 890000},
    "CIPLA.NS":      {"price": 1456.0, "rsi": 59.4, "ma20": 1440.0, "ma50": 1410.0, "chg5d": 1.4,  "pe": 26.4, "volume": 1100000},
}

MOCK_NEWS = {
    "tech": [
        "TCS wins $500M digital transformation deal from UK retailer",
        "Infosys Q3 revenue beats estimates; upgrades FY25 guidance",
        "Indian IT sector sees 15% YoY growth in cloud services demand",
        "HCL Technologies announces 3-for-1 stock split, bullish sign",
        "Wipro restructuring shows early results; margin recovery expected",
    ],
    "energy": [
        "NTPC commissions 1.2GW solar capacity ahead of schedule",
        "Reliance Industries Q3 net profit up 12% YoY on refining margins",
        "India energy demand up 8% YoY; power stocks outperform index",
        "ONGC crude production stable; government royalty cut benefits",
        "Power Grid secures Rs 8,500 Cr transmission corridor project",
    ],
    "healthcare": [
        "Sun Pharma specialty drug Ilumya sees 30% US market share gain",
        "Dr Reddy's gets USFDA approval for generic Revlimid biosimilar",
        "Cipla respiratory portfolio grows 18% in domestic market",
        "India pharma exports hit record $25B in FY25",
        "Apollo Hospitals expansion plan to add 2,000 beds by FY26",
    ],
    "general": [
        "Nifty 50 trades at 18-month high; FII inflows cross $2B this week",
        "RBI maintains repo rate at 6.5%; signals accommodative stance",
        "Q3 earnings season beats expectations; 65% of Nifty companies outperform",
    ],
}

# ─── Session Store ───────────────────────────────────────────────────────────

sessions: Dict[str, Dict] = {}

# ─── Reward / Grading Logic ──────────────────────────────────────────────────

TICKER_PATTERN = re.compile(r'\b([A-Z]{2,10}\.NS)\b')
PRICE_PATTERN  = re.compile(r'(?:₹|Rs\.?|INR\s*)?([\d]{2,5}(?:\.\d{1,2})?)')
NUMBER_PATTERN = re.compile(r'[-+]?\d+\.?\d*')

VALID_TICKERS = set(MOCK_MARKET_DATA.keys())

def extract_tickers(text: str) -> List[str]:
    return [t for t in TICKER_PATTERN.findall(text.upper()) if t in VALID_TICKERS]

def extract_numbers(text: str) -> List[float]:
    return [float(n) for n in NUMBER_PATTERN.findall(text)]

def grade_task1(message: str, session: Dict) -> tuple:
    """Easy task grader: identify_top_stock"""
    reward = 0.0
    feedback_parts = []
    partial = {}

    tickers_found = extract_tickers(message)
    if tickers_found:
        ticker = tickers_found[0]
        partial["ticker_identified"] = 0.25
        reward += 0.25
        feedback_parts.append(f"✅ Valid ticker identified: {ticker}")

        data = MOCK_MARKET_DATA.get(ticker, {})
        rsi  = data.get("rsi", 50)
        chg  = data.get("chg5d", 0)

        # Reward for bullish signal alignment
        if rsi > 50 and rsi < 70 and chg > 0:
            partial["signal_alignment"] = 0.25
            reward += 0.25
            feedback_parts.append("✅ Good: ticker shows bullish RSI+momentum signal")
        elif rsi < 30 or rsi > 80:
            partial["signal_alignment"] = 0.05
            reward += 0.05
            feedback_parts.append("⚠️ Caution: RSI in extreme zone (overbought/oversold)")
        else:
            partial["signal_alignment"] = 0.15
            reward += 0.15
            feedback_parts.append("✅ Ticker signal is neutral-to-bullish")
    else:
        partial["ticker_identified"] = 0.0
        feedback_parts.append("❌ No valid NSE ticker found (format: SYMBOL.NS)")

    # Reward for reasoning quality
    reasoning_keywords = ["rsi", "ma", "moving average", "price", "support", "resistance",
                          "catalyst", "revenue", "earnings", "growth", "technical",
                          "fundamental", "news", "volume", "momentum"]
    kw_count = sum(1 for k in reasoning_keywords if k.lower() in message.lower())
    reasoning_score = min(kw_count / 4.0, 1.0) * 0.30
    partial["reasoning_quality"] = round(reasoning_score, 3)
    reward += reasoning_score
    feedback_parts.append(f"📊 Reasoning quality: {kw_count} relevant keywords → +{reasoning_score:.2f}")

    # Reward for confidence score mention
    if re.search(r'confidence[:\s]+\d', message.lower()) or re.search(r'\b([0-9]|10)/10\b', message):
        partial["confidence_stated"] = 0.20
        reward += 0.20
        feedback_parts.append("✅ Confidence score provided")
    else:
        partial["confidence_stated"] = 0.0
        feedback_parts.append("💡 Tip: State a confidence score (e.g., 'confidence: 8/10')")

    done = bool(tickers_found) and kw_count >= 3
    reward = round(min(reward, 1.0), 4)

    return reward, done, "\n".join(feedback_parts), partial


def grade_task2(message: str, session: Dict) -> tuple:
    """Medium task grader: create_trade_plan"""
    reward   = 0.0
    feedback = []
    partial  = {}
    pitch    = TASKS["create_trade_plan"]["prefilled_pitch"]
    entry_price = pitch["current_price_inr"]  # 1545.0

    nums = extract_numbers(message)
    msg_lower = message.lower()

    # Check for price mentions
    price_mentions = [n for n in nums if 1000 < n < 3000]

    entry_found  = None
    target_found = None
    sl_found     = None

    # Try to parse from message
    entry_match  = re.search(r'entry[:\s_]+(?:price[:\s]+)?([\d]{4,}(?:\.\d+)?)', msg_lower)
    target_match = re.search(r'(?:target|exit)[:\s_]+(?:price[:\s]+)?([\d]{4,}(?:\.\d+)?)', msg_lower)
    sl_match     = re.search(r'(?:stop.?loss|sl|stoploss)[:\s_]+(?:price[:\s]+)?([\d]{4,}(?:\.\d+)?)', msg_lower)

    if entry_match:
        entry_found = float(entry_match.group(1))
    if target_match:
        target_found = float(target_match.group(1))
    if sl_match:
        sl_found = float(sl_match.group(1))

    # If not found by keyword, try positional heuristic
    if not entry_found and len(price_mentions) >= 1:
        price_mentions_sorted = sorted(price_mentions)
        if len(price_mentions_sorted) >= 3:
            sl_found     = sl_found or price_mentions_sorted[0]
            entry_found  = entry_found or price_mentions_sorted[1]
            target_found = target_found or price_mentions_sorted[2]
        elif len(price_mentions_sorted) == 2:
            entry_found  = entry_found or price_mentions_sorted[0]
            target_found = target_found or price_mentions_sorted[1]

    # Grade price ordering
    if entry_found and target_found and sl_found:
        partial["prices_provided"] = 0.20
        reward += 0.20
        feedback.append(f"✅ Prices: SL={sl_found}, Entry={entry_found}, Target={target_found}")

        if sl_found < entry_found < target_found:
            partial["price_ordering"] = 0.20
            reward += 0.20
            feedback.append("✅ Valid price ordering: stop_loss < entry < target")
        else:
            partial["price_ordering"] = 0.0
            feedback.append("❌ Invalid ordering: need stop_loss < entry_price < target_price")

        # Risk-reward ratio
        if entry_found > 0 and sl_found < entry_found and target_found > entry_found:
            rr = (target_found - entry_found) / (entry_found - sl_found)
            if rr >= 1.5:
                partial["risk_reward"] = 0.25
                reward += 0.25
                feedback.append(f"✅ Risk/reward ratio = {rr:.2f} (≥1.5 required)")
            elif rr >= 1.0:
                partial["risk_reward"] = 0.10
                reward += 0.10
                feedback.append(f"⚠️ Risk/reward = {rr:.2f} — acceptable but below 1.5 target")
            else:
                partial["risk_reward"] = 0.0
                feedback.append(f"❌ Risk/reward = {rr:.2f} — below minimum 1.0")
    else:
        partial["prices_provided"] = 0.0
        partial["price_ordering"]  = 0.0
        partial["risk_reward"]     = 0.0
        feedback.append("❌ Please provide entry_price, target_price, and stop_loss in INR")
        feedback.append("💡 Format: 'entry: 1550, target: 1720, stop_loss: 1480'")

    # Position size check
    pos_match = re.search(r'position[_\s]*(?:size)?[:\s]+(\d+(?:\.\d+)?)\s*%?', msg_lower)
    if pos_match:
        pos_pct = float(pos_match.group(1))
        if pos_pct <= 10:
            partial["position_size"] = 0.15
            reward += 0.15
            feedback.append(f"✅ Position size {pos_pct}% is within 10% limit")
        else:
            partial["position_size"] = 0.0
            feedback.append(f"❌ Position size {pos_pct}% exceeds 10% limit — SEBI rule")
    else:
        partial["position_size"] = 0.05
        reward += 0.05
        feedback.append("💡 No position size specified (defaulting to 8%) — be explicit")

    # Time horizon
    if re.search(r'(?:week|month|day|horizon)', msg_lower):
        partial["time_horizon"] = 0.10
        reward += 0.10
        feedback.append("✅ Time horizon specified")
    else:
        partial["time_horizon"] = 0.0
        feedback.append("💡 Add time horizon (e.g., '3-4 weeks')")

    done = (reward >= 0.7)
    reward = round(min(reward, 1.0), 4)
    return reward, done, "\n".join(feedback), partial


def grade_task3_step(message: str, session: Dict) -> tuple:
    """Hard task grader: full_pipeline_recommendation — partial rewards per step"""
    step    = session["step_count"]
    reward  = 0.0
    feedback = []
    partial  = {}

    msg_lower = message.lower()
    cumulative = session.get("cumulative_reward", 0.0)

    # Stage 1 (steps 1-3): Sector analysis
    if step <= 3:
        sectors_mentioned = sum(1 for s in ["tech", "energy", "healthcare"] if s in msg_lower)
        tickers_found = extract_tickers(message)
        if tickers_found and sectors_mentioned >= 1:
            step_reward = 0.05 + (0.02 * min(sectors_mentioned, 3))
            partial["sector_analysis"] = step_reward
            reward += step_reward
            feedback.append(f"✅ Stage 1: Sector analysis — {sectors_mentioned} sector(s), tickers: {tickers_found}")
        else:
            feedback.append("💡 Stage 1: Analyze each sector. Mention sector + specific ticker with reasoning.")

    # Stage 2 (steps 4-6): Supervisor selection
    elif step <= 6:
        tickers = extract_tickers(message)
        if tickers and any(kw in msg_lower for kw in ["best", "select", "pick", "winner", "recommend", "chose"]):
            step_reward = 0.10
            partial["supervisor_selection"] = step_reward
            reward += step_reward
            feedback.append(f"✅ Stage 2: Supervisor selected {tickers[0]} as best pick")
            session["selected_ticker"] = tickers[0]
        else:
            feedback.append("💡 Stage 2: Select the BEST stock across all 3 sectors. State your choice clearly.")

    # Stage 3 (steps 7-9): Trade plan
    elif step <= 9:
        nums = extract_numbers(message)
        price_vals = [n for n in nums if 100 < n < 5000]
        has_sl     = re.search(r'stop.?loss|sl\b', msg_lower)
        has_target = re.search(r'target|exit', msg_lower)
        has_entry  = re.search(r'entry|buy.at|purchase', msg_lower)

        if has_entry and has_target and has_sl and len(price_vals) >= 3:
            step_reward = 0.15
            partial["trade_plan"] = step_reward
            reward += step_reward
            feedback.append("✅ Stage 3: Trade plan complete with entry/target/stop-loss")
        elif len(price_vals) >= 2:
            step_reward = 0.08
            reward += step_reward
            feedback.append("⚠️ Stage 3: Partial plan — include all 3 price levels (entry, target, stop_loss)")
        else:
            feedback.append("💡 Stage 3: Create trade plan — specify entry_price, target_price, stop_loss in INR")

    # Stage 4 (steps 10-11): Compliance
    elif step <= 11:
        compliance_kws = ["compliant", "sebi", "risk", "approved", "position size", "r/r", "risk.reward"]
        if any(kw in msg_lower for kw in compliance_kws):
            nums = extract_numbers(message)
            rr_candidates = [n for n in nums if 1.0 <= n <= 5.0]
            rr = rr_candidates[0] if rr_candidates else 0.0
            if rr >= 1.5:
                step_reward = 0.15
                partial["compliance"] = step_reward
                reward += step_reward
                feedback.append(f"✅ Stage 4: Compliance approved — R/R ratio {rr:.1f}x ≥ 1.5")
            else:
                step_reward = 0.08
                reward += step_reward
                feedback.append("⚠️ Stage 4: Compliance check done but verify R/R ≥ 1.5")
        else:
            feedback.append("💡 Stage 4: Validate compliance — check position size ≤10%%, R/R ≥1.5, SEBI rules")

    # Stage 5 (steps 12-13): Judge scoring
    elif step <= 13:
        score_match = re.search(r'(?:score|judge|grade|rating)[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?', msg_lower)
        if score_match:
            score_val = float(score_match.group(1))
            if 0 <= score_val <= 10:
                normalized = score_val / 10.0 * 0.20
                partial["judge_score"] = normalized
                reward += normalized
                feedback.append(f"✅ Stage 5: Judge score {score_val}/10 → reward +{normalized:.3f}")
        else:
            feedback.append("💡 Stage 5: Provide a judge score (e.g., 'score: 7.5/10') with detailed strengths/weaknesses")

    # Stage 6 (steps 14-15): Final recommendation
    elif step <= 15:
        required = ["ticker", "entry", "target", "stop", "risk", "catalyst"]
        present  = sum(1 for r in required if r in msg_lower)
        completeness = present / len(required)
        step_reward  = 0.35 * completeness
        partial["final_recommendation"] = round(step_reward, 3)
        reward += step_reward
        feedback.append(
            f"✅ Stage 6: Final recommendation completeness {present}/{len(required)} → reward +{step_reward:.3f}"
        )
        if completeness >= 0.8:
            feedback.append("🎉 Excellent final recommendation!")

    done = (step >= TASKS["full_pipeline_recommendation"]["max_steps"]) or (cumulative + reward >= 0.90)
    reward = round(min(reward, 1.0), 4)
    return reward, done, "\n".join(feedback), partial


# ─── Session Helpers ─────────────────────────────────────────────────────────

def build_market_snapshot(task_name: str) -> Dict:
    task = TASKS[task_name]
    tickers = task.get("tickers", [])
    return {
        ticker: {
            **MOCK_MARKET_DATA.get(ticker, {}),
            "signal": (
                "BULLISH" if MOCK_MARKET_DATA.get(ticker, {}).get("rsi", 50) > 52
                          and MOCK_MARKET_DATA.get(ticker, {}).get("chg5d", 0) > 0
                else "BEARISH" if MOCK_MARKET_DATA.get(ticker, {}).get("rsi", 50) < 45
                else "NEUTRAL"
            ),
        }
        for ticker in tickers
    }

def get_news_for_task(task_name: str) -> List[str]:
    task   = TASKS[task_name]
    sector = task.get("sector", "general")
    if sector == "all":
        news = []
        for s in ["tech", "energy", "healthcare", "general"]:
            news.extend(MOCK_NEWS.get(s, [])[:2])
        return news
    return MOCK_NEWS.get(sector, []) + MOCK_NEWS.get("general", [])[:2]

def load_history_summary() -> str:
    path = "prediction_history.json"
    if not os.path.exists(path):
        return "No previous predictions (first run)."
    try:
        with open(path) as f:
            hist = json.load(f)
        total = len([h for h in hist if h.get("outcome")])
        wins  = len([h for h in hist if h.get("outcome") in ["WIN", "PARTIAL_WIN"]])
        rate  = round(wins / total * 100, 1) if total > 0 else 0.0
        recent = hist[-3:] if len(hist) >= 3 else hist
        lines  = [f"Win Rate: {rate}% ({wins}/{total} evaluated)"]
        for r in recent:
            outcome = r.get("outcome", "PENDING")
            lines.append(f"  {r.get('date','?')}: {r.get('ticker','?')} → {outcome} ({r.get('price_change',0)}%)")
        return "\n".join(lines)
    except Exception:
        return "History unavailable."

def make_initial_observation(task_name: str) -> StockyObservation:
    task     = TASKS[task_name]
    snapshot = build_market_snapshot(task_name)
    news     = get_news_for_task(task_name)
    history  = load_history_summary()

    init_feedback = (
        f"🚀 Environment initialized for task: {task['name']} (Difficulty: {task['difficulty'].upper()})\n"
        f"Max steps: {task['max_steps']} | "
        f"Stocks available: {', '.join(task.get('tickers', []))}\n"
        f"Market data loaded for {len(snapshot)} tickers. "
        f"Recent news: {len(news)} headlines.\n"
        f"{'📋 Pre-loaded pitch: ' + task['prefilled_pitch']['ticker'] if 'prefilled_pitch' in task else ''}"
    )

    return StockyObservation(
        echoed_message   = "ENVIRONMENT READY — Make your first move.",
        market_snapshot  = snapshot,
        news_headlines   = news,
        agent_feedback   = init_feedback,
        task_name        = task_name,
        task_description = task["description"],
        step_count       = 0,
        max_steps        = task["max_steps"],
        reward_breakdown = {},
        history_summary  = history,
        valid_tickers    = task.get("tickers", []),
    )

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title        = "Stocky OpenEnv",
    description  = "Indian Stock Market Multi-Agent Environment (OpenEnv Spec)",
    version      = "1.0.0",
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """HuggingFace Space landing page."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stocky OpenEnv 📈</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; 
           max-width: 860px; margin: 40px auto; padding: 24px; }
    h1 { color: #00d4ff; font-size: 2rem; }
    h2 { color: #f59e0b; margin-top: 28px; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; 
             font-size: 12px; font-weight: bold; margin: 2px; }
    .easy   { background: #166534; color: #bbf7d0; }
    .medium { background: #92400e; color: #fde68a; }
    .hard   { background: #7f1d1d; color: #fecaca; }
    code { background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #7dd3fc; }
    pre  { background: #1e293b; padding: 16px; border-radius: 8px; overflow-x: auto; color: #a5f3fc; }
    table { width: 100%; border-collapse: collapse; margin: 16px 0; }
    th, td { padding: 10px 14px; border: 1px solid #334155; text-align: left; }
    th { background: #1e3a5f; color: #7dd3fc; }
    tr:hover { background: #1e293b; }
    .endpoint { color: #86efac; font-weight: bold; }
  </style>
</head>
<body>
  <h1>📈 Stocky OpenEnv</h1>
  <p>Indian Stock Market Multi-Agent Recommendation Environment — OpenEnv Compatible</p>
  <p>Built for <strong>Scaler School of Technology Hackathon</strong></p>

  <h2>🎯 Tasks</h2>
  <table>
    <tr><th>Task</th><th>Difficulty</th><th>Max Steps</th><th>Description</th></tr>
    <tr>
      <td><code>identify_top_stock</code></td>
      <td><span class="badge easy">EASY</span></td>
      <td>5</td>
      <td>Pick the best stock in the tech sector using market signals</td>
    </tr>
    <tr>
      <td><code>create_trade_plan</code></td>
      <td><span class="badge medium">MEDIUM</span></td>
      <td>8</td>
      <td>Build SEBI-compliant trade plan for Infosys with valid R/R ratio</td>
    </tr>
    <tr>
      <td><code>full_pipeline_recommendation</code></td>
      <td><span class="badge hard">HARD</span></td>
      <td>15</td>
      <td>Orchestrate full multi-agent pipeline across 3 sectors</td>
    </tr>
  </table>

  <h2>🔌 API Endpoints</h2>
  <table>
    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
    <tr><td>POST</td><td class="endpoint">/reset</td><td>Start new episode</td></tr>
    <tr><td>POST</td><td class="endpoint">/step</td><td>Take action, get observation + reward</td></tr>
    <tr><td>GET</td><td class="endpoint">/state</td><td>Get current environment state</td></tr>
    <tr><td>GET</td><td class="endpoint">/health</td><td>Health check</td></tr>
    <tr><td>GET</td><td class="endpoint">/docs</td><td>Interactive API docs (Swagger)</td></tr>
  </table>

  <h2>🚀 Quick Start</h2>
  <pre>
# Reset environment
curl -X POST https://&lt;your-space&gt;.hf.space/reset \\
  -H "Content-Type: application/json" \\
  -d '{"task": "identify_top_stock"}'

# Step
curl -X POST https://&lt;your-space&gt;.hf.space/step \\
  -H "Content-Type: application/json" \\
  -d '{"session_id": "&lt;id&gt;", "action": {"message": "I recommend TCS.NS..."}}'
  </pre>
  <p>Full docs: <a href="/docs" style="color:#00d4ff">/docs</a> | 
     <a href="/openapi.json" style="color:#00d4ff">/openapi.json</a> |
     <a href="/health" style="color:#00d4ff">/health</a></p>
</body>
</html>
""")

@app.get("/health")
async def health():
    return {"status": "ok", "env": "stocky-openenv", "version": "1.0.0",
            "tasks": list(TASKS.keys()), "active_sessions": len(sessions)}

@app.post("/reset", response_model=StepResult)
async def reset_env(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    """Reset the environment and start a new episode."""
    task_name = req.task or "full_pipeline_recommendation"
    if task_name not in TASKS:
        raise HTTPException(400, f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}")

    session_id = str(uuid.uuid4())
    obs = make_initial_observation(task_name)

    sessions[session_id] = {
        "task_name":        task_name,
        "step_count":       0,
        "cumulative_reward": 0.0,
        "reward_breakdown": {},
        "done":             False,
        "history":          [],
        "last_observation": obs,
        "created_at":       datetime.utcnow().isoformat(),
        "selected_ticker":  None,
    }

    return StepResult(
        observation = obs,
        reward      = 0.0,
        done        = False,
        info        = {"session_id": session_id, "task": task_name,
                       "message": "Episode started. Call /step with this session_id."},
    )

@app.post("/step", response_model=StepResult)
async def step_env(req: StepRequest):
    """Take one step in the environment."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")
    if session["done"]:
        raise HTTPException(400, "Episode is done. Call /reset to start a new episode.")

    task_name  = session["task_name"]
    task       = TASKS[task_name]
    message    = req.action.message
    step_count = session["step_count"] + 1
    session["step_count"] = step_count

    # Grade the action based on task
    if task_name == "identify_top_stock":
        step_reward, done, feedback, partial = grade_task1(message, session)
    elif task_name == "create_trade_plan":
        step_reward, done, feedback, partial = grade_task2(message, session)
    else:  # full_pipeline_recommendation
        step_reward, done, feedback, partial = grade_task3_step(message, session)

    # Accumulate
    session["cumulative_reward"] = round(
        session.get("cumulative_reward", 0.0) + step_reward, 4
    )
    session["reward_breakdown"].update(partial)

    # Check done
    if step_count >= task["max_steps"]:
        done = True
    session["done"] = done

    # Build new observation
    obs = StockyObservation(
        echoed_message   = message[:500],
        market_snapshot  = build_market_snapshot(task_name),
        news_headlines   = get_news_for_task(task_name),
        agent_feedback   = feedback,
        task_name        = task_name,
        task_description = task["description"],
        step_count       = step_count,
        max_steps        = task["max_steps"],
        reward_breakdown = session["reward_breakdown"],
        history_summary  = load_history_summary(),
        valid_tickers    = task.get("tickers", []),
    )
    session["last_observation"] = obs

    # Log step
    session["history"].append({
        "step": step_count, "action": message[:200],
        "reward": step_reward, "cumulative": session["cumulative_reward"],
        "done": done,
    })

    return StepResult(
        observation = obs,
        reward      = step_reward,
        done        = done,
        info        = {
            "session_id":        req.session_id,
            "step":              step_count,
            "cumulative_reward": session["cumulative_reward"],
            "max_steps":         task["max_steps"],
            "episode_done":      done,
        },
    )

@app.get("/state")
async def get_state(session_id: str):
    """Get current state of the environment session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {
        "session_id":        session_id,
        "task_name":         session["task_name"],
        "step_count":        session["step_count"],
        "cumulative_reward": session["cumulative_reward"],
        "done":              session["done"],
        "reward_breakdown":  session["reward_breakdown"],
        "step_history":      session["history"],
    }

@app.delete("/close")
async def close_env(session_id: str):
    """Close and clean up an environment session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "closed", "session_id": session_id}

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
