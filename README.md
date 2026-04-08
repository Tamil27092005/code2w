---
title: Stocky OpenEnv
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - finance
  - multi-agent
  - reinforcement-learning
  - india
  - stock-market
---

# 📈 Stocky OpenEnv

**Indian Stock Market Multi-Agent Recommendation Environment**  
*Scaler School of Technology Hackathon Submission*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-00d4ff)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌍 Environment Description & Motivation

**Stocky OpenEnv** wraps a real-world Indian stock market recommendation system as a fully-interactive OpenEnv environment. It simulates the daily workflow of an equity research team analyzing NSE/BSE stocks using a **Swarm multi-agent architecture**: Analyst → Supervisor → Strategist → Compliance → Judge.

An AI agent learns to navigate this pipeline by:
- Reading live/mock stock market data (price, RSI, moving averages, 5-day momentum)
- Processing financial news headlines from Indian markets
- Recommending stocks with proper technical and fundamental reasoning
- Creating SEBI-compliant trade plans with valid risk/reward ratios
- Scoring recommendations using LLM judge evaluation

**Why this environment?**
Traditional finance RL environments use toy simulations or simplified price arrays. Stocky OpenEnv models the *actual workflow* a professional analyst follows: news ingestion → fundamental screening → technical confirmation → compliance review → recommendation. This creates a rich, multi-step decision problem with realistic partial rewards.

---

## 🎯 Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `identify_top_stock` | 🟢 Easy | 5 | Pick best stock from tech sector using RSI/MA/news |
| `create_trade_plan` | 🟡 Medium | 8 | Create SEBI-compliant trade plan for Infosys |
| `full_pipeline_recommendation` | 🔴 Hard | 15 | Orchestrate full 6-stage multi-agent pipeline |

### Task 1: Identify Top Stock (Easy)
The agent receives market data for 5 NSE tech stocks (TCS, Infosys, Wipro, HCL, Tech Mahindra) and must:
- Identify the best ticker with signal alignment (RSI 50-70 + positive momentum)
- Provide reasoning mentioning RSI, moving averages, catalysts
- State an explicit confidence score

### Task 2: Create Trade Plan (Medium)
The agent receives a pre-analyzed pitch for Infosys (current price ₹1,545) and must:
- Specify entry_price, target_price, stop_loss (all in INR)
- Ensure: `stop_loss < entry_price < target_price`
- Achieve risk/reward ratio ≥ 1.5
- Keep position size ≤ 10% (SEBI rule)

### Task 3: Full Pipeline (Hard)
The agent orchestrates the complete pipeline in 6 progressive stages:
1. **Sector Analysis** (Steps 1-3): Pick best stock per sector (tech/energy/healthcare)
2. **Supervisor Selection** (Steps 4-6): Choose the overall best pick across sectors
3. **Trade Planning** (Steps 7-9): Define entry, target, stop-loss, position size
4. **Compliance Validation** (Steps 10-11): Verify SEBI rules, R/R ratio
5. **Judge Scoring** (Steps 12-13): Score the recommendation 0-10
6. **Final Recommendation** (Steps 14-15): Write complete recommendation summary

---

## 📊 Observation Space

```python
class StockyObservation(BaseModel):
    echoed_message: str           # Agent's last message echoed back
    market_snapshot: dict         # Per-ticker: {price, rsi, ma20, ma50, chg5d, pe, volume, signal}
    news_headlines: list[str]     # Recent NSE/BSE market headlines
    agent_feedback: str           # Grader feedback with hints
    task_name: str                # Active task
    task_description: str         # Full task instructions
    step_count: int               # Current step (1-indexed)
    max_steps: int                # Maximum steps allowed
    reward_breakdown: dict        # Partial rewards accumulated so far
    history_summary: str          # Historical win/loss context
    valid_tickers: list[str]      # Available NSE tickers
```

**Market snapshot example:**
```json
{
  "TCS.NS":  {"price": 3842.0, "rsi": 61.2, "ma20": 3780.0, "chg5d": 1.8, "signal": "BULLISH"},
  "INFY.NS": {"price": 1545.0, "rsi": 52.3, "ma20": 1530.0, "chg5d": 0.9, "signal": "NEUTRAL"}
}
```

---

## ⚡ Action Space

```python
class StockyAction(BaseModel):
    message: str   # Natural language action (max 2000 chars)
```

The agent communicates entirely through **natural language messages**. For best rewards, include:
- Ticker symbols in `SYMBOL.NS` format
- Numerical price levels in INR
- RSI and moving average analysis
- Fundamental catalysts and risks
- SEBI compliance parameters (R/R ratio, position size)

**Example action:**
```
"I recommend TCS.NS (Tata Consultancy Services). RSI at 61.2 is bullish (50-70 zone),
above MA20 (3780) and MA50 (3650). 5-day momentum +1.8%. Catalyst: $500M deal win.
Entry: 3840, Target: 4100, Stop-loss: 3680. R/R: 2.1x. Confidence: 8/10."
```

---

## 🎁 Reward Function

Rewards are **dense** (every step), **partial** (credit for incomplete actions), and **cumulative** (sum to episode score).

| Component | Reward | When Triggered |
|-----------|--------|----------------|
| Valid ticker identified | +0.25 | Agent mentions valid NSE ticker |
| Signal alignment | +0.25 | Ticker has bullish RSI (50-70) + positive momentum |
| Reasoning quality | +0.30 | ≥4 financial keywords (RSI, MA, catalyst…) |
| Confidence stated | +0.20 | Explicit "confidence: N/10" |
| Price ordering valid | +0.20 | stop_loss < entry < target |
| R/R ratio ≥ 1.5 | +0.25 | Calculated R/R meets minimum |
| Position ≤ 10% | +0.15 | SEBI position size rule followed |
| Compliance approved | +0.15 | All SEBI criteria met |
| Judge score /10 | +0.20 | Score × (0.20/10) |
| Final completeness | +0.35 | All 6 required fields present |

**Episode score** = `sum(step_rewards) / max_total_reward`, clamped to `[0.0, 1.0]`

---

## 📈 Baseline Scores

Tested with a built-in **heuristic agent** (no LLM required, fully deterministic):

| Task | Heuristic Score | Random Score | Difficulty |
|------|----------------|--------------|-----------|
| `identify_top_stock` | 0.72 | 0.15 | Easy |
| `create_trade_plan` | 0.65 | 0.12 | Medium |
| `full_pipeline_recommendation` | 0.55 | 0.08 | Hard |
| **Average** | **0.64** | **0.12** | |

Reproduce these scores:
```bash
python baseline_inference.py --url http://localhost:7860 --heuristic
```

---

## 🚀 Setup & Usage

### Option 1: HuggingFace Space (Recommended)

```python
from stocky_env import StockyEnv, StockyAction

env = StockyEnv(base_url="https://your-space.hf.space")
result = await env.reset(task="identify_top_stock")

for step in range(result.observation.max_steps):
    action = StockyAction(message="I recommend TCS.NS because RSI=61...")
    result = await env.step(action)
    print(f"Step {step+1}: reward={result.reward:.4f}, done={result.done}")
    if result.done:
        break

await env.close()
```

### Option 2: Docker (Local)

```bash
# Build
docker build -t stocky-openenv .

# Run
docker run -p 7860:7860 stocky-openenv

# With API keys (for live pipeline)
docker run -p 7860:7860 \
  -e GROQ_KEY_1=your_key \
  -e GROQ_KEY_2=your_key2 \
  stocky-openenv

# Health check
curl http://localhost:7860/health
```

### Option 3: Python directly

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python app.py

# Or
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
# Heuristic baseline (no API key needed)
python baseline_inference.py --url http://localhost:7860 --heuristic

# Specific task
python baseline_inference.py --url http://localhost:7860 --heuristic --task identify_top_stock

# With LLM agent (requires OpenAI key)
python baseline_inference.py --url http://localhost:7860 --api-key sk-...
```

### API Quick Reference

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "identify_top_stock", "seed": 42}'

# Take action (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "action": {"message": "I recommend TCS.NS..."}}'

# Get current state
curl "http://localhost:7860/state?session_id=<id>"

# Close session
curl -X DELETE "http://localhost:7860/close?session_id=<id>"
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    STOCKY OPENENV                         │
│                  (OpenEnv Server)                         │
│                                                           │
│  /reset  /step  /state  /close  /health                   │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP
                         ▼
┌──────────────────────────────────────────────────────────┐
│              MULTI-AGENT PIPELINE (Swarm)                 │
│                                                           │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐               │
│  │Analyst 1│  │Analyst 2 │  │Analyst 3  │  (parallel)   │
│  │ (Tech)  │  │ (Energy) │  │(Healthcare)│               │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘               │
│       └────────────┴──────────────┘                      │
│                    │                                      │
│             ┌──────▼──────┐                              │
│             │  Supervisor  │  (selects best)             │
│             └──────┬──────┘                              │
│                    │                                      │
│             ┌──────▼──────┐                              │
│             │  Strategist  │  (trade plan)               │
│             └──────┬──────┘                              │
│                    │                                      │
│             ┌──────▼──────┐                              │
│             │  Compliance  │  (SEBI rules)               │
│             └──────┬──────┘                              │
│                    │                                      │
│             ┌──────▼──────┐                              │
│             │  LLM Judge   │  (scores 0-10)              │
│             └──────┬──────┘                              │
│                    │                                      │
│            ┌───────▼──────┐                              │
│            │ Email Report  │  (daily summary)            │
│            └──────────────┘                              │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
stocky-openenv/
├── README.md                    # This file (HF Space frontmatter + docs)
├── Dockerfile                   # Container definition (port 7860)
├── openenv.yaml                 # Environment specification
├── requirements.txt             # Python dependencies
├── app.py                       # FastAPI OpenEnv server
├── stocky_env.py                # Python client class
├── baseline_inference.py        # Baseline inference script
├── agent_pipeline.py            # Original Swarm multi-agent pipeline
└── prediction_history.json      # Prediction history (self-learning)
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `7860` | Server port |
| `GROQ_KEY_1` | No | `""` | Groq API key (for live LLM agents) |
| `GROQ_KEY_2` | No | `""` | Second Groq API key (fallback) |
| `FINNHUB_API_KEY` | No | `""` | Finnhub for live market data |
| `GMAIL_SENDER` | No | `""` | Gmail for email reports |
| `GMAIL_RECEIVER` | No | `""` | Recipient email |
| `GMAIL_PASSWORD` | No | `""` | Gmail app password |
| `PORTFOLIO_SIZE_INR` | No | `100000` | Portfolio size (₹) |

> **Note:** The OpenEnv server works with mock data out of the box. API keys are only needed for the full live pipeline (`agent_pipeline.py`).

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

*Built with ❤️ for the Scaler School of Technology Hackathon. Not financial advice.*
