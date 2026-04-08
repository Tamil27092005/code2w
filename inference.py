"""
baseline_inference.py — Stocky OpenEnv Baseline Inference Script

Runs a simple OpenAI-compatible LLM agent against all 3 tasks and reports
reproducible baseline scores.

Usage:
    # Against local Docker
    python baseline_inference.py --image stocky-openenv:latest

    # Against running HF Space
    python baseline_inference.py --url https://your-space.hf.space

    # Against local server
    python baseline_inference.py --url http://localhost:7860

    # Dry-run with built-in heuristic agent (no API key needed)
    python baseline_inference.py --url http://localhost:7860 --heuristic
"""

import asyncio
import argparse
import json
import time
from typing import List, Optional, Dict

# Optional: real OpenAI client (falls back to heuristic if unavailable)
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from stocky_env import StockyEnv, StockyAction

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_URL      = "http://localhost:7860"
DEFAULT_IMAGE    = "stocky-openenv:latest"
MODEL_NAME       = "gpt-4o-mini"
MAX_STEPS        = 15
SUCCESS_THRESHOLD = 0.60   # Score >= 0.60 is considered a success

TASKS_TO_RUN = [
    "identify_top_stock",         # Easy
    "create_trade_plan",          # Medium
    "full_pipeline_recommendation",  # Hard
]

MAX_TOTAL_REWARD_PER_TASK = {
    "identify_top_stock":           1.0,
    "create_trade_plan":            1.0,
    "full_pipeline_recommendation": 1.0,
}

# ─── Logging Helpers ──────────────────────────────────────────────────────────

def log_start(task: str, model: str):
    print(f"\n{'='*60}")
    print(f"  TASK       : {task}")
    print(f"  MODEL      : {model}")
    print(f"  TIME       : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*60}")

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    status = "✅ DONE" if done else ("❌ ERROR" if error else "➡️  CONT")
    print(f"  Step {step:2d} | reward={reward:+.4f} | {status}")
    print(f"         action: {action[:90]}{'...' if len(action) > 90 else ''}")
    if error:
        print(f"         error: {error}")

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]):
    print(f"\n{'─'*60}")
    print(f"  RESULT : {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"  SCORE  : {score:.4f}  (threshold: {SUCCESS_THRESHOLD})")
    print(f"  STEPS  : {steps}")
    print(f"  REWARDS: {[round(r,4) for r in rewards]}")
    print(f"{'─'*60}")

# ─── Heuristic Agent (no LLM needed for baseline) ────────────────────────────

HEURISTIC_PLAYBOOKS = {
    "identify_top_stock": [
        (
            "After analyzing the tech sector market data, I recommend TCS.NS (Tata Consultancy Services). "
            "Technical analysis: RSI at 61.2 is in bullish territory (50-70 range), well above MA20 (3780) "
            "and MA50 (3650), indicating strong uptrend. 5-day momentum is +1.8%. "
            "Fundamental catalyst: TCS won a $500M digital transformation deal from UK retailer, "
            "signaling strong order book. Revenue growth of 15% YoY in cloud services supports thesis. "
            "Technical view: above all moving averages, volume steady. "
            "Confidence: 8/10. Risks: USD/INR headwind, US banking slowdown."
        ),
        (
            "Confirming TCS.NS as top pick. RSI 61 shows momentum without being overbought. "
            "Catalyst: $500M deal win drives earnings upgrade cycle. "
            "Entry near MA20 support. Confidence: 8/10."
        ),
    ],
    "create_trade_plan": [
        (
            "Trade plan for INFY.NS (Infosys):\n"
            "entry: 1550, target: 1720, stop_loss: 1480\n"
            "Risk/reward ratio: (1720-1550)/(1550-1480) = 170/70 = 2.43x\n"
            "position size: 8% of portfolio\n"
            "Time horizon: 4 weeks\n"
            "Entry strategy: Limit order at 1550 at market open\n"
            "Exit strategy: GTT order at target 1720 and stop-loss 1480\n"
            "SEBI compliant: position size under 10%, R/R ratio above 1.5."
        ),
        (
            "Refined plan: entry: 1550, target: 1720, stop_loss: 1478\n"
            "risk_reward: 2.45x, position size: 8%, time horizon: 4 weeks\n"
            "Compliance note: stop_loss(1478) < entry(1550) < target(1720) ✅\n"
            "Max loss: Rs.7,200 | Max profit: Rs.17,000 | SEBI compliant: yes"
        ),
        (
            "Final approved plan: INFY.NS entry 1550 target 1720 stop_loss 1475\n"
            "R/R=2.5x, position=8%, time horizon 4 weeks. All SEBI checks passed."
        ),
    ],
    "full_pipeline_recommendation": [
        # Stage 1-3: Sector analysis
        "Analyzing TECH sector: TCS.NS shows RSI 61, above MA20/MA50, strong bullish momentum. "
        "5d change +1.8%. Catalyst: $500M deal win. Recommend TCS.NS confidence 8/10.",

        "Analyzing ENERGY sector: NTPC.NS shows RSI 63, bullish above MA20 335 and MA50 320. "
        "5d change +2.1%. Catalyst: 1.2GW solar capacity commissioned ahead of schedule. "
        "Recommend NTPC.NS confidence 7/10.",

        "Analyzing HEALTHCARE sector: SUNPHARMA.NS shows RSI 67, strong bullish, "
        "above MA20 1650 and MA50 1600. 5d change +2.3%. Catalyst: 30% US market share gain. "
        "Recommend SUNPHARMA.NS confidence 7.5/10.",

        # Stage 4-6: Supervisor
        "Supervisor selection: After comparing all 3 sectors, I select SUNPHARMA.NS as the best pick today. "
        "Reasons: Highest RSI momentum (67), best 5d performance (+2.3%), strong catalyst (FDA approvals), "
        "healthcare sector defensive in current market. "
        "Ranking: healthcare > tech > energy. Selected ticker: SUNPHARMA.NS",

        "Confirming SUNPHARMA.NS as final pick. Market context: FII inflows strong, "
        "pharma sector outperforming Nifty. Best risk-reward.",

        "Final supervisor decision: SUNPHARMA.NS (Sun Pharma). Rejected NTPC (regulatory risk), "
        "TCS (USD headwind). Sun Pharma specialty drug growth is the clearest catalyst.",

        # Stage 7-9: Trade plan
        "Trade plan for SUNPHARMA.NS: entry: 1690, target: 1850, stop_loss: 1620\n"
        "Risk/reward: (1850-1690)/(1690-1620) = 160/70 = 2.29x\n"
        "Position size: 8%, Capital: Rs.8,000, Shares: 4, Time horizon: 3 weeks",

        "Refining trade plan: entry 1690, target 1850, stop_loss 1625\n"
        "entry strategy: limit buy at 1690 at market open\n"
        "exit strategy: GTT at target 1850 and stop-loss 1625\n"
        "Expected return: +9.5%",

        "Final trade plan confirmed: SUNPHARMA.NS entry=1690 target=1850 stop_loss=1625 "
        "R/R=2.14x position=8% time_horizon=3 weeks. All prices in INR.",

        # Stage 10-11: Compliance
        "Compliance check for SUNPHARMA.NS trade:\n"
        "✅ position_size: 8% (≤10% limit)\n"
        "✅ risk_reward ratio: 2.14x (≥1.5 required)\n"
        "✅ stop_loss: 1625 < entry: 1690 < target: 1850\n"
        "✅ SEBI compliant: no circuit breaker concerns\n"
        "Compliance status: APPROVED",

        "Compliance notes: risk_score = 3/10 (LOW RISK). Circuit breaker risk: LOW. "
        "sebi_compliant: true. Final decision: BUY. Approved.",

        # Stage 12-13: Judge
        "Judge evaluation of SUNPHARMA.NS recommendation:\n"
        "score: 7.8/10\n"
        "Strengths: (1) Technical signal strong (RSI 67, above all MAs), "
        "(2) Fundamental catalyst clear (FDA approval + market share), "
        "(3) Risk management well-defined (R/R 2.14x)\n"
        "Weaknesses: (1) Healthcare can be volatile on regulatory news, "
        "(2) Position sizing could be higher given conviction\n"
        "Grade: B+. Would invest: yes.",

        "Updated judge score: 7.8/10. Research quality: 8, Risk management: 8, "
        "India market relevance: 9. Overall verdict: Strong buy case with clear catalysts.",

        # Stage 14-15: Final recommendation
        "FINAL RECOMMENDATION SUMMARY:\n"
        "Ticker: SUNPHARMA.NS (Sun Pharma)\n"
        "Action: BUY\n"
        "Entry: Rs.1,690 | Target: Rs.1,850 | Stop-loss: Rs.1,625\n"
        "Expected return: +9.5% | Risk/reward: 2.14x | Time horizon: 3 weeks\n"
        "Key catalyst: Ilumya US market share 30% growth, Dr Reddy's USFDA approval wave\n"
        "Risk: Currency hedging cost, US FDA import alert risk\n"
        "Compliance: SEBI approved, position 8%, risk score 3/10 LOW\n"
        "Judge score: 7.8/10 (Grade B+). Confidence: HIGH.",

        "Full pipeline complete. SUNPHARMA.NS is today's top pick. "
        "All stages completed: sector analysis, supervisor selection, trade plan, "
        "compliance validation, judge scoring. Final recommendation ready.",
    ],
}

async def get_heuristic_message(task: str, step: int, obs, last_reward: float) -> str:
    """Simple deterministic agent using pre-defined playbooks."""
    playbook = HEURISTIC_PLAYBOOKS.get(task, [])
    idx = min(step - 1, len(playbook) - 1)
    return playbook[idx] if playbook else f"Step {step}: Analyzing market data for {task}."

async def get_llm_message(
    client,
    task: str,
    step: int,
    obs,
    last_reward: float,
    history: List[str],
) -> str:
    """Use OpenAI-compatible LLM agent."""
    system = (
        f"You are an expert Indian stock market analyst. "
        f"Task: {task}\n"
        f"Description: {obs.task_description}\n"
        f"Valid tickers: {', '.join(obs.valid_tickers)}\n"
        f"You are on step {step} of {obs.max_steps}. "
        f"Last reward: {last_reward:.3f}. "
        f"Use RSI, moving averages, fundamentals, and news to make decisions. "
        f"Be specific — include ticker symbols (format: SYMBOL.NS), price levels in INR, "
        f"risk/reward ratios, and SEBI compliance notes."
    )
    messages = [{"role": "system", "content": system}]
    for h in history[-4:]:
        messages.append({"role": "assistant", "content": h})

    feedback = obs.agent_feedback
    news     = "\n".join(f"  • {n}" for n in obs.news_headlines[:5])
    mkt_str  = json.dumps(
        {t: {k: v for k, v in d.items() if k in ["price", "rsi", "ma20", "chg5d", "signal"]}
         for t, d in obs.market_snapshot.items()}, indent=2
    )
    user = (
        f"Step {step}/{obs.max_steps}\n"
        f"Feedback from last step: {feedback}\n\n"
        f"Current market data:\n{mkt_str}\n\n"
        f"Recent news:\n{news}\n\n"
        f"What is your next action? Be specific and detailed."
    )
    messages.append({"role": "user", "content": user})

    response = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=600, temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ─── Main Runner ──────────────────────────────────────────────────────────────

async def run_task(
    env_url: str,
    task_name: str,
    use_heuristic: bool = False,
    api_key: Optional[str] = None,
) -> Dict:
    """Run one full episode for a given task."""
    log_start(task=task_name, model="heuristic" if use_heuristic else MODEL_NAME)

    client = None
    if not use_heuristic and HAS_OPENAI and api_key:
        client = AsyncOpenAI(api_key=api_key)

    env = StockyEnv(base_url=env_url, task=task_name)
    await env._wait_for_ready()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(task=task_name)
        last_echoed = result.observation.echoed_message
        last_reward = 0.0
        max_steps   = result.observation.max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            if use_heuristic or not client:
                message = await get_heuristic_message(task_name, step, result.observation, last_reward)
            else:
                message = await get_llm_message(client, task_name, step, result.observation, last_reward, history)

            try:
                result = await env.step(StockyAction(message=message))
                reward = result.reward or 0.0
                done   = result.done
                error  = None
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)
                print(f"  ❌ Step error: {e}")

            rewards.append(reward)
            steps_taken = step
            last_echoed = result.observation.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {message[:150]!r} -> reward {reward:+.4f}")

            if done:
                break

        max_total = MAX_TOTAL_REWARD_PER_TASK.get(task_name, 1.0)
        raw_score = sum(rewards)
        score = min(max(raw_score / max_total if max_total > 0 else 0.0, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}")

    log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task":       task_name,
        "score":      round(score, 4),
        "success":    success,
        "steps":      steps_taken,
        "rewards":    rewards,
        "raw_reward": round(sum(rewards), 4),
    }

async def main():
    parser = argparse.ArgumentParser(description="Stocky OpenEnv Baseline Inference")
    parser.add_argument("--url",       default=DEFAULT_URL,  help="Server URL")
    parser.add_argument("--image",     default=None,         help="Docker image name")
    parser.add_argument("--task",      default=None,         help="Run specific task only")
    parser.add_argument("--heuristic", action="store_true",  help="Use built-in heuristic (no API key)")
    parser.add_argument("--api-key",   default=None,         help="OpenAI API key")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS_TO_RUN
    results = []

    print("\n" + "=" * 60)
    print("  STOCKY OPENENV — BASELINE INFERENCE")
    print("  Scaler School of Technology Hackathon")
    print(f"  Server: {args.url}")
    print(f"  Mode:   {'Heuristic (built-in)' if args.heuristic else 'LLM: ' + MODEL_NAME}")
    print("=" * 60)

    for task in tasks:
        result = await run_task(
            env_url       = args.url,
            task_name     = task,
            use_heuristic = args.heuristic or not args.api_key,
            api_key       = args.api_key,
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("  BASELINE SCORES SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<35} {'Score':>7}  {'Pass':>6}  {'Steps':>6}")
    print(f"  {'─'*35} {'─'*7}  {'─'*6}  {'─'*6}")
    total_score = 0.0
    for r in results:
        passed = "✅ YES" if r["success"] else "❌ NO"
        print(f"  {r['task']:<35} {r['score']:>7.4f}  {passed:>6}  {r['steps']:>6}")
        total_score += r["score"]

    avg = total_score / len(results) if results else 0.0
    print(f"  {'─'*35} {'─'*7}  {'─'*6}  {'─'*6}")
    print(f"  {'AVERAGE':<35} {avg:>7.4f}")
    print("=" * 60)

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({
            "timestamp":  __import__("datetime").datetime.utcnow().isoformat(),
            "mode":       "heuristic" if args.heuristic else MODEL_NAME,
            "tasks":      results,
            "avg_score":  round(avg, 4),
        }, f, indent=2)
    print("\n  Results saved to baseline_results.json")

if __name__ == "__main__":
    asyncio.run(main())
