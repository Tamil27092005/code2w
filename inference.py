"""
inference.py — Stocky OpenEnv Inference Script

Uses the LiteLLM proxy provided by the validator via:
  - API_BASE_URL  : LLM proxy endpoint
  - API_KEY       : Proxy API key
  - MODEL_NAME    : Model to use for inference
  - HF_TOKEN      : HuggingFace API token

Prints structured [START]/[STEP]/[END] blocks to stdout.
"""

import asyncio
import os
import json
from typing import List, Dict
from openai import AsyncOpenAI
from stocky_env import StockyEnv, StockyAction

# ─── Configuration from environment ──────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", ""))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
SERVER_URL   = os.environ.get("SPACE_URL", "https://TamilSelvan0709-code2w-space.hf.space")

SUCCESS_THRESHOLD = 0.60

TASKS_TO_RUN = [
    "identify_top_stock",
    "create_trade_plan",
    "full_pipeline_recommendation",
]

MAX_TOTAL_REWARD_PER_TASK = {
    "identify_top_stock":           1.0,
    "create_trade_plan":            1.0,
    "full_pipeline_recommendation": 1.0,
}

# ─── Structured Output ────────────────────────────────────────────────────────

def print_start(task: str):
    print(f"[START] task={task}", flush=True)

def print_step(step: int, reward: float, done: bool):
    print(f"[STEP] step={step} reward={round(reward, 4)} done={str(done).lower()}", flush=True)

def print_end(task: str, : float, steps: int):
    print(f"[END] task={task} score={round(score, 4)} steps={steps}", flush=True)

# ─── LLM Agent ────────────────────────────────────────────────────────────────

async def get_llm_message(
    client: AsyncOpenAI,
    task: str,
    step: int,
    obs,
    history: List[str],
) -> str:
    system_prompt = (
        f"You are an expert Indian stock market analyst using the OpenEnv framework.\n"
        f"Task: {task}\n"
        f"Description: {obs.task_description}\n"
        f"Valid tickers: {', '.join(obs.valid_tickers)}\n"
        f"You are on step {step} of {obs.max_steps}.\n"
        f"Rules:\n"
        f"- Always reference tickers in SYMBOL.NS format (e.g., TCS.NS)\n"
        f"- Include numerical price levels in INR\n"
        f"- Mention RSI, MA20, MA50 values\n"
        f"- State confidence score explicitly (e.g., confidence: 8/10)\n"
        f"- For trade plans: stop_loss < entry < target\n"
        f"- Keep position_size <= 10%\n"
        f"- Maintain risk_reward >= 1.5\n"
    )

    market_str = json.dumps(
        {
            t: {k: v for k, v in d.items() if k in ["price", "rsi", "ma20", "ma50", "chg5d", "signal"]}
            for t, d in obs.market_snapshot.items()
        },
        indent=2,
    )
    news_str = "\n".join(f"- {n}" for n in obs.news_headlines[:5])

    user_prompt = (
        f"Step {step}/{obs.max_steps}\n"
        f"Last feedback: {obs.agent_feedback}\n\n"
        f"Market data:\n{market_str}\n\n"
        f"Recent news:\n{news_str}\n\n"
        f"Previous steps summary:\n" + ("\n".join(history[-3:]) if history else "None") +
        f"\n\nProvide your next action. Be specific and detailed."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ─── Task Runner ──────────────────────────────────────────────────────────────

async def run_task(client: AsyncOpenAI, env_url: str, task_name: str) -> Dict:
    print_start(task_name)

    env = StockyEnv(base_url=env_url, task=task_name)
    await env._wait_for_ready()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    history: List[str] = []

    try:
        result = await env.reset(task=task_name)
        max_steps = result.observation.max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            try:
                message = await get_llm_message(client, task_name, step, result.observation, history)
            except Exception as e:
                print(f"[ERROR] LLM call failed at step={step}: {e}", flush=True)
                message = f"Step {step}: Analyzing market data for {task_name}."

            try:
                result = await env.step(StockyAction(message=message))
                reward = result.reward or 0.0
                done   = result.done
            except Exception as e:
                print(f"[ERROR] env step failed at step={step}: {e}", flush=True)
                reward = 0.0
                done   = True

            rewards.append(reward)
            steps_taken = step
            history.append(f"Step {step}: {message[:120]} -> reward {reward:.4f}")
            print_step(step=step, reward=reward, done=done)

            if done:
                break

        max_total = MAX_TOTAL_REWARD_PER_TASK.get(task_name, 1.0)
        raw_score = sum(rewards)
        score = min(max(raw_score / max_total if max_total > 0 else 0.01, 0.01), 0.99)
        try:
            await env.close()
        except Exception:
            pass

    print_end(task=task_name, score=score, steps=steps_taken)

    return {
        "task":    task_name,
        "score":   round(score, 4),
        "success": score >= SUCCESS_THRESHOLD,
        "steps":   steps_taken,
        "rewards": rewards,
    }

# ─── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    # Initialise OpenAI client pointing to the validator's LiteLLM proxy
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[INFO] SERVER_URL={SERVER_URL}", flush=True)

    results = []
    for task in TASKS_TO_RUN:
        result = await run_task(client=client, env_url=SERVER_URL, task_name=task)
        results.append(result)

    total_score = sum(r["score"] for r in results)
    avg = total_score / len(results) if results else 0.0
    print(f"[SUMMARY] avg_score={round(avg, 4)} tasks={len(results)}", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({"tasks": results, "avg_score": round(avg, 4)}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
