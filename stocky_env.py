"""
stocky_env.py — Python client for the Stocky OpenEnv.

Usage:
    env = await StockyEnv.from_docker_image("your-hf-space-url-or-localhost:7860")
    result = await env.reset(task="identify_top_stock")
    result = await env.step(StockyAction(message="I recommend TCS.NS because..."))
    await env.close()
"""

import asyncio
import subprocess
import time
import os
from typing import Optional, List, Dict, Any

import httpx
from pydantic import BaseModel

# ─── Models (mirrors server models) ──────────────────────────────────────────

class StockyAction(BaseModel):
    message: str

class StockyObservation(BaseModel):
    echoed_message: str
    market_snapshot: Dict[str, Any]
    news_headlines: List[str]
    agent_feedback: str
    task_name: str
    task_description: str
    step_count: int
    max_steps: int
    reward_breakdown: Dict[str, float]
    history_summary: str
    valid_tickers: List[str]

class StepResult(BaseModel):
    observation: StockyObservation
    reward: float
    done: bool
    info: Dict[str, Any]

# ─── Client Class ─────────────────────────────────────────────────────────────

class StockyEnv:
    """
    OpenEnv-compatible client for the Stocky Indian Stock Market environment.

    Supports two modes:
    1. Remote: connect to a running HuggingFace Space or server URL
    2. Docker: spin up a local container (via from_docker_image)
    """

    def __init__(self, base_url: str, session_id: Optional[str] = None,
                 task: str = "full_pipeline_recommendation"):
        self.base_url   = base_url.rstrip("/")
        self.session_id = session_id
        self.task       = task
        self._client    = httpx.AsyncClient(timeout=30.0)
        self._container = None  # set by from_docker_image

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        task: str = "full_pipeline_recommendation",
        port: int = 7860,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> "StockyEnv":
        """
        Pull and run the Docker image, wait for it to be ready, return a connected client.

        Args:
            image_name: Docker image (e.g., 'your-username/stocky-openenv:latest'
                        or HF Space URL like 'https://your-space.hf.space')
            task: Which task to run
            port: Local port to map to container's 7860
            env_vars: Extra environment variables for the container
        """
        # If it's a URL (HF Space), connect directly without spinning up Docker
        if image_name.startswith("http"):
            instance = cls(base_url=image_name, task=task)
            await instance._wait_for_ready()
            return instance

        # Otherwise, run Docker container locally
        env_args = []
        if env_vars:
            for k, v in env_vars.items():
                env_args += ["-e", f"{k}={v}"]

        cmd = [
            "docker", "run", "-d", "--rm",
            "-p", f"{port}:7860",
            *env_args,
            image_name,
        ]
        print(f"[StockyEnv] Starting container: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        container_id = result.stdout.strip()
        print(f"[StockyEnv] Container started: {container_id[:12]}")

        instance = cls(base_url=f"http://localhost:{port}", task=task)
        instance._container = container_id
        await instance._wait_for_ready(max_retries=30, delay=2.0)
        return instance

    async def _wait_for_ready(self, max_retries: int = 15, delay: float = 2.0):
        """Poll /health until the server is up."""
        for attempt in range(max_retries):
            try:
                resp = await self._client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    print(f"[StockyEnv] Server ready at {self.base_url}")
                    return
            except Exception:
                pass
            print(f"[StockyEnv] Waiting for server... ({attempt+1}/{max_retries})")
            await asyncio.sleep(delay)
        raise TimeoutError(f"Server at {self.base_url} did not become ready.")

    async def reset(self, task: Optional[str] = None, seed: int = 42) -> StepResult:
        """Reset the environment and start a new episode."""
        chosen_task = task or self.task
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task": chosen_task, "seed": seed},
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["info"]["session_id"]
        return StepResult(**data)

    async def step(self, action: StockyAction) -> StepResult:
        """Take one action in the environment."""
        if not self.session_id:
            raise RuntimeError("No active session. Call reset() first.")
        resp = await self._client.post(
            f"{self.base_url}/step",
            json={"session_id": self.session_id, "action": {"message": action.message}},
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    async def state(self) -> Dict:
        """Get current state of the environment."""
        if not self.session_id:
            raise RuntimeError("No active session.")
        resp = await self._client.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        """Close the session and stop the container if running locally."""
        if self.session_id:
            try:
                await self._client.delete(
                    f"{self.base_url}/close",
                    params={"session_id": self.session_id},
                )
            except Exception:
                pass
        await self._client.aclose()

        if self._container:
            subprocess.run(["docker", "stop", self._container],
                           capture_output=True, check=False)
            print(f"[StockyEnv] Container {self._container[:12]} stopped.")
