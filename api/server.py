"""
ProScale — REST API Server
--------------------------
Exposes the trained DDQN agent as a REST service.

Endpoints:
  GET  /                    Health check
  GET  /status              Agent and environment status
  POST /predict             Get scaling recommendation for current cluster state
  POST /train               Launch a background training run
  GET  /results             Latest training metrics

Start the server:
  uvicorn api.server:app --reload --port 8000

Interactive docs (auto-generated):
  http://localhost:8000/docs
"""

import os
import time
import threading
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.config import cfg

log = get_logger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ProScale API",
    description=(
        "Proactive cloud resource scaling via reinforcement learning. "
        "ProScale learns workload patterns and recommends scaling actions "
        "before demand peaks arrive — not after."
    ),
    version="1.0.0",
)

# ── Global agent state ─────────────────────────────────────────────────────────

_agent       = None          # Loaded DDQNAgent instance
_agent_lock  = threading.Lock()
_train_status = {
    "running":  False,
    "seed":     None,
    "episode":  0,
    "episodes": cfg["training"]["episodes"],
    "started_at": None,
}


def _load_agent():
    """Lazy-load the trained agent from the latest checkpoint."""
    global _agent
    from agents.ddqn import DDQNAgent
    from env.task_env import TaskSchedulingEnv

    env        = TaskSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)
    agent.epsilon = 0.0   # no exploration during inference

    # Load weights if a checkpoint exists
    ckpt_path = Path("results/agent_checkpoint.pt")
    if ckpt_path.exists():
        import torch
        agent.online_net.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        log.info("Agent checkpoint loaded from %s", ckpt_path)
    else:
        log.warning("No checkpoint found at %s — agent will return untrained predictions", ckpt_path)

    _agent = agent
    return _agent


# ── Request / Response models ──────────────────────────────────────────────────

class ClusterState(BaseModel):
    """Current state of the cluster, as observed by the ProScale agent."""
    total_cpu:       float = Field(..., ge=0, le=1, description="Total CPU cores (normalised 0–1)")
    available_cpu:   float = Field(..., ge=0, le=1, description="Available CPU cores (normalised 0–1)")
    running_tasks:   float = Field(..., ge=0, le=1, description="Running task count (normalised 0–1)")
    queue_length:    float = Field(..., ge=0, le=1, description="Queue occupancy (normalised 0–1)")
    arrival_prob:    float = Field(..., ge=0, le=1, description="Current workload arrival probability (0.3=quiet, 0.9=burst)")
    phase_progress:  float = Field(..., ge=0, le=1, description="Progress within current workload phase (0=start, 1=end)")
    queue_tasks:     List[List[float]] = Field(
        default=[],
        description="Up to 5 queued tasks, each as [cpu_norm, duration_norm, priority_norm, deadline_norm]",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_cpu":      0.5,
                "available_cpu":  0.5,
                "running_tasks":  0.0,
                "queue_length":   0.0,
                "arrival_prob":   0.3,
                "phase_progress": 0.0,
                "queue_tasks":    [],
            }
        }


class PredictResponse(BaseModel):
    action:      int    = Field(..., description="Recommended action (0=no-op, 1-5=schedule task, 6=scale up, 7=scale down)")
    action_name: str    = Field(..., description="Human-readable action label")
    confidence:  float  = Field(..., description="Q-value gap between top-2 actions (higher = more confident)")
    phase:       str    = Field(..., description="Detected workload phase: burst or quiet")
    note:        str    = Field(..., description="Explanation of the recommendation")


class TrainRequest(BaseModel):
    seed: int = Field(default=0, description="Random seed for this training run")


class TrainResponse(BaseModel):
    status:  str
    seed:    int
    message: str


class StatusResponse(BaseModel):
    agent_loaded:   bool
    training:       bool
    train_episode:  int
    train_episodes: int
    uptime_seconds: float


# ── Action label helper ────────────────────────────────────────────────────────

_start_time = time.time()

def _action_label(action: int) -> str:
    if action == 0:
        return "no-op"
    if 1 <= action <= cfg["environment"]["max_queue"]:
        return f"schedule task {action}"
    if action == cfg["environment"]["max_queue"] + 1:
        return "scale up"
    if action == cfg["environment"]["max_queue"] + 2:
        return "scale down"
    return "unknown"


# ── Background training ────────────────────────────────────────────────────────

def _background_train(seed: int):
    """Run a full training loop in a background thread."""
    import torch, random
    from env.task_env import TaskSchedulingEnv
    from agents.ddqn import DDQNAgent

    global _agent, _train_status

    _train_status.update({"running": True, "seed": seed, "episode": 0,
                           "started_at": time.time()})
    log.info("Background training started | seed=%d", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env        = TaskSchedulingEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    episodes   = cfg["training"]["episodes"]

    agent   = DDQNAgent(state_dim, action_dim)
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(cfg["environment"]["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            agent.decay_epsilon()
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        _train_status["episode"] = episode

        if episode % cfg["training"]["log_freq"] == 0:
            avg10 = np.mean(rewards[-10:])
            log.info("Training episode %4d | reward=%8.2f | avg10=%8.2f | epsilon=%.4f",
                     episode, total_reward, avg10, agent.epsilon)

    # Save checkpoint and results
    Path("results").mkdir(exist_ok=True)
    torch.save(agent.online_net.state_dict(), "results/agent_checkpoint.pt")
    np.save(f"results/rewards_seed{seed}.npy", rewards)
    np.save("results/rewards.npy", rewards)

    with _agent_lock:
        _agent = agent
        _agent.epsilon = 0.0

    _train_status["running"] = False
    log.info("Background training complete | seed=%d | final_avg10=%.2f",
             seed, np.mean(rewards[-10:]))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    """Health check — confirms the ProScale API is running."""
    return {"service": "ProScale", "status": "ok", "version": "1.0.0"}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def status():
    """Returns current agent load state and training progress."""
    return StatusResponse(
        agent_loaded   = _agent is not None,
        training       = _train_status["running"],
        train_episode  = _train_status["episode"],
        train_episodes = _train_status["episodes"],
        uptime_seconds = round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(state: ClusterState):
    """
    Get a proactive scaling recommendation for the current cluster state.

    The agent uses the arrival_prob and phase_progress fields to anticipate
    upcoming workload changes and recommend pre-emptive scaling actions.
    """
    import torch

    with _agent_lock:
        agent = _agent if _agent is not None else _load_agent()

    # Build observation vector (matches env/_get_obs() layout)
    max_queue = cfg["environment"]["max_queue"]
    obs = [
        state.total_cpu,
        state.available_cpu,
        state.running_tasks,
        state.queue_length,
        state.arrival_prob,
        state.phase_progress,
    ]
    for i in range(max_queue):
        if i < len(state.queue_tasks):
            obs.extend(state.queue_tasks[i][:4])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.online_net(obs_tensor).squeeze(0)

    action     = int(q_values.argmax().item())
    sorted_q   = q_values.sort(descending=True).values
    confidence = float(sorted_q[0] - sorted_q[1])
    phase      = "burst" if state.arrival_prob >= 0.6 else "quiet"

    label = _action_label(action)
    note  = (
        f"Agent recommends '{label}' based on {phase} phase "
        f"(arrival_prob={state.arrival_prob:.2f}, phase_progress={state.phase_progress:.2f})."
    )

    log.info("Predict | action=%d (%s) | confidence=%.3f | phase=%s",
             action, label, confidence, phase)

    return PredictResponse(
        action      = action,
        action_name = label,
        confidence  = round(confidence, 4),
        phase       = phase,
        note        = note,
    )


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Launch a DDQN training run in the background.

    Training runs asynchronously — poll GET /status to monitor progress.
    When complete, the agent is hot-swapped and /predict uses the new weights.
    """
    if _train_status["running"]:
        raise HTTPException(status_code=409,
                            detail="A training run is already in progress. "
                                   "Poll GET /status for progress.")

    background_tasks.add_task(_background_train, request.seed)
    log.info("Training task queued | seed=%d", request.seed)

    return TrainResponse(
        status  = "accepted",
        seed    = request.seed,
        message = f"Training started in background (seed={request.seed}). "
                  f"Poll GET /status for progress.",
    )


@app.get("/results", tags=["Training"])
def results():
    """Return the latest training metrics from the most recent run."""
    path = Path("results/rewards.npy")
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="No training results found. Run POST /train first.")

    rewards = np.load(str(path))
    window  = cfg["evaluation"]["plot_window"]
    avg     = [float(np.mean(rewards[max(0, i - window + 1):i + 1]))
               for i in range(len(rewards))]

    return {
        "episodes":      len(rewards),
        "final_reward":  round(float(rewards[-1]), 2),
        "final_avg":     round(avg[-1], 2),
        "mean_all":      round(float(np.mean(rewards)), 2),
        "std_all":       round(float(np.std(rewards)), 2),
        "baselines": {
            "random":    cfg["evaluation"]["random_baseline"],
            "fifo":      cfg["evaluation"]["fifo_baseline"],
            "threshold": cfg["evaluation"]["threshold_baseline"],
        },
    }
