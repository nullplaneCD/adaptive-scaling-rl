"""
Systematic inference grid test for ProScale DDQN agent.
Samples real environment states to guarantee consistent observation vectors.
"""

import requests
import numpy as np
from env.task_env import TaskSchedulingEnv

URL = "http://localhost:8000/predict"

def obs_to_payload(obs, arrival_prob):
    """Convert env observation vector to API payload."""
    queue_tasks = []
    for i in range(5):
        slot = obs[6 + i*4 : 6 + i*4 + 4].tolist()
        if any(v > 0 for v in slot):
            queue_tasks.append(slot)
    return {
        "total_cpu":      float(obs[0]),
        "available_cpu":  float(obs[1]),
        "running_tasks":  float(obs[2]),
        "queue_length":   float(obs[3]),
        "arrival_prob":   float(obs[4]),
        "phase_progress": float(obs[5]),
        "queue_tasks":    queue_tasks,
    }

def collect_state(phase, min_queue=0, exact_queue=None, label=""):
    """Step through env until a matching state is found."""
    env = TaskSchedulingEnv()
    obs, _ = env.reset(seed=42)
    for _ in range(500):
        action = 0  # no-op to let states accumulate
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
        current_phase = (env.steps // env.phase_length) % 2
        q = len(env.queue)
        queue_ok = (q == exact_queue) if exact_queue is not None else (q >= min_queue)
        if current_phase == phase and queue_ok:
            return obs, env._get_arrival_prob()
    return obs, env._get_arrival_prob()  # fallback

print(f"\n{'─'*95}")
print(f"{'State':<28} {'Action':>4}  {'Action Name':<16} {'Conf':>7}  {'Expected':<18} {'Pass?'}")
print(f"{'─'*95}")

CASES = [
    # (label,                  phase, min_q, exact_q, expected)
    ("quiet | empty queue",   0, 0, 0,    "no-op"),
    ("quiet | tasks waiting", 0, 2, None, "schedule"),
    ("burst | empty queue",   1, 0, 0,    "no-op/scale-down"),
    ("burst | tasks waiting", 1, 2, None, "schedule"),
]

for label, phase, min_queue, exact_queue, expected in CASES:
    obs, arrival_prob = collect_state(phase, min_queue, exact_queue, label)
    payload = obs_to_payload(obs, arrival_prob)
    resp = requests.post(URL, json=payload).json()

    action_name = resp["action_name"]
    confidence  = resp["confidence"]
    action      = resp["action"]
    passed = any(e in action_name for e in expected.split("/"))
    mark = "✅" if passed else "❌"

    print(f"{label:<28} {action:>4}  {action_name:<16} {confidence:>7.4f}  {expected:<18} {mark}")

print(f"{'─'*95}\n")
