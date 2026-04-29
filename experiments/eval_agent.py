"""
Episode-level evaluation of trained DDQN agent (epsilon=0, pure exploitation).
Reports mean and std reward over N episodes.
"""

import torch
import numpy as np
import random
from agents.ddqn import DDQNAgent
from env.task_env import TaskSchedulingEnv
from utils.config import cfg

EPISODES = 500
SEED     = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env   = TaskSchedulingEnv()
agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.epsilon = 0.0  # pure exploitation

ckpt = "results/agent_checkpoint.pt"
agent.online_net.load_state_dict(torch.load(ckpt, map_location="cpu"))
print(f"Checkpoint loaded: {ckpt}")

rewards = []
for ep in range(EPISODES):
    state, _ = env.reset(seed=SEED + ep)
    total = 0
    for _ in range(cfg["environment"]["max_steps"]):
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    rewards.append(total)

mean_r = np.mean(rewards)
std_r  = np.std(rewards)
print(f"\n── DDQN Agent Evaluation (epsilon=0, {EPISODES} episodes) ──")
print(f"  Mean reward : {mean_r:.1f}")
print(f"  Std  reward : {std_r:.1f}")
print(f"  Min  reward : {np.min(rewards):.1f}")
print(f"  Max  reward : {np.max(rewards):.1f}")
print(f"\n  FIFO baseline  : 3872")
print(f"  Random baseline: 1443")
print(f"  Improvement vs FIFO: {(mean_r - 3872) / 3872 * 100:.1f}%")
