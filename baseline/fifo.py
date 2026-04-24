import numpy as np
from env.task_env import TaskSchedulingEnv
from utils.logger import get_logger
from utils.config import cfg

log = get_logger(__name__)


def fifo_policy(env: TaskSchedulingEnv):
    """Schedule the first affordable task in queue (FIFO order), no scaling.
    Functionally equivalent to standard queue management used in industry schedulers.
    """
    for i, task in enumerate(env.queue):
        if task["cpu_required"] <= env.available_cpu:
            return i + 1  # action = queue index + 1
    return 0  # no-op if nothing affordable


def run_fifo(n_episodes=None):
    n_episodes = n_episodes or cfg["evaluation"]["baseline_episodes"]
    log.info("FIFO baseline evaluation started | episodes=%d", n_episodes)

    env     = TaskSchedulingEnv()
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(cfg["environment"]["max_steps"]):
            action = fifo_policy(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    rewards = np.array(rewards)
    log.info("FIFO baseline | episodes=%d | mean=%.2f | std=%.2f | min=%.2f | max=%.2f",
             n_episodes, rewards.mean(), rewards.std(), rewards.min(), rewards.max())
    return rewards


if __name__ == "__main__":
    run_fifo()
