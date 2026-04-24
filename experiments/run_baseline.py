from env.task_env import TaskSchedulingEnv
from utils.logger import get_logger
from utils.config import cfg
import numpy as np

log = get_logger(__name__)


def run_random_baseline(n_episodes=None):
    n_episodes = n_episodes or cfg["evaluation"]["baseline_episodes"]
    log.info("Random baseline evaluation started | episodes=%d", n_episodes)

    env     = TaskSchedulingEnv()
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(cfg["environment"]["max_steps"]):
            action = env.action_space.sample()  # pure random
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    rewards = np.array(rewards)
    log.info("Random baseline | episodes=%d | mean=%.2f | std=%.2f | min=%.2f | max=%.2f",
             n_episodes, rewards.mean(), rewards.std(), rewards.min(), rewards.max())
    return rewards


if __name__ == "__main__":
    run_random_baseline()
