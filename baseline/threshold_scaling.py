import numpy as np
from env.task_env import TaskSchedulingEnv
from utils.logger import get_logger
from utils.config import cfg

log = get_logger(__name__)


def threshold_policy(env: TaskSchedulingEnv):
    """Kubernetes HPA-equivalent reactive scaling policy.

    Scales CPU up when queue length exceeds a fixed threshold,
    scales down when idle CPU is excessive, otherwise schedules
    the first affordable task. No workload pattern learning.

    This is the prevailing commercial approach (mirrors K8s HPA behaviour).
    """
    queue_len = len(env.queue)
    idle_cpu  = env.available_cpu

    scale_up   = env.max_queue + 1
    scale_down = env.max_queue + 2

    if queue_len > 3:           # scale up reactively when queue is long
        return scale_up
    elif idle_cpu > 3:          # scale down when too many idle CPUs
        return scale_down
    elif queue_len > 0:         # schedule first task if affordable
        task = env.queue[0]
        if task["cpu_required"] <= env.available_cpu:
            return 1
    return 0                    # no-op


def run_threshold(n_episodes=None):
    n_episodes = n_episodes or cfg["evaluation"]["baseline_episodes"]
    log.info("Threshold (HPA-equivalent) baseline evaluation started | episodes=%d", n_episodes)

    env     = TaskSchedulingEnv()
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(cfg["environment"]["max_steps"]):
            action = threshold_policy(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    rewards = np.array(rewards)
    log.info("Threshold (HPA-equivalent) baseline | episodes=%d | mean=%.2f | std=%.2f | min=%.2f | max=%.2f",
             n_episodes, rewards.mean(), rewards.std(), rewards.min(), rewards.max())
    return rewards


if __name__ == "__main__":
    run_threshold()
