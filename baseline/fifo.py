import numpy as np
from env.task_env import TaskSchedulingEnv


def fifo_policy(env: TaskSchedulingEnv):
    """Schedule the first affordable task in queue (FIFO order), no scaling."""
    for i, task in enumerate(env.queue):
        if task["cpu_required"] <= env.available_cpu:
            return i + 1  # action = queue index + 1
    return 0  # no-op if nothing affordable


def run_fifo(n_episodes=500):
    env = TaskSchedulingEnv()
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = fifo_policy(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    rewards = np.array(rewards)
    print(f"FIFO baseline over {n_episodes} episodes:")
    print(f"  Mean reward : {np.mean(rewards):.2f}")
    print(f"  Std  reward : {np.std(rewards):.2f}")
    print(f"  Min  reward : {np.min(rewards):.2f}")
    print(f"  Max  reward : {np.max(rewards):.2f}")
    return rewards


if __name__ == "__main__":
    run_fifo()
