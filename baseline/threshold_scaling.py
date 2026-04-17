import numpy as np
from env.task_env import TaskSchedulingEnv


def threshold_policy(env: TaskSchedulingEnv):
    queue_len = len(env.queue)
    idle_cpu = env.available_cpu

    scale_up = env.max_queue + 1
    scale_down = env.max_queue + 2

    # Scale up reactively when queue is long
    if queue_len > 3:
        return scale_up
    # Scale down when too many idle CPUs
    elif idle_cpu > 3:
        return scale_down
    # Schedule first task in queue if affordable
    elif queue_len > 0:
        task = env.queue[0]
        if task["cpu_required"] <= env.available_cpu:
            return 1
    return 0  # no-op


def run_threshold(n_episodes=500):
    env = TaskSchedulingEnv()
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = threshold_policy(env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    rewards = np.array(rewards)
    print(f"Threshold baseline over {n_episodes} episodes:")
    print(f"  Mean reward : {np.mean(rewards):.2f}")
    print(f"  Std  reward : {np.std(rewards):.2f}")
    print(f"  Min  reward : {np.min(rewards):.2f}")
    print(f"  Max  reward : {np.max(rewards):.2f}")
    return rewards


if __name__ == "__main__":
    run_threshold()
