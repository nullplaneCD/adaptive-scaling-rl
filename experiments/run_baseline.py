from env.task_env import TaskSchedulingEnv
import numpy as np

def run_random_baseline(n_episodes=500):
    env = TaskSchedulingEnv()
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = env.action_space.sample()  # pure random
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)

    print(f"Random baseline over {n_episodes} episodes:")
    print(f"  Mean reward : {np.mean(rewards):.2f}")
    print(f"  Std  reward : {np.std(rewards):.2f}")
    print(f"  Min  reward : {np.min(rewards):.2f}")
    print(f"  Max  reward : {np.max(rewards):.2f}")

if __name__ == "__main__":
    run_random_baseline()
