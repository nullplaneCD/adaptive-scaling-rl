from env.task_env import TaskSchedulingEnv
from agents.ddqn import DDQNAgent
import torch
import random
import numpy as np


def train(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = TaskSchedulingEnv()
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)
    rewards = []

    for episode in range(5000):
        state, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            agent.decay_epsilon()

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        avg10 = sum(rewards[-10:]) / len(rewards[-10:])
        np.save(f"results/rewards_seed{seed}.npy", rewards)
        np.save("results/rewards.npy", rewards)
        if episode % 100 == 0:
            print(f"Episode {episode}, reward={total_reward:.2f}, avg10={avg10:.2f}, epsilon={agent.epsilon:.3f}")

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train(seed=seed)
