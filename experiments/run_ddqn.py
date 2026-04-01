from env.task_env import TaskSchedulingEnv
from agents.ddqn import DDQNAgent
import numpy as np


def train():
    env = TaskSchedulingEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)
    rewards = []

    for episode in range(1000):
        state, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
            
        rewards.append(total_reward)
        avg10 = sum(rewards[-10:]) / len(rewards[-10:])
        agent.decay_epsilon()
        print(f"Episode {episode}, reward={total_reward:.2f}, avg10={avg10:.2f}, epsilon={agent.epsilon:.3f}")
        np.save("results/rewards.npy", rewards)
        print("Saved reward to results/rewards.npy")

if __name__ == "__main__":
    train()