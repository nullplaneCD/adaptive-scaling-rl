from env.task_env import TaskSchedulingEnv
from agents.ddqn import DDQNAgent
from utils.logger import get_logger
from utils.config import cfg
import torch
import random
import numpy as np

log = get_logger(__name__)


def train(seed=0):
    log.info("ProScale DDQN training started | seed=%d", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = TaskSchedulingEnv()
    env.reset(seed=seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    episodes   = cfg["training"]["episodes"]
    log_freq   = cfg["training"]["log_freq"]
    results_dir = cfg["training"]["results_dir"]

    log.info("Environment | state_dim=%d  action_dim=%d  episodes=%d",
             state_dim, action_dim, episodes)

    agent   = DDQNAgent(state_dim, action_dim)
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(cfg["environment"]["max_steps"]):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            agent.decay_epsilon()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        avg10 = sum(rewards[-10:]) / len(rewards[-10:])

        np.save(f"{results_dir}/rewards_seed{seed}.npy", rewards)
        np.save(f"{results_dir}/rewards.npy", rewards)

        if episode % log_freq == 0:
            log.info("Episode %4d | reward=%8.2f | avg10=%8.2f | epsilon=%.4f",
                     episode, total_reward, avg10, agent.epsilon)

    log.info("Training complete | seed=%d | final_avg10=%.2f", seed, avg10)


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["training"]["seed"]
    train(seed=seed)
