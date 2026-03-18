from env.task_env import TaskSchedulingEnv
from agents.dqn import DQNAgent


def train():
    env = TaskSchedulingEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.train_step(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Episode {episode}, reward={total_reward}")


if __name__ == "__main__":
    train()