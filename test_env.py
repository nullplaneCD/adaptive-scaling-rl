from env.task_env import TaskSchedulingEnv
import random

env = TaskSchedulingEnv()
obs, info = env.reset()

print("obs shape:", obs.shape)
print("initial info:", info)

for step in range(10):
    action = random.randint(0, env.action_space.n - 1)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={step}, action={action}, reward={reward:.2f}, info={info}")

    if terminated or truncated:
        break