from env.task_env import TaskSchedulingEnv


def fifo_policy(env: TaskSchedulingEnv):
    queue_len = len(env.queue)

    scale_up = env.max_queue + 1

    # 如果队列很长 → 扩容
    if queue_len > 3:
        return scale_up

    # 有任务就调度第一个
    if queue_len > 0:
        return 1

    return 0


def run_fifo():
    env = TaskSchedulingEnv()
    obs, info = env.reset()

    total_reward = 0

    for step in range(100):
        action = fifo_policy(env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(f"step={step}, action={action}, reward={reward:.2f}")

        if terminated or truncated:
            break

    print("TOTAL REWARD:", total_reward)


if __name__ == "__main__":
    run_fifo()