from env.task_env import TaskSchedulingEnv


def threshold_policy(env: TaskSchedulingEnv):
    queue_len = len(env.queue)
    idle_cpu = env.available_cpu

    # action定义
    scale_up = env.max_queue + 1
    scale_down = env.max_queue + 2

    # 规则
    if queue_len > 3:
        return scale_up
    elif idle_cpu > 3:
        return scale_down
    elif queue_len > 0:
        return 1  # 调度第一个任务
    else:
        return 0  # do nothing


def run_threshold():
    env = TaskSchedulingEnv()
    obs, info = env.reset()

    total_reward = 0

    for step in range(100):
        action = threshold_policy(env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(f"step={step}, action={action}, reward={reward:.2f}")

        if terminated or truncated:
            break

    print("TOTAL REWARD:", total_reward)


if __name__ == "__main__":
    run_threshold()