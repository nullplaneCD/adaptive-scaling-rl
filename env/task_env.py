import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TaskSchedulingEnv(gym.Env):
    """
    CPU-core based scheduling + scaling environment.

    Actions:
        0                -> do nothing
        1..max_queue     -> schedule task at queue index (action - 1)
        max_queue + 1    -> scale up by 1 CPU core
        max_queue + 2    -> scale down by 1 CPU core
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_queue: int = 5,
        min_cpu: int = 2,
        max_cpu: int = 16,
        init_cpu: int = 8,
        max_running: int = 16,
        max_steps: int = 200,
        arrival_prob: float = 0.7,
        scale_up_cost: float = 0.5,
        scale_down_cost: float = 0.2,
    ):
        super().__init__()

        assert min_cpu <= init_cpu <= max_cpu

        self.max_queue = max_queue
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.init_cpu = init_cpu
        self.max_running = max_running
        self.max_steps = max_steps
        self.arrival_prob = arrival_prob
        self.scale_up_cost = scale_up_cost
        self.scale_down_cost = scale_down_cost

        # 0 = no-op
        # 1..max_queue = schedule one queued task
        # max_queue + 1 = scale up
        # max_queue + 2 = scale down
        self.action_space = spaces.Discrete(self.max_queue + 3)

        # state:
        # [total_cpu_norm, available_cpu_norm, running_count_norm, queue_len_norm]
        # + queued tasks: [cpu_required, duration, priority, deadline] * max_queue
        obs_dim = 4 + self.max_queue * 4
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.steps = 0
        self.total_cpu = self.init_cpu
        self.available_cpu = self.init_cpu
        self.running_tasks = []
        self.queue = []
        self.completed_tasks = 0
        self.failed_tasks = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.total_cpu = self.init_cpu
        self.available_cpu = self.init_cpu
        self.running_tasks = []
        self.queue = []
        self.completed_tasks = 0
        self.failed_tasks = 0

        for _ in range(3):
            self._maybe_add_task(force=True)

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        self.steps += 1
        reward = 0.0

        # 1. Update running tasks
        finished = []
        for task in self.running_tasks:
            task["remaining"] -= 1
            if task["remaining"] <= 0:
                finished.append(task)

        for task in finished:
            self.running_tasks.remove(task)
            self.available_cpu += task["cpu_required"]
            self.completed_tasks += 1
            reward += 10.0 * task["priority"]

        # 2. Update waiting tasks
        expired = []
        for task in self.queue:
            task["deadline"] -= 1
            if task["deadline"] <= 0:
                expired.append(task)

        for task in expired:
            self.queue.remove(task)
            self.failed_tasks += 1
            reward -= 8.0

        # 3. Apply action
        scale_up_action = self.max_queue + 1
        scale_down_action = self.max_queue + 2

        if 1 <= action <= self.max_queue:
            idx = action - 1
            if idx < len(self.queue):
                task = self.queue[idx]
                if task["cpu_required"] <= self.available_cpu:
                    self.available_cpu -= task["cpu_required"]
                    task["remaining"] = task["duration"]
                    self.running_tasks.append(task)
                    self.queue.remove(task)
                else:
                    reward -= 2.0
            else:
                reward -= 1.0

        elif action == scale_up_action:
            if self.total_cpu < self.max_cpu:
                self.total_cpu += 1
                self.available_cpu += 1
                reward -= self.scale_up_cost
            else:
                reward -= 1.0

        elif action == scale_down_action:
            # only allow scale down if there is at least 1 idle CPU
            if self.total_cpu > self.min_cpu and self.available_cpu >= 1:
                self.total_cpu -= 1
                self.available_cpu -= 1
                reward -= self.scale_down_cost
            else:
                reward -= 2.0

        # 4. Ongoing penalties
        reward -= 0.2 * len(self.queue)      # waiting penalty
        reward -= 0.05 * self.available_cpu  # idle CPU penalty

        # 5. New arrivals
        self._maybe_add_task(force=False)
        self._maybe_add_task(force=False)

        terminated = self.steps >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _maybe_add_task(self, force=False):
        if force or random.random() < self.arrival_prob:
            if len(self.queue) < self.max_queue:
                task = {
                    "cpu_required": random.randint(1, 4),
                    "duration": random.randint(1, 6),
                    "priority": random.randint(1, 3),
                    "deadline": random.randint(4, 12),
                }
                self.queue.append(task)

    def _get_obs(self):
        obs = [
            self.total_cpu / self.max_cpu,
            self.available_cpu / self.max_cpu,
            len(self.running_tasks) / max(1, self.max_running),
            len(self.queue) / self.max_queue,
        ]

        for i in range(self.max_queue):
            if i < len(self.queue):
                task = self.queue[i]
                obs.extend([
                    task["cpu_required"] / 4.0,
                    task["duration"] / 6.0,
                    task["priority"] / 3.0,
                    task["deadline"] / 12.0,
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        used_cpu = self.total_cpu - self.available_cpu
        return {
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "queue_length": len(self.queue),
            "running_tasks": len(self.running_tasks),
            "total_cpu": self.total_cpu,
            "available_cpu": self.available_cpu,
            "used_cpu": used_cpu,
        }