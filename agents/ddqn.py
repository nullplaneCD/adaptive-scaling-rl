import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ── SumTree ────────────────────────────────────────────────────────────────────
class SumTree:
    """Binary tree where each leaf holds a priority; internal nodes hold sums.
    Enables O(log n) priority-weighted sampling.
    Uses 1-indexed style: leaves at indices [capacity, 2*capacity-1].
    """

    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity)   # index 0 unused; leaves at [capacity, 2*capacity-1]
        self.data      = np.zeros(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[1]   # root is at index 1

    def add(self, priority, data):
        idx = self.write + self.capacity   # leaf indices: capacity to 2*capacity-1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx      = self._retrieve(1, s)   # start from root at index 1
        data_idx = idx - self.capacity    # maps leaf index back to data array
        data_idx = max(0, min(data_idx, self.capacity - 1))  # safety clamp
        return idx, self.tree[idx], self.data[data_idx]


# ── Prioritized Replay Buffer ──────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """Experience replay with TD-error based priority sampling.

    Replaces uniform sampling to address burst/quiet phase class imbalance:
    rare but high-error quiet-phase transitions are sampled more frequently,
    correcting the policy failure observed during FastAPI inference testing.
    """

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=1e-5, epsilon=1e-5):
        self.tree            = SumTree(capacity)
        self.capacity        = capacity
        self.alpha           = alpha            # priority exponent (0=uniform, 1=full priority)
        self.beta            = beta             # importance sampling exponent
        self.beta_increment  = beta_increment   # anneal beta → 1.0 over training
        self.epsilon         = epsilon          # small constant to avoid zero priority
        self.max_priority    = 1.0             # new experiences get max priority

    def push(self, state, action, reward, next_state, done):
        """New experiences get maximum priority so they are sampled at least once."""
        self.tree.add(self.max_priority ** self.alpha,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch weighted by priority; return importance sampling weights."""
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Importance sampling weights to correct bias introduced by prioritised sampling
        probs   = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights, dtype=np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-error after each training step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ── Keep uniform buffer as fallback (for ablation comparison) ─────────────────
class Replaybuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
    
class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.online_net = DDQN(state_dim, action_dim)
        self.target_net = DDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss: robust to large reward scale

        self.train_steps = 0
        self.target_update_freq = 500  # spans ~15 burst-quiet cycles to prevent phase-biased Q-value propagation

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999995  # per-step decay: reaches 0.05 at ~300k steps

        self.replay_buffer = Replaybuffer(50000)  # Fix 1: uniform buffer, reward signal change only
        self.batch_size = 64
        self.warmup_steps = 5000  # collect experience before training starts

        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.online_net(state)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.warmup_steps:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        dones       = torch.FloatTensor(dones).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        q_values = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            # --- DDQN target ---
            # 1. online net picks the best action for next state
            best_actions  = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # 2. target net evaluates that action (decouples selection from evaluation)
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            target        = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
       