import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Replaybuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return(
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

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=5e-4)
        self.loss_fn = nn.MSELoss()

        self.train_steps = 0
        self.target_update_freq = 100

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        self.replay_buffer = Replaybuffer(50000)
        self.batch_size = 64

        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.online_net(state)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        q_values = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            # --- DDQN target ---
            # 1. online net picks the best action for next state
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # 2. target net evaluates that action (decouples selection from evaluation)
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps > 500:
            self.decay_epsilon()
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
       