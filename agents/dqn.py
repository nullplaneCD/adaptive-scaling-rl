import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 0.1

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.model.net[-1].out_features - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return q_values.argmax().item()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        target = q_values.clone().detach()
        target[action] = reward + self.gamma * torch.max(next_q_values) * (1 - done)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()