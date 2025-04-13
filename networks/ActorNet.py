import torch
from torch import nn
import numpy as np


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)       # action average
        self.log_std_head = nn.Linear(hidden_dim, action_dim)  # logarithmic standard deviation

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mu)  # random normal distribution epsilon value
        action = mu + eps * std
        action_tanh = torch.tanh(action)
        log_prob = (-0.5 * ((action - mu) / (std + 1e-6)).pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(dim=1,
                                                                                                          keepdim=True)
        log_prob -= torch.sum(torch.log(1 - action_tanh.pow(2) + 1e-6), dim=1, keepdim=True)
        return action_tanh, log_prob
