import torch
from torch import nn

from networks.ActorNet import ActorNet
from networks.CriticNet import CriticNet


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        automatic_entropy_tuning=False,
        target_entropy=None,
        lr=3e-4,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = ActorNet(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critics
        self.critic1 = CriticNet(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNet(state_dim, action_dim).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # Target critics
        self.critic1_target = CriticNet(state_dim, action_dim).to(self.device)
        self.critic2_target = CriticNet(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = (
                -float(action_dim) if target_entropy is None else target_entropy
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.max_action = max_action

    # Interaction
    def select_action(self, state, eval_mode=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, log_std = self.actor(state_tensor)
            std = log_std.exp()
            if eval_mode:
                action = torch.tanh(mu)
            else:
                eps = torch.randn_like(mu)
                action = torch.tanh(mu + eps * std)
            return action.cpu().numpy().flatten() * self.max_action

    # Training step
    def train(self, replay_buffer, batch_size=64):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_action)
            target_q2 = self.critic2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_Q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(current_q1, target_Q)
        critic2_loss = nn.MSELoss()(current_q2, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        new_actions, log_prob = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (entropy)
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Target critic soft update
        for param, target_param in zip(
            self.critic1.parameters(), self.critic1_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.critic2.parameters(), self.critic2_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (
            critic1_loss.item(),
            critic2_loss.item(),
            actor_loss.item(),
            alpha_loss.item(),
        )
