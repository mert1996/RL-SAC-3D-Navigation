import os
from collections import deque
import torch
from networks.ReplayBuffer.ReplayBuffer import ReplayBuffer


def save_checkpoint(agent, replay_buffer, stats_data, episode, path="checkpoints"):
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        # networks
        "actor_state": agent.actor.state_dict(),
        "critic1_state": agent.critic1.state_dict(),
        "critic2_state": agent.critic2.state_dict(),
        "critic1_target_state": agent.critic1_target.state_dict(),
        "critic2_target_state": agent.critic2_target.state_dict(),
        # optimizers
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic1_optimizer": agent.critic1_optimizer.state_dict(),
        "critic2_optimizer": agent.critic2_optimizer.state_dict(),
        # replay buffer
        "replay_buffer": list(replay_buffer.buffer),
        "replay_buffer_capacity": replay_buffer.buffer.maxlen,
        # training statistics
        "stats_data": stats_data,
        "episode": episode,
    }

    if agent.automatic_entropy_tuning:
        checkpoint["log_alpha"] = agent.log_alpha.detach().cpu()
        checkpoint["alpha_optimizer"] = agent.alpha_optimizer.state_dict()

    torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))


def load_checkpoint(agent, path="checkpoints/checkpoint.pt"):
    if not os.path.exists(path):
        return ReplayBuffer(capacity=100_000), 0, {"episodes": 0, "rewards": [], "actor_losses": []}

    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)

    # network parameters
    agent.actor.load_state_dict(checkpoint["actor_state"])
    agent.critic1.load_state_dict(checkpoint["critic1_state"])
    agent.critic2.load_state_dict(checkpoint["critic2_state"])
    agent.critic1_target.load_state_dict(checkpoint["critic1_target_state"])
    agent.critic2_target.load_state_dict(checkpoint["critic2_target_state"])

    # optimizer steps
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    agent.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
    agent.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

    # alpha / entropy
    if agent.automatic_entropy_tuning and "log_alpha" in checkpoint:
        agent.log_alpha = checkpoint["log_alpha"].to(agent.device).requires_grad_()
        agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        agent.alpha = agent.log_alpha.exp().item()

    # replay buffer
    capacity = checkpoint.get("replay_buffer_capacity", 100_000)
    replay_buffer = ReplayBuffer(capacity=capacity)
    replay_buffer.buffer = deque(checkpoint["replay_buffer"], maxlen=capacity)

    return replay_buffer, checkpoint["episode"], checkpoint["stats_data"]
