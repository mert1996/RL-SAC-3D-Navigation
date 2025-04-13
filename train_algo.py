from checkpoints import load_checkpoint, save_checkpoint

from networks.ReplayBuffer.ReplayBuffer import ReplayBuffer
from networks.SACAgent import SACAgent


def train_episodes(n_episodes=100, max_steps=500):
    from gym_environment import MyPyBulletEnvContinuous

    env = MyPyBulletEnvContinuous(render=False)
    obs = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action, gamma=0.99,
                     tau=0.005, alpha=0.2, automatic_entropy_tuning=True)

    replay_buffer, start_episode, stats_data = load_checkpoint(agent)
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(capacity=100_000)

    warmup_steps = 1000
    batch_size = 64

    for ep_offset in range(n_episodes):
        episode = start_episode + ep_offset + 1
        state = env.reset()
        episode_reward = 0
        actor_loss_sum = 0
        actor_loss_count = 0

        for step in range(max_steps):
            if replay_buffer.size() < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            replay_buffer.store(state, action, reward, next_state, done)
            state = next_state

            if replay_buffer.size() >= warmup_steps:
                c1_loss, c2_loss, actor_loss, alpha_loss = agent.train(replay_buffer, batch_size)
                actor_loss_sum += actor_loss
                actor_loss_count += 1

            if step == max_steps - 1:
                done = True

            if done:
                break

        avg_actor_loss = actor_loss_sum / actor_loss_count if actor_loss_count > 0 else 0
        stats_data["episodes"] = episode
        stats_data["rewards"].append(float(episode_reward))
        stats_data["actor_losses"].append(float(avg_actor_loss))

        print(f"Episode {episode}: Total Reward: {episode_reward:.2f}")

        save_checkpoint(agent, replay_buffer, stats_data, episode)

    env.close()
    return stats_data


if __name__ == "__main__":
    train_episodes(n_episodes=100)
