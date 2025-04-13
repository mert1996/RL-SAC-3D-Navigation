import time

import torch
from gym_environment import MyPyBulletEnvContinuous
from networks.ActorNet import ActorNet


def test_agent(episode_number: int):
    model_path = f"trained/actor/actor_ep{episode_number}.pt"

    env = MyPyBulletEnvContinuous(render=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = ActorNet(obs_dim, act_dim)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        time.sleep(0.05)
        with torch.no_grad():
            mu, _ = actor(state_tensor)
            action = torch.tanh(mu).squeeze(0).numpy() * max_action
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Episode {episode_number+1} finished. Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    test_agent(episode_number=600)
