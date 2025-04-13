# SAC-RL-Navigation

A reinforcement learning project that trains an autonomous agent using the **Soft Actor-Critic (SAC)** algorithm in a 3D **PyBullet** environment. The agent learns to navigate toward a target position while avoiding static obstacles using ray-based observations.

---

## 🚀 Features

- ✅ Continuous action space control with Soft Actor-Critic (SAC)
- ✅ Obstacle-aware navigation in a 3D simulation
- ✅ Custom PyBullet environment with goal and random obstacles
- ✅ Ray-based perception (8 azimuth x 5 elevation rays)
- ✅ Replay buffer saving and full checkpointing
- ✅ Training resume support

---

## 🧠 Algorithms

This project uses the **Soft Actor-Critic (SAC)** algorithm, an off-policy actor-critic method based on the maximum entropy framework. It provides a trade-off between exploration and exploitation by encouraging stochastic policies.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/SAC-RL-Navigation.git
cd SAC-RL-Navigation
pip install -r requirements.txt
