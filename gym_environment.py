import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class MyPyBulletEnvContinuous(gym.Env):
    def __init__(self, render=False):
        super(MyPyBulletEnvContinuous, self).__init__()

        self.obstacle_ids = []
        self.wall_ids = []

        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )

        obs_dim = 6 + (8 * 5)  # = 46
        high = np.full(obs_dim, 10, dtype=np.float32)
        low = np.full(obs_dim, -10, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.wall_thickness = 0.1
        self.wall_height = 1
        self.sphere_radius = 0.3

        self.max_step_size = 0.2

        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self._create_walls()

        obstacle_positions = [
            (0, 0, 1),
            (3, -4, 1),
            (-4, 4, 1),
            (5, 1, 1),
            (-7, 6, 1),
        ]
        for pos in obstacle_positions:
            self._create_obstacle(pos)

        self.agent_start_pos = np.random.uniform(low=[-5, -5, 0.5], high=[5, 5, 1.5])
        self.goal_pos = np.random.uniform(low=[-5, -5, 0.5], high=[5, 5, 1.5])

        self.agent_id = self._create_agent(self.agent_start_pos)
        self.goal_id = self._create_goal(self.goal_pos)
        self.prev_dist = None

        return self._get_observation()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        current_pos = np.array(p.getBasePositionAndOrientation(self.agent_id)[0])  # (x, y, z)

        delta = action * self.max_step_size
        new_pos = current_pos + delta

        if new_pos[2] < self.sphere_radius:
            new_pos[2] = self.sphere_radius
        if new_pos[2] > 2.0:
            new_pos[2] = 2.0

        if new_pos[0] < -10 + self.sphere_radius:
            new_pos[0] = -10 + self.sphere_radius
        elif new_pos[0] > 10 - self.sphere_radius:
            new_pos[0] = 10 - self.sphere_radius

        if new_pos[1] < -10 + self.sphere_radius:
            new_pos[1] = -10 + self.sphere_radius
        elif new_pos[1] > 10 - self.sphere_radius:
            new_pos[1] = 10 - self.sphere_radius

        p.resetBasePositionAndOrientation(self.agent_id, new_pos.tolist(), [0, 0, 0, 1])

        for _ in range(5):
            p.stepSimulation()
            time.sleep(1 / 240.)

        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)
        info = {}

        contacts = p.getContactPoints(bodyA=self.agent_id)
        if contacts:
            for c in contacts:
                if c[2] in self.obstacle_ids or c[2] in self.wall_ids:
                    reward -= 500.0
                    done = True
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def _create_walls(self):
        # X-Wall
        wall_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.wall_thickness, 10, self.wall_height]
        )
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.wall_thickness, 10, self.wall_height],
            rgbaColor=[0.7, 0.7, 0.7, 1]
        )

        # Wall X-1
        wall_x_1 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[10, 0, self.wall_height]
        )

        # Wall X-2
        wall_x_2 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[-10, 0, self.wall_height]
        )

        # Wall Y
        wall_collision_y = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[10, self.wall_thickness, self.wall_height]
        )
        wall_visual_y = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[10, self.wall_thickness, self.wall_height],
            rgbaColor=[0.7, 0.7, 0.7, 1]
        )
        # Wall Y-1
        wall_y_1 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_y,
            baseVisualShapeIndex=wall_visual_y,
            basePosition=[0, 10, self.wall_height]
        )
        # Wall Y-2
        wall_y_2 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_y,
            baseVisualShapeIndex=wall_visual_y,
            basePosition=[0, -10, self.wall_height]
        )

        self.wall_ids.extend([wall_x_1, wall_x_2, wall_y_1, wall_y_2])

    def _create_obstacle(self, pos):
        box_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[1,1,1]
        )
        box_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1,1,1],
            rgbaColor=[0.6, 0.6, 0.3, 1]
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=pos
        )
        self.obstacle_ids.append(obstacle_id)

    def _create_agent(self, start_pos):
        sphere_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.sphere_radius
        )
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.sphere_radius,
            rgbaColor=[0, 0, 1, 1]
        )
        agent_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=start_pos
        )
        return agent_id

    def _create_goal(self, goal_pos):
        goal_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.2]
        )
        goal_visual=p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2,0.2,0.2],
            rgbaColor=[0,1,0,1]
        )
        goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=goal_collision,
            baseVisualShapeIndex=goal_visual,
            basePosition=goal_pos
        )
        return goal_id

    def _get_observation(self):
        agent_pos = p.getBasePositionAndOrientation(self.agent_id)[0]
        goal_x, goal_y, goal_z = self.goal_pos

        n_azimuth = 8
        n_elev = 5
        max_range = 1.0

        ray_starts = []
        ray_ends = []

        for az in range(n_azimuth):
            theta = 2 * np.pi * az / n_azimuth
            for el in range(n_elev):
                phi = np.pi * el / (n_elev - 1)
                range_x = max_range * np.sin(phi) * np.cos(theta)
                range_y = max_range * np.sin(phi) * np.sin(theta)
                range_z = max_range * np.cos(phi)

                ray_start = [agent_pos[0], agent_pos[1], agent_pos[2]]
                ray_end = [agent_pos[0] + range_x, agent_pos[1] + range_y, agent_pos[2] + range_z]

                ray_starts.append(ray_start)
                ray_ends.append(ray_end)

        ray_results = p.rayTestBatch(ray_starts, ray_ends)
        distances = []
        for r in ray_results:
            hit_uid = r[0]
            fraction = r[2]
            if hit_uid >= 0:
                dist = fraction * max_range
            else:
                dist = max_range
            distances.append(dist)

        obs_list = [
            agent_pos[0], agent_pos[1], agent_pos[2],
            goal_x, goal_y, goal_z
        ]
        obs_list.extend(distances)

        obs = np.array(obs_list, dtype=np.float32)
        return obs

    def _compute_reward(self, obs):
        agent_x, agent_y, agent_z, goal_x, goal_y, goal_z = obs[:6]
        dist_to_goal = np.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2 + (goal_z - agent_z) ** 2)

        if self.prev_dist is None:
            self.prev_dist = dist_to_goal

        delta = self.prev_dist - dist_to_goal

        reward = delta * 10
        self.prev_dist = dist_to_goal

        if dist_to_goal < 0.5:
            reward += 100.0

        return reward

    def _check_done(self, obs):
        agent_x, agent_y, agent_z, goal_x, goal_y, goal_z = obs[:6]
        dist_to_goal = np.sqrt((agent_x - goal_x) ** 2 + (agent_y - goal_y) ** 2 + (agent_z - goal_z) ** 2)

        if dist_to_goal < 0.5:
            return True
        if agent_z < 0:
            self.reward -= 500.0
            return True
        return False


if __name__ == "__main__":
    env = MyPyBulletEnvContinuous(render=True)
    obs = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action={action}, Obs={obs}, Reward={reward}, Done={done}")
        if done:
            print("Episode completed, resetting...")
            obs = env.reset()

    env.close()