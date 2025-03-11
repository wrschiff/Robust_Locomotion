import gym
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os
import numpy as np
from gym.envs.registration import register
import gym
from gym import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import os
import numpy as np


class HumanBigTray(MujocoEnv):

    metadata = {"render_modes": ["human"], "video.frames_per_second": 30}

    def __init__(self, render_mode = None, terminate_when_unhealthy = True):
        xml_file = f"{os.getcwd()}/basic_humanoid_bigger_tray.xml"  # Path to Mujoco XML

        # 1️⃣ **TEMPORARY observation_space for initialization** (dummy value)
        dummy_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        MujocoEnv.__init__(self, xml_file, 5, observation_space=dummy_obs_space)

        obs_dim = self.model.nq + self.model.nv + 6  # +6 for tray and box positions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.action_dim = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        ########### REWARD DEFINITIONS ###########
        self.FW_W = 1.25   # Forward velocity weight
        self.CTRL_W = 0.1   # Control cost weight
        self.HEALTH_R = 5.0 # Healthy reward
        self.HEALTHY_Z = (1.0, 2.0)  # Height range for being "healthy"
        self.BAL_BOX_R = 4.0  # Balance reward weight
        self.END_ON_BOX_DROP = terminate_when_unhealthy  # Termination if box falls
        self.RSRT_NOISE_SCALE = 0.01  # Reset noise scale
        self.EXCLUDE_CURR_POS_OBS = True  # Exclude root position from obs

        # Get Mujoco IDs for the tray and box
        self.tray_id = self.model.body("tray").id
        self.box_id = self.model.body("box").id

        self.render_mode = render_mode

        self.tray_id = self.model.body("tray").id
        self.box_id = self.model.body("box").id



    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        # Compute forward velocity reward
        root_x_before = self.data.qpos[0]  # X position before step
        root_x_after = self.data.qpos[0]   # X position after step
        forward_velocity = (root_x_after - root_x_before) / self.dt
        forward_reward = self.FW_W * forward_velocity

        # Compute health reward
        torso_height = self.data.qpos[2]  # Assuming Z-axis height
        is_healthy = self.HEALTHY_Z[0] <= torso_height <= self.HEALTHY_Z[1]
        healthy_reward = self.HEALTH_R if is_healthy else 0

        # Compute control cost
        ctrl_cost = self.CTRL_W * np.sum(np.square(action))

        # Compute balance cost (tray-box distance penalty)
        tray_pos = self.data.xpos[self.tray_id]
        box_pos = self.data.xpos[self.box_id]
        balance_cost = self.BAL_BOX_R * np.linalg.norm(tray_pos - box_pos)

        # Total reward
        reward = forward_reward + healthy_reward - ctrl_cost - balance_cost

        # Check termination conditions
        done = self._is_unhealthy() or (self.END_ON_BOX_DROP and self._is_box_dropped())

        if self.render_mode == "human":
            self.render()

        truncated = False

        info = {}
        return obs, reward, done, truncated, info

    def _is_unhealthy(self):
        """Returns True if the humanoid has fallen or is unstable."""
        torso_height = self.data.qpos[2]  # Z-axis height
        joint_angles = self.data.qpos[3:]  # Excluding root position
        joint_velocities = self.data.qvel[:]

        if torso_height < 0.5:  # Fallen condition
            return True
        if np.any(np.abs(joint_angles) > np.pi):  # Joint limits exceeded
            return True
        if np.any(np.abs(joint_velocities) > 10.0):  # Excessive velocity
            return True

        return False

    def _is_box_dropped(self):
        """Returns True if the box has fallen off the tray."""
        tray_height = self.data.xpos[self.tray_id][2]
        box_height = self.data.xpos[self.box_id][2]
        return box_height < tray_height

    def reset_model(self):
        """Resets the environment and returns the initial observation."""

        # Apply reset noise to joint positions (qpos) and velocities (qvel)
        qpos = self.init_qpos + self.RSRT_NOISE_SCALE * np.random.uniform(low=-1.0, high=1.0, size=self.model.nq)
        qvel = self.init_qvel + self.RSRT_NOISE_SCALE * np.random.uniform(low=-1.0, high=1.0, size=self.model.nv)

        self.set_state(qpos, qvel)  # Properly resets the Mujoco state

        return self._get_obs()

    def render(self):
        """Ensures Mujoco properly renders frames, only if simulation is running."""
        if self.render_mode == "human":
            if self.data.time > 0:  # ✅ Ensure simulation has stepped before rendering
                self.mujoco_renderer.render(self.render_mode)


    def _get_obs(self):
        """Returns the observation, including humanoid state and tray/box positions."""
        base_obs = np.concatenate([self.data.qpos, self.data.qvel])
        tray_pos = self.data.xpos[self.tray_id]
        box_pos = self.data.xpos[self.box_id]
        obs = np.concatenate([base_obs, tray_pos, box_pos])
        # print(f"Observation Shape: {obs.shape}")
        return obs.astype(np.float32)
    

register(
    id="HumanBigTray-v0",
    entry_point="load_xml:HumanBigTray",  # Make sure 'load_xml.py' is in the same directory
)