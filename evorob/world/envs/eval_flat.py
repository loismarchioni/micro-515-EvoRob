from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalFlatEnv(MujocoEnv, utils.EzPickle):
    """Flat terrain evaluation environment.

    Termination: robot torso must stay between 0.2 m and 1.0 m above the
    ground (height-based).

    Training reward:  healthy_reward + x_velocity - ctrl_cost - cfrc_cost

    Tune ctrl_cost_weight and cfrc_cost_weight to shape behaviour on the
    flat surface.

    The info dict always exposes the four keys required by the neutral
    leaderboard formula: healthy_reward, x_position, ctrl_cost, cfrc_cost.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.005,
        cfrc_cost_weight: float = 5e-4,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        xml_file_path = robot_path if path.isabs(robot_path) else path.join(
            path.dirname(path.realpath(__file__)), robot_path
        )

        utils.EzPickle.__init__(
            self, xml_file_path, frame_skip, default_camera_config,
            ctrl_cost_weight, cfrc_cost_weight, reset_noise_scale, **kwargs,
        )

        self._ctrl_cost_weight = min(ctrl_cost_weight, 0.01)
        self._cfrc_cost_weight = cfrc_cost_weight
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self, xml_file_path, frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def step(self, action):
        xyz_before = self.data.body(1).xpos[:3].copy()
        self.do_simulation(action, self.frame_skip)
        xyz_after = self.data.body(1).xpos[:3].copy()

        terminated = self._is_terminated()

        xyz_velocity = (xyz_after - xyz_before) / self.dt
        x_velocity = float(xyz_velocity[0])
        x_position = float(xyz_after[0])
        x_delta_pos = float(xyz_after[0]) - float(xyz_before[0])


        healthy_reward = -5.0 if terminated else 0.0
        ctrl_cost = float(np.sum(action ** 2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)


        # ADDONS
        forward_reward = 20.0 * x_delta_pos
        idling_cost    = 1.0 if abs(x_delta_pos) < 0.01 else 0.0
        distance_x     = 0.1 * x_position

        reward = healthy_reward + forward_reward + distance_x - ctrl_cost - cfrc_cost - idling_cost

        info = {
            "healthy_reward": healthy_reward,
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "idling_cost": idling_cost,
            "x_position": x_position,
            "delta_posx": x_delta_pos,
            "x_velocity": x_velocity,
        }

        # print(info)
        # print("\n")

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def _is_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        return (
            not np.isfinite(self.state_vector()).all()
            or z < 0.25
            or z > 2.0
        )

    def _get_obs(self):
        return np.concatenate((self.data.qpos.flat[2:], self.data.qvel.flat.copy()))

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": float(self.data.qpos[0])}
