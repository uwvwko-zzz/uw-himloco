"""Numerical regression tests for the W1W RobotAdapter implementation."""

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PLUGIN_SRC = SCRIPT_DIR.parent / "mujoco-gui" / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from play_onnx import (  # noqa: E402
    DEFAULT_POS,
    KD,
    KP,
    LEG_INDICES,
    WHEEL_DIRECTIONS,
    WHEEL_INDICES,
)
from runtime_control import compute_pd_torques, scale_torque_limits  # noqa: E402
from w1w_adapter import W1W16Adapter  # noqa: E402


class W1WAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / "w1w.xml"))
        self.data = mujoco.MjData(self.model)
        self.adapter = W1W16Adapter().bind_and_validate(self.model)
        self.config = {
            "simulation": {
                "initial_position": [0.0, 0.0, 0.48],
                "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
            }
        }
        self.adapter.reset(self.model, self.data, self.config)

    def tearDown(self):
        self.adapter.unbind()

    def test_observation_contract_and_history_reset(self):
        observation = self.adapter.build_observation(
            self.model, self.data, np.zeros(3), height=0.45
        )
        self.assertEqual(observation.shape, (1, 348))
        self.assertEqual(observation.dtype, np.float32)
        self.assertGreater(np.linalg.norm(observation), 0.0)
        self.adapter.reset_policy_state()
        np.testing.assert_allclose(self.adapter.history, 0.0)

    def test_mixed_control_matches_original_equations(self):
        action = np.linspace(-0.8, 0.8, 16)
        state = {
            "kp": 80.0,
            "kd": 2.5,
            "torque_limit": 175.38,
            "motor_strength": 0.93,
            "motor_delay_ms": 0.0,
            "wheel_velocity_kp": 1.5,
        }
        actual = self.adapter.apply_control(
            self.model, self.data, action, state
        )

        joint_pos = self.data.qpos[self.adapter.qpos_adr]
        joint_vel = self.data.qvel[self.adapter.qvel_adr]
        limits = scale_torque_limits(
            self.adapter.nominal_limits, 175.38, 175.38
        )
        target = DEFAULT_POS + 0.25 * action
        expected = np.zeros(16)
        expected[LEG_INDICES] = compute_pd_torques(
            target[LEG_INDICES],
            joint_pos[LEG_INDICES],
            joint_vel[LEG_INDICES],
            KP[LEG_INDICES],
            KD[LEG_INDICES],
            motor_strength=0.93,
            torque_limit=limits[LEG_INDICES],
        )
        wheel_target = action[WHEEL_INDICES] * 1.5 * WHEEL_DIRECTIONS
        expected[WHEEL_INDICES] = np.clip(
            0.93 * 1.5 * (wheel_target - joint_vel[WHEEL_INDICES]),
            -limits[WHEEL_INDICES],
            limits[WHEEL_INDICES],
        )
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
