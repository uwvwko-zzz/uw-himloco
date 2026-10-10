"""YQZ-000 policy/robot contract for the MuJoCo runtime-control plugin."""

import mujoco
import numpy as np

from runtime_control import (
    MotorCommandDelay,
    RobotAdapter,
    compute_pd_torques,
    scale_torque_limits,
)


JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
]
WHEEL_INDICES = np.array([3, 7, 11, 15])
LEG_INDICES = np.array([i for i in range(16) if i not in WHEEL_INDICES])
WHEEL_DIRECTIONS = np.array([-1.0, 1.0, -1.0, 1.0])
DEFAULT_POS = np.array([
    0.12, -0.8, 1.55, 0.0,
    -0.12, 0.8, -1.55, 0.0,
    -0.12, 0.8, -1.55, 0.0,
    0.12, -0.8, 1.55, 0.0,
])
KP = np.array([80.0, 80.0, 80.0, 0.0] * 4)
KD = np.array([2.0, 2.0, 2.0, 0.0] * 4)


def quat_to_rotation(q):
    """Return the world-from-body rotation for MuJoCo [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


class YQZ000Adapter(RobotAdapter):
    """Adapt the YQZ 12 leg-position + 4 wheel-velocity policy."""

    root_body_name = "base_link"
    expected_dimensions = (23, 22, 16)
    observation_size = 348
    one_step_observation_size = 58
    action_size = 16
    control_dt = 0.02

    def __init__(
        self,
        action_scale=0.25,
        wheel_action_scale=30.0,
        kp_reference=80.0,
        kd_reference=2.0,
        torque_limit_reference=120.0,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        self.wheel_action_scale = float(wheel_action_scale)
        self.kp_reference = float(kp_reference)
        self.kd_reference = float(kd_reference)
        self.torque_limit_reference = float(torque_limit_reference)
        self.qpos_adr = None
        self.qvel_adr = None
        self.base_body_id = None
        self.nominal_limits = None
        self.effective_limits = None
        self.motor_delay = None
        self.history = np.zeros(self.observation_size, dtype=np.float32)
        self.last_action = np.zeros(self.action_size, dtype=np.float32)

    def bind(self, model):
        qpos, qvel, actuators = [], [], []
        for joint_name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            actuator_name = joint_name.replace("_joint", "")
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(
                    "Missing YQZ joint/actuator: %s / %s"
                    % (joint_name, actuator_name)
                )
            qpos.append(model.jnt_qposadr[joint_id])
            qvel.append(model.jnt_dofadr[joint_id])
            actuators.append(actuator_id)
        self.qpos_adr = np.asarray(qpos, dtype=np.int32)
        self.qvel_adr = np.asarray(qvel, dtype=np.int32)
        self._actuator_ids = np.asarray(actuators, dtype=np.int32)
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name
        )
        self.nominal_limits = np.max(
            np.abs(model.actuator_ctrlrange[self._actuator_ids]), axis=1
        )
        self.motor_delay = MotorCommandDelay(model.opt.timestep)
        self.reset_policy_state()

    def unbind(self):
        super().unbind()
        self.qpos_adr = None
        self.qvel_adr = None
        self.base_body_id = None
        self.nominal_limits = None
        self.effective_limits = None
        self.motor_delay = None

    def control_steps(self, model):
        self._require_bound(model)
        return max(1, round(self.control_dt / model.opt.timestep))

    def reset_policy_state(self):
        self.history.fill(0.0)
        self.last_action.fill(0.0)
        self.last_control = None
        if self.motor_delay is not None:
            self.motor_delay.reset()

    def reset(self, model, data, runtime_config):
        self._require_bound(model)
        simulation = runtime_config["simulation"]
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = simulation["initial_position"]
        data.qpos[3:7] = simulation["initial_quaternion"]
        data.qpos[self.qpos_adr] = DEFAULT_POS
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.reset_policy_state()
        mujoco.mj_forward(model, data)

    def build_observation(self, model, data, command, **context):
        self._require_bound(model)
        height = float(context["height"])
        joint_pos = data.qpos[self.qpos_adr].copy()
        joint_vel = data.qvel[self.qvel_adr].copy()
        pos_obs = joint_pos - DEFAULT_POS
        pos_obs[WHEEL_INDICES] = 0.0
        gravity_body = (
            quat_to_rotation(data.qpos[3:7]).T
            @ np.array([0.0, 0.0, -1.0])
        )
        one_step = np.concatenate((
            np.asarray(command) * np.array([2.0, 2.0, 0.25]),
            data.qvel[3:6] * 0.25,
            gravity_body,
            pos_obs,
            joint_vel * 0.05,
            self.last_action,
            np.array([(height - 0.51) / 0.11]),
        )).astype(np.float32)
        if one_step.size != self.one_step_observation_size:
            raise ValueError(
                "YQZ observation has %d values, expected %d"
                % (one_step.size, self.one_step_observation_size)
            )
        self.history[self.one_step_observation_size:] = self.history[
            :-self.one_step_observation_size
        ].copy()
        self.history[:self.one_step_observation_size] = one_step
        return self.history[np.newaxis].copy()

    def compute_control(self, model, data, action, runtime_state):
        self._require_bound(model)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(
                "YQZ action has shape %s, expected (%d,)"
                % (action.shape, self.action_size)
            )
        self.last_action[:] = action
        delayed = self.motor_delay.apply(
            action, runtime_state["motor_delay_ms"]
        )
        joint_pos = data.qpos[self.qpos_adr].copy()
        joint_vel = data.qvel[self.qvel_adr].copy()
        self.effective_limits = scale_torque_limits(
            self.nominal_limits,
            runtime_state["torque_limit"],
            reference_limit=self.torque_limit_reference,
        )
        target = DEFAULT_POS + self.action_scale * delayed
        kp = KP * (runtime_state["kp"] / self.kp_reference)
        kd = KD * (runtime_state["kd"] / self.kd_reference)
        torque = np.zeros(self.action_size, dtype=np.float64)
        torque[LEG_INDICES] = compute_pd_torques(
            target[LEG_INDICES],
            joint_pos[LEG_INDICES],
            joint_vel[LEG_INDICES],
            kp[LEG_INDICES],
            kd[LEG_INDICES],
            motor_strength=runtime_state["motor_strength"],
            torque_limit=self.effective_limits[LEG_INDICES],
        )
        wheel_target = (
            delayed[WHEEL_INDICES]
            * self.wheel_action_scale
            * WHEEL_DIRECTIONS
        )
        torque[WHEEL_INDICES] = np.clip(
            runtime_state["motor_strength"]
            * runtime_state["wheel_velocity_kp"]
            * (wheel_target - joint_vel[WHEEL_INDICES]),
            -self.effective_limits[WHEEL_INDICES],
            self.effective_limits[WHEEL_INDICES],
        )
        return torque

