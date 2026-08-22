"""W1W policy/robot contract for the reusable MuJoCo runtime."""

import mujoco
import numpy as np

from runtime_control import (
    MotorCommandDelay,
    RobotAdapter,
    compute_pd_torques,
    scale_torque_limits,
)

from play_onnx import (
    DEFAULT_POS,
    HIP_INDICES,
    JOINT_NAMES,
    KD,
    KP,
    LEG_INDICES,
    WHEEL_DIRECTIONS,
    WHEEL_INDICES,
    quat_to_rotation,
)


class W1W16Adapter(RobotAdapter):
    """Adapt the 12 leg-position + 4 wheel-velocity W1W policy."""

    root_body_name = "base"
    expected_dimensions = (23, 22, 16)
    observation_size = 348
    one_step_observation_size = 58
    action_size = 16
    control_dt = 0.02

    def __init__(
        self,
        action_scale=0.25,
        wheel_action_scale=1.5,
        kp_reference=80.0,
        kd_reference=2.5,
        torque_limit_reference=175.38,
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
        self.wheel_geoms = set()
        self.wheel_radius = 0.0
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
                    "Missing W1W joint/actuator: %s / %s"
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

        wheel_bodies = set()
        for index in WHEEL_INDICES:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[index]
            )
            wheel_bodies.add(int(model.jnt_bodyid[joint_id]))
        self.wheel_geoms = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in wheel_bodies
            and model.geom_contype[geom_id] != 0
        }
        if len(self.wheel_geoms) != 4:
            raise ValueError(
                "Expected 4 W1W wheel collision geoms, found %d"
                % len(self.wheel_geoms)
            )
        self.wheel_radius = float(np.mean([
            model.geom_size[geom_id, 0] for geom_id in self.wheel_geoms
        ]))
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
        self.wheel_geoms = set()
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
        if "height" not in context:
            raise ValueError("W1W observation requires height=...")
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
                "W1W one-step observation has %d values, expected %d"
                % (one_step.size, self.one_step_observation_size)
            )
        self.history[self.one_step_observation_size:] = self.history[
            :-self.one_step_observation_size
        ].copy()
        self.history[:self.one_step_observation_size] = one_step
        return self.history[np.newaxis, :].copy()

    def compute_control(self, model, data, action, runtime_state):
        self._require_bound(model)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(
                "W1W action has shape %s, expected (%d,)"
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
        target[HIP_INDICES] = (
            DEFAULT_POS[HIP_INDICES]
            + self.action_scale * delayed[HIP_INDICES]
        )
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

