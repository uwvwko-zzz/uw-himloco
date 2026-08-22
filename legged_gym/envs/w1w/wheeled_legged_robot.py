import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class WheeledLeggedRobot(LeggedRobot):
    """LeggedRobot with position-controlled legs and velocity-controlled wheels."""

    def _init_buffers(self):
        super()._init_buffers()

        wheel_names = set(self.cfg.control.wheel_joint_names)
        self.wheel_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if name in wheel_names],
            dtype=torch.long,
            device=self.device,
        )
        if len(self.wheel_dof_indices) != len(wheel_names):
            missing = wheel_names.difference(self.dof_names)
            raise ValueError(f"Wheel joints not found in URDF: {sorted(missing)}")

        wheel_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device)
        wheel_mask[self.wheel_dof_indices] = True
        self.leg_dof_indices = torch.arange(self.num_dof, device=self.device)[~wheel_mask]
        self.hip_dof_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "hip_joint" in name],
            dtype=torch.long,
            device=self.device,
        )
        if len(self.hip_dof_indices) != 4:
            raise ValueError(
                f"Expected 4 hip joints, found {len(self.hip_dof_indices)}"
            )

        directions = [
            self.cfg.control.wheel_directions[name]
            for name in self.dof_names
            if name in wheel_names
        ]
        self.wheel_directions = torch.tensor(
            directions, dtype=torch.float, device=self.device
        )

    def _compute_torques(self, actions):
        """Convert leg actions to positions and wheel actions to angular speeds."""
        leg_actions = actions[:, self.leg_dof_indices]
        leg_targets = (
            self.default_dof_pos[:, self.leg_dof_indices]
            + self.cfg.control.action_scale * leg_actions
        )

        hip_mask = torch.tensor(
            ["hip_joint" in self.dof_names[i] for i in self.leg_dof_indices],
            dtype=torch.bool,
            device=self.device,
        )
        leg_targets[:, hip_mask] = (
            self.default_dof_pos[:, self.leg_dof_indices][:, hip_mask]
            + self.cfg.control.action_scale
            * self.cfg.control.hip_reduction
            * leg_actions[:, hip_mask]
        )

        # Keep a full-sized target for generic logging/reward code.
        self.joint_pos_target = self.default_dof_pos.repeat(self.num_envs, 1)
        self.joint_pos_target[:, self.leg_dof_indices] = leg_targets

        torques = torch.zeros_like(actions)
        torques[:, self.leg_dof_indices] = (
            self.p_gains[self.leg_dof_indices]
            * self.Kp_factors
            * (leg_targets - self.dof_pos[:, self.leg_dof_indices])
            - self.d_gains[self.leg_dof_indices]
            * self.Kd_factors
            * self.dof_vel[:, self.leg_dof_indices]
        )

        wheel_targets = (
            actions[:, self.wheel_dof_indices]
            * self.cfg.control.wheel_action_scale
            * self.wheel_directions
        )
        torques[:, self.wheel_dof_indices] = (
            self.cfg.control.wheel_velocity_kp
            * (wheel_targets - self.dof_vel[:, self.wheel_dof_indices])
        )
        torques *= self.motor_strength_factors
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_dof_pos_obs(self):
        """Go2W-style position observation: wheel positions are always zero."""
        joint_pos_rel = self.dof_pos - self.default_dof_pos
        joint_pos_rel = joint_pos_rel.clone()
        joint_pos_rel[:, self.wheel_dof_indices] = 0.0
        return joint_pos_rel

    def _get_height_command_obs(self):
        """按 W1W 的高度命令范围归一化到约 [-1, 1]。"""
        lower, upper = self.command_ranges["range_height"]
        center = 0.5 * (lower + upper)
        half_range = max(0.5 * (upper - lower), 1.0e-6)
        return (self.commands[:, 4] - center) / half_range

    def _resample_commands(self, env_ids):
        """采样运动/高度指令，并显式保留一部分严格静止环境。"""
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        if not hasattr(self, "standing_env_mask"):
            self.standing_env_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        standing = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.commands.standing_env_ratio
        )
        self.standing_env_mask[env_ids] = standing
        standing_ids = env_ids[standing]

        # 默认关节姿态只对应默认高度，因此静止样本不能再随机高度。
        self.commands[standing_ids, :3] = 0.0
        self.commands[standing_ids, 4] = self.cfg.rewards.base_height_target

    def _post_physics_step_callback(self):
        """执行通用回调，并防止 heading 模式覆盖严格零指令样本。"""
        super()._post_physics_step_callback()
        if hasattr(self, "standing_env_mask"):
            self.commands[self.standing_env_mask, :3] = 0.0
            self.commands[
                self.standing_env_mask, 4
            ] = self.cfg.rewards.base_height_target

    def _reward_stand_still(self):
        """RobotLab 风格：零指令时用 L1 距离约束腿部默认姿态。"""
        stand_mask = (
            torch.norm(self.commands[:, :3], dim=1)
            < self.cfg.rewards.reward_command_threshold
        ).float()
        leg_error = torch.abs(
            self.dof_pos[:, self.leg_dof_indices]
            - self.default_dof_pos[:, self.leg_dof_indices]
        )
        upright_scale = torch.clamp(
            -self.projected_gravity[:, 2], 0.0, 0.7
        ) / 0.7
        return torch.sum(leg_error, dim=1) * stand_mask * upright_scale

    def _robotlab_moving_mask(self, velocity_threshold):
        """Return RobotLab's command/body-speed based moving-state mask."""
        command_speed = torch.norm(self.commands[:, :3], dim=1)
        body_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return (
            command_speed > self.cfg.rewards.reward_command_threshold
        ) | (body_speed > velocity_threshold)

    def _reward_joint_pos_penalty(self):
        """RobotLab-style L2 leg posture penalty with stronger standing cost."""
        leg_error = torch.linalg.vector_norm(
            self.dof_pos[:, self.leg_dof_indices]
            - self.default_dof_pos[:, self.leg_dof_indices],
            dim=1,
        )
        moving = self._robotlab_moving_mask(
            self.cfg.rewards.joint_pos_velocity_threshold
        )
        error_scale = torch.where(
            moving,
            torch.ones_like(leg_error),
            torch.full_like(
                leg_error, self.cfg.rewards.joint_pos_stand_still_scale
            ),
        )
        upright_scale = torch.clamp(
            -self.projected_gravity[:, 2], 0.0, 0.7
        ) / 0.7
        return leg_error * error_scale * upright_scale

    def _reward_default_pos_linear(self):
        """惩罚腿关节偏离默认姿态，不包含无界的轮子角度。"""
        return torch.sum(
            torch.abs(
                self.dof_pos[:, self.leg_dof_indices]
                - self.default_dof_pos[:, self.leg_dof_indices]
            ),
            dim=1,
        )

    def _reward_wheel_speed(self):
        """惩罚超过实用速度阈值的轮速，阈值内不产生该项惩罚。"""
        speed = torch.abs(self.dof_vel[:, self.wheel_dof_indices])
        return torch.sum(
            torch.square(
                torch.clamp(speed - self.cfg.rewards.max_wheel_speed, min=0.0)
            ),
            dim=1,
        )

    def _reward_wheel_vel_penalty(self):
        """RobotLab-style wheel penalty for moving and standing commands."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        airborne = ~contact
        wheel_speed = torch.abs(
            self.dof_vel[:, self.wheel_dof_indices]
        )
        moving_penalty = torch.sum(
            wheel_speed * airborne.float(), dim=1
        )
        standing_penalty = torch.sum(wheel_speed, dim=1)
        return torch.where(
            self._robotlab_moving_mask(
                self.cfg.rewards.wheel_vel_velocity_threshold
            ),
            moving_penalty,
            standing_penalty,
        )

    def _reward_wheel_contact_loss(self):
        """Penalize unnecessary wheel contact loss while following commands."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # 使用一步接触滞回过滤滚动接触抖动，只有连续两个
        # 控制周期未接触才认为真正离地。
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        airborne_count = torch.sum((~contact_filt).float(), dim=1)

        commanded_motion = (
            torch.norm(self.commands[:, :3], dim=1)
            > self.cfg.rewards.reward_command_threshold
        ).float()
        upright_scale = torch.clamp(
            -self.projected_gravity[:, 2], 0.0, 0.7
        ) / 0.7
        return airborne_count * commanded_motion * upright_scale

    def _reward_leg_dof_acc(self):
        """12个腿关节的加速度平方和。"""
        leg_acceleration = (
            self.last_dof_vel[:, self.leg_dof_indices]
            - self.dof_vel[:, self.leg_dof_indices]
        ) / self.dt
        return torch.sum(torch.square(leg_acceleration), dim=1)

    def _reward_wheel_dof_acc(self):
        """4个轮关节的加速度平方和。"""
        wheel_acceleration = (
            self.last_dof_vel[:, self.wheel_dof_indices]
            - self.dof_vel[:, self.wheel_dof_indices]
        ) / self.dt
        return torch.sum(torch.square(wheel_acceleration), dim=1)

    def _zero_command_upright_mask(self):
        """返回三轴速度指令为零且机身基本朝上的环境掩码。"""
        zero_command = (
            (torch.norm(self.commands[:, :2], dim=1) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        )
        # 直立时 projected_gravity 的 z 分量接近 -1。
        upright = -self.projected_gravity[:, 2] > 0.7
        return (zero_command & upright).float()

    def _reward_zero_command_leg_default(self):
        """零指令时惩罚12个腿关节与默认角度之间的绝对误差。"""
        leg_position_error = torch.sum(
            torch.abs(
                self.dof_pos[:, self.leg_dof_indices]
                - self.default_dof_pos[:, self.leg_dof_indices]
            ),
            dim=1,
        )
        return leg_position_error * self._zero_command_upright_mask()

    def _reward_moving_hip_default(self):
        """运动时约束 hip-roll 回中，避免四个轮端形成内八姿态。"""
        moving = (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        ) | (torch.abs(self.commands[:, 2]) > 0.1)
        upright = -self.projected_gravity[:, 2] > 0.7
        hip_error = (
            self.dof_pos[:, self.hip_dof_indices]
            - self.default_dof_pos[:, self.hip_dof_indices]
        )
        return torch.sum(torch.abs(hip_error), dim=1) * (
            moving & upright
        ).float()

    def _reward_zero_command_wheel_action(self):
        """零指令时惩罚轮子策略输出，使四个轮子动作趋近于零。"""
        wheel_action = torch.sum(
            torch.abs(self.actions[:, self.wheel_dof_indices]), dim=1
        )
        return wheel_action * self._zero_command_upright_mask()

    def _reward_zero_command_wheel_velocity(self):
        """零指令时惩罚实际轮速，抑制机器人缓慢漂移。"""
        wheel_velocity = torch.sum(
            torch.abs(self.dof_vel[:, self.wheel_dof_indices]), dim=1
        )
        return wheel_velocity * self._zero_command_upright_mask()

    def _reward_base_height(self):
        """机身相对地形高度与高度指令之间的平方误差。"""
        height_error = self._get_base_heights() - self.commands[:, 4]
        return torch.square(height_error)

    def _reward_height_tracking(self):
        """接近目标时提供精细梯度的指数高度跟踪奖励。"""
        height_error = self._get_base_heights() - self.commands[:, 4]
        return torch.exp(
            -torch.square(height_error) / self.cfg.rewards.height_tracking_sigma
        )
