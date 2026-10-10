import torch

from legged_gym.envs.w1w.wheeled_legged_robot import WheeledLeggedRobot
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float


class YQZHighRobot(WheeledLeggedRobot):
    """YQZ wheel-legged task for climbing one vertical-faced platform."""

    def _init_buffers(self):
        super()._init_buffers()
        self.platform_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.platform_success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.platform_progress_max = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.front_progress_max = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.rear_progress_max = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.front_platform_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.front_platform_stage_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.front_platform_stage_event = torch.zeros_like(
            self.front_platform_stage_buf
        )
        self.platform_success_event = torch.zeros_like(
            self.platform_success_buf
        )
        self.post_success_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.descent_progress_max = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.descent_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.descent_started_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.descent_success_buf = torch.zeros_like(
            self.descent_started_buf
        )
        self.descent_success_event = torch.zeros_like(
            self.descent_started_buf
        )
        self.post_descent_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.platform_last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.platform_contact_filt = torch.zeros_like(
            self.platform_last_contacts
        )
        self.platform_failure_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.top_level_replay_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obstacle_impact_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.obstacle_reflex_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obstacle_reflex_ever_buf = torch.zeros_like(
            self.obstacle_reflex_buf
        )
        self.obstacle_impact_peak = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _flat_terrain_mask(self, env_ids=None):
        """Return environments assigned to obstacle-free terrain columns."""
        terrain_types = self.terrain_types
        if env_ids is not None:
            terrain_types = terrain_types[env_ids]
        column_choice = (
            terrain_types.float() / self.cfg.terrain.num_cols + 0.001
        )
        return column_choice < self.cfg.terrain.flat_terrain_ratio

    def _resample_commands(self, env_ids):
        """Sample explicit standing/forward/back/strafe/yaw/mixed modes."""
        if len(env_ids) == 0:
            return

        cfg = self.cfg.commands
        if not hasattr(self, "_command_distribution_start"):
            start_values = cfg.command_distribution_start
            end_values = cfg.command_distribution_end
            if len(start_values) != 6 or len(end_values) != 6:
                raise ValueError("YQZ command distributions must contain six modes")
            if any(
                value < 0.0
                for value in tuple(start_values) + tuple(end_values)
            ):
                raise ValueError("YQZ command probabilities must be non-negative")
            if abs(sum(start_values) - 1.0) > 1.0e-6 or abs(
                sum(end_values) - 1.0
            ) > 1.0e-6:
                raise ValueError("YQZ command probabilities must sum to one")
            self._command_distribution_start = torch.tensor(
                start_values, dtype=torch.float, device=self.device
            )
            self._command_distribution_end = torch.tensor(
                end_values, dtype=torch.float, device=self.device
            )
        start = self._command_distribution_start
        end = self._command_distribution_end

        count = len(env_ids)
        curriculum_progress = torch.clamp(
            self.terrain_levels[env_ids].float()
            / max(self.cfg.terrain.num_rows - 1, 1),
            0.0,
            1.0,
        ).unsqueeze(1)
        probabilities = start.unsqueeze(0) + curriculum_progress * (
            end - start
        ).unsqueeze(0)

        selector = torch.rand(count, device=self.device)
        boundaries = probabilities.cumsum(dim=1)
        standing = selector < boundaries[:, 0]
        forward = (selector >= boundaries[:, 0]) & (selector < boundaries[:, 1])
        backward = (selector >= boundaries[:, 1]) & (selector < boundaries[:, 2])
        lateral = (selector >= boundaries[:, 2]) & (selector < boundaries[:, 3])
        yaw = (selector >= boundaries[:, 3]) & (selector < boundaries[:, 4])
        mixed = selector >= boundaries[:, 4]

        self.commands[env_ids, :3] = 0.0
        minimum = cfg.min_directional_command

        def sample_positive(mask, command_index, maximum):
            ids = env_ids[mask]
            if len(ids) > 0:
                self.commands[ids, command_index] = torch_rand_float(
                    minimum, maximum, (len(ids), 1), device=self.device
                ).squeeze(1)

        def sample_signed(mask, command_index, maximum):
            ids = env_ids[mask]
            if len(ids) > 0:
                magnitude = torch_rand_float(
                    minimum, maximum, (len(ids), 1), device=self.device
                ).squeeze(1)
                sign = torch.where(
                    torch.rand(len(ids), device=self.device) < 0.5,
                    -torch.ones_like(magnitude),
                    torch.ones_like(magnitude),
                )
                self.commands[ids, command_index] = sign * magnitude

        flat = self._flat_terrain_mask(env_ids)
        sample_positive(
            forward & flat,
            0,
            self.command_ranges["lin_vel_x"][1],
        )
        platform_forward = forward & (~flat)
        platform_forward_ids = env_ids[platform_forward]
        if len(platform_forward_ids) > 0:
            self.commands[platform_forward_ids, 0] = torch_rand_float(
                cfg.climb_min_forward_command,
                self.command_ranges["lin_vel_x"][1],
                (len(platform_forward_ids), 1),
                device=self.device,
            ).squeeze(1)
        backward_ids = env_ids[backward]
        if len(backward_ids) > 0:
            self.commands[backward_ids, 0] = -torch_rand_float(
                minimum,
                abs(self.command_ranges["lin_vel_x"][0]),
                (len(backward_ids), 1),
                device=self.device,
            ).squeeze(1)
        sample_signed(
            lateral,
            1,
            max(abs(value) for value in self.command_ranges["lin_vel_y"]),
        )
        sample_signed(
            yaw,
            2,
            max(abs(value) for value in self.command_ranges["ang_vel_yaw"]),
        )

        mixed_ids = env_ids[mixed]
        if len(mixed_ids) > 0:
            for command_index, range_name in enumerate(
                ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
            ):
                lower, upper = self.command_ranges[range_name]
                self.commands[mixed_ids, command_index] = torch_rand_float(
                    lower,
                    upper,
                    (len(mixed_ids), 1),
                    device=self.device,
                ).squeeze(1)

        if not hasattr(self, "standing_env_mask"):
            self.standing_env_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self.standing_env_mask[env_ids] = standing
        self.commands[env_ids, 4] = self.cfg.rewards.base_height_target

        # Exposed for diagnostics and training logs without CPU synchronization.
        self.command_mode_probabilities = probabilities.mean(dim=0)

    def _compute_torques(self, actions):
        """Guarantee default legs and stopped wheels for a zero command."""
        zero_command = (
            torch.norm(self.commands[:, :3], dim=1)
            < self.cfg.rewards.reward_command_threshold
        )
        applied_actions = actions * (~zero_command).unsqueeze(1)
        return super()._compute_torques(applied_actions)

    def _reward_zero_command_actor_action(self):
        """Teach the actor itself to output zero behind the safety gate."""
        return torch.sum(torch.square(self.actions), dim=1) * (
            torch.norm(self.commands[:, :3], dim=1)
            < self.cfg.rewards.reward_command_threshold
        ).float()

    def _reset_dofs(self, env_ids):
        """Reset near the configured pose instead of the base class's 0.5-1.5 range."""
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            lower, upper = self.cfg.domain_rand.initial_joint_pos_range
            position_scale = torch_rand_float(
                lower,
                upper,
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        else:
            position_scale = torch.ones(
                len(env_ids), self.num_dof, device=self.device
            )
        self.dof_pos[env_ids] = self.default_dof_pos * position_scale
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        """Reset episode state without leaking observation/contact history."""
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.extras["episode"]["obstacle_reflex_rate"] = torch.mean(
            self.obstacle_reflex_ever_buf[env_ids].float()
        )
        self.extras["episode"]["obstacle_impact_peak"] = torch.mean(
            self.obstacle_impact_peak[env_ids]
        )
        self.extras["episode"]["front_platform_stage_rate"] = torch.mean(
            self.front_platform_stage_buf[env_ids].float()
        )
        self.extras["episode"]["descent_started_rate"] = torch.mean(
            self.descent_started_buf[env_ids].float()
        )
        self.extras["episode"]["descent_success_rate"] = torch.mean(
            self.descent_success_buf[env_ids].float()
        )
        self.extras["episode"]["top_level_replay_rate"] = torch.mean(
            self.top_level_replay_buf[env_ids].float()
        )
        platform_envs = ~self._flat_terrain_mask()
        if torch.any(platform_envs):
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[platform_envs].float()
            )
        self.obs_buf[env_ids] = 0.0
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.platform_stable_steps[env_ids] = 0
        self.platform_success_buf[env_ids] = False
        self.platform_success_event[env_ids] = False
        self.platform_progress_max[env_ids] = 0.0
        self.front_progress_max[env_ids] = 0.0
        self.rear_progress_max[env_ids] = 0.0
        self.front_platform_stable_steps[env_ids] = 0
        self.front_platform_stage_buf[env_ids] = False
        self.front_platform_stage_event[env_ids] = False
        self.post_success_steps[env_ids] = 0
        self.descent_progress_max[env_ids] = 0.0
        self.descent_stable_steps[env_ids] = 0
        self.descent_started_buf[env_ids] = False
        self.descent_success_buf[env_ids] = False
        self.descent_success_event[env_ids] = False
        self.post_descent_steps[env_ids] = 0
        self.top_level_replay_buf[env_ids] = False
        self.platform_last_contacts[env_ids] = False
        self.platform_contact_filt[env_ids] = False
        self.obstacle_impact_streak[env_ids] = 0
        self.obstacle_reflex_buf[env_ids] = False
        self.obstacle_reflex_ever_buf[env_ids] = False
        self.obstacle_impact_peak[env_ids] = 0.0

    def refresh_actor_rigid_shape_props(self, env_ids):
        """Skip per-actor property writes when shape randomization is disabled."""
        if (
            self.cfg.domain_rand.randomize_friction
            or self.cfg.domain_rand.randomize_restitution
        ):
            return super().refresh_actor_rigid_shape_props(env_ids)

    def check_termination(self):
        """Terminate unsafe motion and completed climbs."""
        current_wheel_contacts = (
            self.contact_forces[:, self.feet_indices, 2] > 1.0
        )
        # Isaac contact forces can disappear for a single frame while a wheel
        # is still physically supported. One-frame hysteresis prevents that
        # numerical flicker from resetting the success hold counter.
        self.platform_contact_filt = torch.logical_or(
            current_wheel_contacts, self.platform_last_contacts
        )
        self.platform_last_contacts = current_wheel_contacts
        self.front_platform_stage_event[:] = False
        self.platform_success_event[:] = False
        self.descent_success_event[:] = False
        self._update_obstacle_reflex()
        super().check_termination()
        leg_limits = self.dof_vel_limits[self.leg_dof_indices]
        leg_over_speed = torch.any(
            torch.abs(self.dof_vel[:, self.leg_dof_indices])
            > leg_limits * self.cfg.rewards.terminate_leg_dof_vel_ratio,
            dim=1,
        )
        self.reset_buf |= leg_over_speed
        self._update_platform_milestones()
        if (
            self.cfg.rewards.train_platform_descent
            and self.cfg.rewards.terminate_on_descent_success
        ):
            recovered = self.descent_success_buf & (
                self.post_descent_steps
                >= self.cfg.rewards.post_descent_recovery_steps
            )
            self.reset_buf |= recovered
        elif self.cfg.rewards.terminate_on_platform_success:
            recovered = self.platform_success_buf & (
                self.post_success_steps
                >= self.cfg.rewards.post_success_recovery_steps
            )
            self.reset_buf |= recovered

    def _update_platform_milestones(self):
        """Latch front placement and success without allowing reward farming."""
        platform_command = self._platform_command_mask()
        front_candidate = (
            self._front_wheels_on_platform()
            & platform_command
            & self.obstacle_reflex_buf
        )
        self.front_platform_stable_steps = torch.where(
            front_candidate,
            self.front_platform_stable_steps + 1,
            torch.zeros_like(self.front_platform_stable_steps),
        )
        new_front_stage = (
            self.front_platform_stable_steps
            >= self.cfg.rewards.front_platform_hold_steps
        ) & (~self.front_platform_stage_buf)
        self.front_platform_stage_event = new_front_stage
        self.front_platform_stage_buf |= new_front_stage

        stable_candidate = (
            self._instant_platform_success_mask() & platform_command
        )
        self.platform_stable_steps = torch.where(
            stable_candidate,
            self.platform_stable_steps + 1,
            torch.zeros_like(self.platform_stable_steps),
        )
        new_success = (
            self.platform_stable_steps
            >= self.cfg.rewards.platform_success_hold_steps
        ) & (~self.platform_success_buf)
        self.platform_success_event = new_success
        self.platform_success_buf |= new_success

        # Once fully on top, restore ordinary four-wheel/default constraints.
        # The success remains latched for curriculum accounting at reset.
        self.obstacle_reflex_buf &= ~self.platform_success_buf
        self.post_success_steps = torch.where(
            self.platform_success_buf,
            self.post_success_steps + 1,
            torch.zeros_like(self.post_success_steps),
        )

        if self.cfg.rewards.train_platform_descent:
            self._update_descent_milestones(new_success)

    def _update_descent_milestones(self, new_platform_success):
        """Latch a safe four-wheel landing beyond the platform's far edge."""
        progress = self._descent_progress_potential()
        self.descent_progress_max = torch.where(
            new_platform_success,
            progress,
            self.descent_progress_max,
        )
        self.descent_started_buf |= self._descent_transition_mask()

        stable_candidate = self._instant_descent_success_mask()
        self.descent_stable_steps = torch.where(
            stable_candidate,
            self.descent_stable_steps + 1,
            torch.zeros_like(self.descent_stable_steps),
        )
        new_success = (
            self.descent_stable_steps
            >= self.cfg.rewards.descent_success_hold_steps
        ) & (~self.descent_success_buf)
        self.descent_success_event = new_success
        self.descent_success_buf |= new_success
        self.post_descent_steps = torch.where(
            self.descent_success_buf,
            self.post_descent_steps + 1,
            torch.zeros_like(self.post_descent_steps),
        )

    def _update_obstacle_reflex(self):
        """Latch a blind climb reflex after the leading wheels hit a wall.

        The trigger uses a physical event that is inferable from the policy's
        proprioceptive history (wheel command/velocity mismatch, joint motion,
        and IMU response).  It deliberately does not use platform position or
        terrain height, so identical flat-ground observations never receive a
        hidden reward-mode switch before contact.
        """
        leading_indices = torch.topk(
            self.feet_pos[:, :, 0], k=2, dim=1, largest=True
        ).indices
        opposing_x_force = torch.gather(
            -self.contact_forces[:, self.feet_indices, 0],
            1,
            leading_indices,
        ).amax(dim=1)
        self.obstacle_impact_peak = torch.maximum(
            self.obstacle_impact_peak,
            torch.where(
                self._platform_command_mask(),
                opposing_x_force,
                torch.zeros_like(opposing_x_force),
            ),
        )
        impact = self._platform_command_mask() & (~self.platform_success_buf) & (
            opposing_x_force
            > self.cfg.rewards.obstacle_impact_force_threshold
        )
        self.obstacle_impact_streak = torch.where(
            impact,
            self.obstacle_impact_streak + 1,
            torch.zeros_like(self.obstacle_impact_streak),
        )
        new_reflex = (
            self.obstacle_impact_streak
            >= self.cfg.rewards.obstacle_impact_hold_steps
        )
        newly_triggered = new_reflex & (~self.obstacle_reflex_buf)
        self.obstacle_reflex_buf |= new_reflex
        self.obstacle_reflex_ever_buf |= newly_triggered
        # Do not pay front-wheel approach progress that occurred before the
        # wall was physically detected.
        front_potential = self._front_wheel_progress_potential()
        self.front_progress_max = torch.where(
            newly_triggered,
            front_potential,
            self.front_progress_max,
        )

    def _reset_root_states(self, env_ids):
        """Spawn on the approach, aligned with and facing the platform."""
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        spawn_x_min, spawn_x_max = self.cfg.rewards.spawn_x_range
        self.root_states[env_ids, 0] += torch_rand_float(
            spawn_x_min,
            spawn_x_max,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        spawn_y_min, spawn_y_max = self.cfg.rewards.spawn_y_range
        self.root_states[env_ids, 1] += torch_rand_float(
            spawn_y_min,
            spawn_y_max,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _platform_height(self):
        ratio = self.terrain_levels.float() / max(self.cfg.terrain.num_rows - 1, 1)
        return self.cfg.terrain.min_platform_height + ratio * (
            self.cfg.terrain.max_platform_height
            - self.cfg.terrain.min_platform_height
        )

    def _platform_command_mask(self):
        """Only straight, forward commands are platform-climb attempts."""
        threshold = self.cfg.rewards.reward_command_threshold
        return (
            (self.commands[:, 0] > threshold)
            & (torch.abs(self.commands[:, 1]) < threshold)
            & (torch.abs(self.commands[:, 2]) < threshold)
            & (~self._flat_terrain_mask())
        )

    def _normal_rolling_mask(self):
        """Use all four wheels until a proprioceptively detectable impact."""
        threshold = self.cfg.rewards.reward_command_threshold
        straight_translation = (
            (torch.abs(self.commands[:, 0]) > threshold)
            & (torch.abs(self.commands[:, 1]) < threshold)
            & (torch.abs(self.commands[:, 2]) < threshold)
        )
        return (
            straight_translation
            & (~self.obstacle_reflex_buf)
            & (~self._descent_transition_mask())
        )

    def _descent_transition_mask(self):
        """Return post-climb environments close enough to react to the drop."""
        edge_start = (
            0.5 * self.cfg.terrain.terrain_length
            - self.cfg.rewards.descent_edge_margin
        )
        return (
            self.platform_success_buf
            & (~self.descent_success_buf)
            & self._platform_command_mask()
            & (self._local_x() > edge_start)
        )

    def _descent_progress_potential(self):
        """Bounded forward progress from climb completion to lower ground."""
        start = (
            self.cfg.terrain.platform_start_x
            + self.cfg.rewards.platform_success_margin
        )
        finish = (
            0.5 * self.cfg.terrain.terrain_length
            + self.cfg.rewards.descent_landing_margin
        )
        return torch.clamp(
            (self._local_x() - start) / max(finish - start, 1.0e-6),
            0.0,
            1.0,
        )

    def _instant_descent_success_mask(self):
        """Require a settled, upright, four-wheel landing on lower ground."""
        past_edge = self._local_x() > (
            0.5 * self.cfg.terrain.terrain_length
            + self.cfg.rewards.descent_landing_margin
        )
        ground_height = self.env_origins[:, 2]
        low_enough = self.root_states[:, 2] < (
            ground_height
            + self.cfg.rewards.base_height_target
            + self.cfg.rewards.descent_height_tolerance
        )
        wheel_count = torch.sum(self.platform_contact_filt, dim=1)
        enough_wheels = (
            wheel_count >= self.cfg.rewards.descent_success_wheel_count
        )
        settled = torch.abs(self.root_states[:, 9]) < (
            self.cfg.rewards.descent_vertical_speed_tolerance
        )
        upright = -self.projected_gravity[:, 2] > 0.85
        return (
            self.platform_success_buf
            & self.descent_started_buf
            & self._platform_command_mask()
            & past_edge
            & low_enough
            & enough_wheels
            & settled
            & upright
        )

    def _reward_wheel_contact_loss(self):
        """Prevent tripod rolling, while allowing wheel lift near the platform."""
        airborne_count = torch.sum(
            (~self.platform_contact_filt).float(), dim=1
        )
        upright_scale = torch.clamp(
            -self.projected_gravity[:, 2], 0.0, 0.7
        ) / 0.7
        return (
            airborne_count
            * self._normal_rolling_mask().float()
            * upright_scale
        )

    def _reward_rolling_leg_default(self):
        """Keep all leg joints near default during ordinary wheel rolling."""
        leg_error = torch.sum(
            torch.abs(
                self.dof_pos[:, self.leg_dof_indices]
                - self.default_dof_pos[:, self.leg_dof_indices]
            ),
            dim=1,
        )
        upright = (-self.projected_gravity[:, 2] > 0.7).float()
        return leg_error * self._normal_rolling_mask().float() * upright

    def _reward_rolling_orientation(self):
        """Keep the chassis level during rolling, but release it after impact."""
        tilt_error = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        return tilt_error * self._normal_rolling_mask().float()

    def _reward_tracking_lin_vel(self):
        """Track world +X while climbing so turning around cannot game reward."""
        body_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        desired_world_velocity = torch.stack(
            (self.commands[:, 0], torch.zeros_like(self.commands[:, 0])),
            dim=1,
        )
        world_error = torch.sum(
            torch.square(desired_world_velocity - self.root_states[:, 7:9]),
            dim=1,
        )
        velocity_error = torch.where(
            self._platform_command_mask(), world_error, body_error
        )
        return torch.exp(-velocity_error / self.cfg.rewards.tracking_sigma)

    def _local_x(self):
        return self.root_states[:, 0] - self.env_origins[:, 0]

    def _wheel_platform_mask(self):
        edge_x = self.env_origins[:, 0:1] + self.cfg.terrain.platform_start_x
        centre_y = self.env_origins[:, 1:2]
        surface_height = (
            self.env_origins[:, 2:3] + self._platform_height().unsqueeze(1)
        )
        inside_platform = (
            torch.abs(self.feet_pos[:, :, 1] - centre_y)
            < self.cfg.terrain.platform_half_width
        )
        above_top = self.feet_pos[:, :, 2] > (
            surface_height - self.cfg.rewards.platform_height_tolerance
        )
        top_contact = self.platform_contact_filt
        return (
            (self.feet_pos[:, :, 0] > edge_x)
            & inside_platform
            & above_top
            & top_contact
        )

    def _rear_wheel_platform_mask(self):
        """Return platform contact for the two wheels trailing in world +X."""
        rear_indices = torch.topk(
            self.feet_pos[:, :, 0], k=2, dim=1, largest=False
        ).indices
        return torch.gather(self._wheel_platform_mask(), 1, rear_indices)

    def _front_wheel_platform_mask(self):
        """Return platform contact for the two wheels leading in world +X."""
        front_indices = torch.topk(
            self.feet_pos[:, :, 0], k=2, dim=1, largest=True
        ).indices
        return torch.gather(self._wheel_platform_mask(), 1, front_indices)

    def _front_wheels_on_platform(self):
        return torch.all(self._front_wheel_platform_mask(), dim=1)

    def _rear_wheels_on_platform(self):
        return torch.all(self._rear_wheel_platform_mask(), dim=1)

    def _instant_platform_success_mask(self):
        past_edge = self._local_x() > (
            self.cfg.terrain.platform_start_x
            + self.cfg.rewards.platform_success_margin
        )
        wheel_count = torch.sum(self._wheel_platform_mask(), dim=1)
        enough_wheels = (
            wheel_count >= self.cfg.rewards.platform_success_wheel_count
        ) & self._rear_wheels_on_platform()
        climbed = self.root_states[:, 2] > (
            self.env_origins[:, 2]
            + self._platform_height()
            + 0.75 * self.cfg.rewards.base_height_target
        )
        upright = -self.projected_gravity[:, 2] > 0.85
        return past_edge & enough_wheels & climbed & upright

    def _success_mask(self):
        """Return only success that survived the configured stability window."""
        if self.cfg.rewards.train_platform_descent:
            return self.descent_success_buf
        return self.platform_success_buf

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        attempted = self._platform_command_mask()[env_ids]
        success = self._success_mask()[env_ids] & attempted
        failed = attempted & (~success)
        streak = self.platform_failure_streak[env_ids]
        streak = torch.where(failed, streak + 1, streak)
        streak = torch.where(success, torch.zeros_like(streak), streak)
        move_down = failed & (
            streak >= self.cfg.rewards.platform_failures_before_demotion
        )
        streak = torch.where(move_down, torch.zeros_like(streak), streak)
        self.platform_failure_streak[env_ids] = streak
        levels = self.terrain_levels[env_ids]
        top_success = success & (levels >= self.max_terrain_level - 1)
        levels = torch.clamp(
            levels + success.long() - move_down.long(),
            min=0,
            max=self.max_terrain_level - 1,
        )
        replay_probability = self.cfg.rewards.top_level_replay_probability
        replay = top_success & (
            torch.rand(len(env_ids), device=self.device)
            < replay_probability
        )
        replay_levels = torch.randint(
            self.cfg.rewards.top_level_replay_min_level,
            self.max_terrain_level,
            (len(env_ids),),
            device=self.device,
            dtype=levels.dtype,
        )
        self.terrain_levels[env_ids] = torch.where(
            replay, replay_levels, levels
        )
        self.top_level_replay_buf[env_ids] = replay
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def _reward_platform_progress(self):
        """Potential-based approach progress that can be collected only once."""
        potential = torch.clamp(
            self._local_x() / self.cfg.terrain.platform_start_x, 0.0, 1.0
        )
        progress = torch.clamp(
            potential - self.platform_progress_max, min=0.0
        )
        self.platform_progress_max = torch.maximum(
            self.platform_progress_max, potential
        )
        # Reward scales are multiplied by dt in the base class. Dividing the
        # potential delta by dt makes the episode-integrated bonus equal to
        # the configured scale while remaining bounded.
        return (
            progress / self.dt * self._platform_command_mask().float()
        )

    def _front_wheel_progress_potential(self):
        """Monotonic progress of the slower leading wheel across the lip."""
        front_x = torch.topk(
            self.feet_pos[:, :, 0], k=2, dim=1, largest=True
        ).values
        edge_x = self.env_origins[:, 0] + self.cfg.terrain.platform_start_x
        margin = self.cfg.rewards.front_wheel_progress_margin
        travel = margin + self.cfg.rewards.platform_success_margin
        return torch.clamp(
            (front_x.min(dim=1).values - (edge_x - margin)) / travel,
            0.0,
            1.0,
        )

    def _reward_front_wheel_progress(self):
        """One-shot front placement shaping; oscillation cannot repay it."""
        potential = self._front_wheel_progress_potential()
        progress = torch.clamp(potential - self.front_progress_max, min=0.0)
        active = self.obstacle_reflex_buf & self._platform_command_mask()
        self.front_progress_max = torch.where(
            active,
            torch.maximum(self.front_progress_max, potential),
            self.front_progress_max,
        )
        return progress / self.dt * active.float()

    def _reward_front_platform_stage(self):
        """Pay once after both leading wheels remain on top for five steps."""
        return self.front_platform_stage_event.float()

    def _reward_front_contact_regression(self):
        """After front placement, penalize dropping either leading wheel."""
        regressed = (
            self.front_platform_stage_buf
            & (~self.platform_success_buf)
            & (~self._front_wheels_on_platform())
            & self._platform_command_mask()
        )
        return regressed.float()

    def _reward_wheels_on_platform(self):
        wheel_count = torch.sum(self._wheel_platform_mask(), dim=1)
        reward = torch.where(
            (
                wheel_count >= self.cfg.rewards.platform_reward_wheel_count
            )
            & self._rear_wheels_on_platform(),
            wheel_count.float() / self.feet_pos.shape[1],
            torch.zeros_like(wheel_count, dtype=torch.float),
        )
        return reward * self._platform_command_mask().float()

    def _reward_rear_wheel_progress(self):
        """One-shot potential progress for moving trailing wheels over the lip."""
        rear_x = torch.topk(
            self.feet_pos[:, :, 0], k=2, dim=1, largest=False
        ).values
        edge_x = self.env_origins[:, 0:1] + self.cfg.terrain.platform_start_x
        margin = self.cfg.rewards.rear_wheel_progress_margin
        travel = margin + self.cfg.rewards.platform_success_margin
        potential = torch.clamp(
            (rear_x.mean(dim=1) - (edge_x.squeeze(1) - margin)) / travel,
            0.0,
            1.0,
        )
        progress = torch.clamp(potential - self.rear_progress_max, min=0.0)
        self.rear_progress_max = torch.maximum(
            self.rear_progress_max, potential
        )
        return (
            progress / self.dt * self._platform_command_mask().float()
        )

    def _reward_base_on_platform(self):
        past_edge = self._local_x() > self.cfg.terrain.platform_start_x
        climbed = self.root_states[:, 2] > (
            self.env_origins[:, 2]
            + self._platform_height()
            + 0.75 * self.cfg.rewards.base_height_target
        )
        enough_wheels = (
            torch.sum(self._wheel_platform_mask(), dim=1)
            >= self.cfg.rewards.platform_reward_wheel_count
        ) & self._rear_wheels_on_platform()
        return (
            past_edge
            & climbed
            & enough_wheels
            & self._platform_command_mask()
        ).float()

    def _reward_platform_success(self):
        return (
            self.platform_success_event & self._platform_command_mask()
        ).float()

    def _reward_descent_progress(self):
        """One-shot progress across the top and beyond the far edge."""
        potential = self._descent_progress_potential()
        active = self.platform_success_buf & (~self.descent_success_buf)
        progress = torch.clamp(
            potential - self.descent_progress_max, min=0.0
        )
        self.descent_progress_max = torch.where(
            active,
            torch.maximum(self.descent_progress_max, potential),
            self.descent_progress_max,
        )
        return progress / self.dt * active.float()

    def _reward_descent_success(self):
        """Pay once for a stable lower-ground landing."""
        return self.descent_success_event.float()

    def _reward_descent_ground_contact(self):
        """Reward contact recovery only after the chassis reaches ground level."""
        low_enough = self.root_states[:, 2] < (
            self.env_origins[:, 2]
            + self.cfg.rewards.base_height_target
            + self.cfg.rewards.descent_height_tolerance
        )
        contact_fraction = torch.mean(
            self.platform_contact_filt.float(), dim=1
        )
        upright = torch.clamp(
            -self.projected_gravity[:, 2], 0.0, 1.0
        )
        return (
            contact_fraction
            * upright
            * (self.descent_started_buf & low_enough).float()
        )

    def _reward_backward_motion(self):
        # Penalize X motion opposite to the requested direction. Unlike the old
        # definition, this does not punish a correctly executed S/back command.
        x_velocity = torch.where(
            self._platform_command_mask(),
            self.root_states[:, 7],
            self.base_lin_vel[:, 0],
        )
        return torch.clamp(-self.commands[:, 0] * x_velocity, min=0.0)

    def _reward_highplatform_yaw(self):
        """Keep the chassis square to the platform instead of climbing diagonally."""
        # Identity is the commanded orientation for this dedicated task.
        yaw_sin = 2.0 * (
            self.base_quat[:, 3] * self.base_quat[:, 2]
            + self.base_quat[:, 0] * self.base_quat[:, 1]
        )
        yaw_cos = 1.0 - 2.0 * (
            torch.square(self.base_quat[:, 1])
            + torch.square(self.base_quat[:, 2])
        )
        yaw_error = torch.square(torch.atan2(yaw_sin, yaw_cos))
        return yaw_error * self._platform_command_mask().float()

    def _reward_highplatform_world_z_vel(self):
        """Penalize only violent vertical impacts, not the required lift."""
        excess = torch.clamp(
            torch.abs(self.root_states[:, 9])
            - self.cfg.rewards.highplatform_world_z_speed_limit,
            min=0.0,
        )
        return torch.square(excess)

    def _reward_highplatform_leg_dof_vel(self):
        """Soft penalty before the hard leg-speed termination threshold."""
        leg_speed = torch.abs(self.dof_vel[:, self.leg_dof_indices])
        excess = torch.clamp(
            leg_speed - self.cfg.rewards.highplatform_leg_dof_vel_limit,
            min=0.0,
        )
        return torch.sum(torch.square(excess), dim=1)
