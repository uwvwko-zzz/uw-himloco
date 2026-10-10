from legged_gym.envs.yqz_000.yqz_000_config import (
    YQZ000RoughCfg,
    YQZ000RoughCfgPPO,
)


class YQZHighCfg(YQZ000RoughCfg):
    """Single-high-platform curriculum for the YQZ/M20-style robot."""

    class env(YQZ000RoughCfg.env):
        # A slow 0.5 m/s attempt needs time to approach, climb, traverse the
        # platform and react to the blind drop on its far edge.
        episode_length_s = 18

    class terrain(YQZ000RoughCfg.terrain):
        terrain_type = "high_platform"
        curriculum = True
        # Keep roughly the old 4.7 cm curriculum increment while extending
        # the final level from 50 cm to 80 cm.
        num_rows = 16
        num_cols = 10
        # A fresh blind policy must first learn the complete 8 cm
        # climb-traverse-descent sequence before receiving harder platforms.
        max_init_terrain_level = 0
        terrain_length = 8.0
        terrain_width = 6.0
        # Level 0 begins at 8 cm and level 15 reaches 80 cm.
        min_platform_height = 0.08
        max_platform_height = 0.80
        # The vertical face is 1.5 m in front of the nominal spawn.
        platform_start_x = 1.5
        platform_half_width = 2.5
        # Two of ten terrain columns contain no obstacle.  These environments
        # teach sustained four-wheel rolling instead of making every forward
        # command synonymous with an imminent climb.
        flat_terrain_ratio = 0.20
        terrain_proportions = [1.0]

    class commands(YQZ000RoughCfg.commands):
        curriculum = False
        # One command mode is held for the whole episode. This gives the
        # platform curriculum a well-defined forward-attempt signal while
        # still training all keyboard directions and an exact zero command.
        resampling_time = 1.0e6
        # [stand, climb-forward, backward, lateral, yaw, mixed]. Sampling is
        # interpolated independently for every environment from its terrain
        # level. Hard platforms therefore receive more dedicated attempts,
        # while every level still contains zero-command and full WASD data.
        command_distribution_start = [0.15, 0.60, 0.04, 0.06, 0.07, 0.08]
        command_distribution_end = [0.12, 0.73, 0.03, 0.04, 0.04, 0.04]
        min_directional_command = 0.30
        climb_min_forward_command = 0.50

        class ranges(YQZ000RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            range_height = [0.47, 0.47]

    class domain_rand(YQZ000RoughCfg.domain_rand):
        # Moderate sim-to-real randomization around the intended 20 kg load.
        # The ranges are deliberately narrower than the generic base task so
        # the 50 cm contact manoeuvre remains learnable from the first stage.
        randomize_payload_mass = True
        payload_mass_range = [18.0, 22.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]
        randomize_link_mass = True
        link_mass_range = [0.90, 1.10]
        randomize_friction = True
        friction_range = [0.50, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 0.20]
        randomize_motor_strength = True
        motor_strength_range = [0.90, 1.10]
        randomize_kp = True
        kp_range = [0.90, 1.10]
        randomize_kd = True
        kd_range = [0.90, 1.10]
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.95, 1.05]
        delay = True
        disturbance = True
        disturbance_range = [-40.0, 40.0]
        disturbance_interval = 25
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class rewards(YQZ000RoughCfg.rewards):
        # Match the reference high-platform branch: clipping early negative
        # returns keeps exploration alive before the robot can clear the lip.
        only_positive_rewards = True
        platform_success_margin = 0.35
        platform_height_tolerance = 0.06
        # Three contacts are robust to one lightly unloaded wheel, but the
        # condition must persist for 0.1 s (5 control steps) before success.
        # Wheel contact additionally uses one-frame hysteresis to reject
        # PhysX contact-force flicker without relaxing the geometry checks.
        platform_reward_wheel_count = 3
        platform_success_wheel_count = 3
        platform_success_hold_steps = 5
        front_platform_hold_steps = 5
        # Compatibility fallback when descent training is disabled.
        post_success_recovery_steps = 50
        # A complete platform episode now includes the blind far-edge drop.
        # The actor receives no edge location; it must react to proprioceptive
        # wheel-contact/IMU history after the leading wheels begin to fall.
        train_platform_descent = True
        terminate_on_descent_success = True
        descent_edge_margin = 0.45
        descent_landing_margin = 0.35
        descent_height_tolerance = 0.18
        descent_vertical_speed_tolerance = 0.50
        descent_success_wheel_count = 4
        descent_success_hold_steps = 10
        post_descent_recovery_steps = 25
        platform_failures_before_demotion = 3
        # After completing the hardest climb+descent, replay the full
        # curriculum occasionally to prevent forgetting without draining most
        # samples away from the 80 cm objective.
        top_level_replay_probability = 0.25
        top_level_replay_min_level = 0
        front_wheel_progress_margin = 0.45
        rear_wheel_progress_margin = 0.60
        # Blind obstacle reflex: a platform is detected only after a leading
        # wheel receives sustained force opposite the forward command.  Flat
        # rolling produced 0 N in this direction, while a 50 cm wall produced
        # a 29 N median and >100 N peak in the nominal evaluation.
        obstacle_impact_force_threshold = 20.0
        obstacle_impact_hold_steps = 2
        # Randomize the approach distance so the policy cannot learn a timed
        # lift from episode reset.  At the minimum climb command this still
        # leaves several seconds to complete the manoeuvre.
        spawn_x_range = [-0.60, 0.15]
        spawn_y_range = [-0.20, 0.20]
        terminate_on_platform_success = False
        highplatform_world_z_speed_limit = 0.75
        highplatform_leg_dof_vel_limit = 16.0
        # Terminate genuinely unsafe leg motion, while ignoring unbounded
        # wheel speeds as the reference M20 implementation does.
        terminate_leg_dof_vel_ratio = 0.95

        class scales(YQZ000RoughCfg.rewards.scales):
            tracking_lin_vel = 3.0
            platform_progress = 2.0
            front_wheel_progress = 3.0
            front_platform_stage = 20.0
            front_contact_regression = -2.0
            rear_wheel_progress = 4.0
            wheels_on_platform = 2.0
            base_on_platform = 1.0
            platform_success = 100.0
            descent_progress = 3.0
            descent_success = 100.0
            descent_ground_contact = 1.0
            backward_motion = -3.0
            # Unlike posture rewards behind the safety gate, this term acts
            # directly on the actor's raw 16-D output at a zero command.
            zero_command_actor_action = -1.0
            rolling_leg_default = -1.0
            rolling_orientation = -2.0
            highplatform_yaw = -2.0
            highplatform_world_z_vel = -10.0
            highplatform_leg_dof_vel = -1.0
            collision = -1.0
            orientation = -1.0
            wheel_contact_loss = -2.0


class YQZHighCfgPPO(YQZ000RoughCfgPPO):
    class runner(YQZ000RoughCfgPPO.runner):
        experiment_name = "yqz_high"
        run_name = ""
        max_iterations = 5000
