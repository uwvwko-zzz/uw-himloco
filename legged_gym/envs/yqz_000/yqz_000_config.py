from legged_gym.envs.w1w.w1w_config import W1WRoughCfg, W1WRoughCfgPPO


class YQZ000RoughCfg(W1WRoughCfg):
    """Training configuration for the YQZ-000 wheeled quadruped."""

    class init_state(W1WRoughCfg.init_state):
        # This pose respects the mirrored joint limits in the exported URDF.
        # Its nominal wheel-bottom height is approximately 0.45 m.
        pos = [0.0, 0.0, 0.50]
        default_joint_angles = {
            # About 5.7 degrees of inward hip roll reduces the holding effort
            # needed to keep the wheels under the body during long runs.
            "FL_hip_joint": 0.12,
            "FL_thigh_joint": -0.8,
            "FL_calf_joint": 1.55,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": -0.12,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.55,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": -0.12,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.55,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.12,
            "RR_thigh_joint": -0.8,
            "RR_calf_joint": 1.55,
            "RR_foot_joint": 0.0,
        }

    class control(W1WRoughCfg.control):
        # Give the policy enough joint-target range to unload a wheel and
        # replace it with a short corrective step under the 20 kg payload.
        # 0.25 rad made the old policy look almost locked in MuJoCo.
        action_scale = 0.35
        # YQZ wheel radius is 0.106 m. A 30 rad/s target corresponds to
        # approximately 3.18 m/s at the wheel rim, covering the 1-3 m/s
        # command range without forcing the policy to generate legged motion.
        wheel_action_scale = 30.0
        # The exported YQZ wheel inertia is only 2.85e-5 kg*m^2. A lower
        # velocity gain avoids discrete-time overshoot while retaining enough
        # torque to track 1-3 m/s commands.
        wheel_velocity_kp = 0.5
        wheel_joint_names = [
            "FL_foot_joint",
            "FR_foot_joint",
            "RL_foot_joint",
            "RR_foot_joint",
        ]
        wheel_directions = {
            "FL_foot_joint": -1.0,
            "FR_foot_joint": 1.0,
            "RL_foot_joint": -1.0,
            "RR_foot_joint": 1.0,
        }

    class commands(W1WRoughCfg.commands):
        # Keep zero-command samples so the deployed robot learns to hold the
        # low-effort default pose without slowly rolling away. Other samples
        # stay in the requested forward-speed range.
        standing_env_ratio = 0.15

        class ranges(W1WRoughCfg.commands.ranges):
            lin_vel_x = [1.0, 3.0]

    class asset(W1WRoughCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            "YQZ-000/urdf/YQZ-000.urdf"
        )
        name = "yqz_000"
        # Model the reflected motor/gear inertia. Without this armature the
        # very light 43 g wheels can jump from rest to >100 rad/s in one step.
        armature = 0.01
        foot_name = "foot_Link"
        penalize_contacts_on = ["hip_Link", "thigh_Link", "calf_Link", "base_link"]
        terminate_after_contacts_on = ["base_link"]

    class terrain(W1WRoughCfg.terrain):
        # Start on levels 0-1 so the policy first discovers lifting and foot
        # placement.  The inherited terrain curriculum can then promote it to
        # higher stairs and obstacles instead of starting as high as level 5.
        max_init_terrain_level = 1

    class domain_rand(W1WRoughCfg.domain_rand):
        # The bare robot is about 14.22 kg.  A 10-22 kg distribution lets part
        # of the batch discover a gait at medium load while retaining many
        # samples around the customer's intended 20 kg payload.
        randomize_payload_mass = True
        payload_mass_range = [10.0, 22.0]
        # Model small mounting/packing errors around a nominal centred load.
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]
        # Start close enough to the inward default stance that training learns
        # controlled foot replacement rather than recovering from huge poses.
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.95, 1.05]

    class rewards(W1WRoughCfg.rewards):
        # Default body height corresponding to the configured low-effort pose.
        base_height_target = 0.47
        # Wheel-centre height in the body frame while taking a corrective
        # step. Default is about -0.375 m, so this requests ~5 cm lift.
        clearance_height_target = -0.325

        class scales(W1WRoughCfg.rewards.scales):
            # Base/body contact termination penalty.
            termination = -0.5
            # Reward tracking of commanded planar linear velocity.
            tracking_lin_vel = 3.0
            # Reward tracking of commanded yaw angular velocity.
            tracking_ang_vel = 2.0
            # Penalize vertical body velocity and bouncing.
            lin_vel_z = -0.5
            # Penalize body roll/pitch angular velocity.
            ang_vel_xy = -0.1
            # Keep the body upright.
            orientation = -0.5
            # Penalize squared error from the commanded body height.
            base_height = -2.0
            # Reward accurate body-height tracking near the target.
            height_tracking = 1.5
            # Penalize sustained motor torque to reduce current and heat.
            torques = -7.5e-6
            # Penalize mechanical power |joint velocity * torque|.
            joint_power = -2.0e-6
            # Disable generic acceleration cost; legs and wheels use separate terms.
            dof_acc = 0.0
            # Disable generic joint-speed cost.
            dof_vel = 0.0
            # Keep enough acceleration freedom to lift while preventing the
            # high-frequency thrashing seen in the failed model_4300 policy.
            leg_dof_acc = -5.0e-8
            # Suppress abrupt wheel-speed changes with a wheel-specific scale.
            wheel_dof_acc = -2.5e-9
            # Penalize first-order action changes between control steps.
            action_rate = -0.003
            # Penalize second-order action changes for smoother commands.
            smoothness = -0.003
            # Penalize hip, thigh, calf or base collisions with terrain.
            collision = -0.5
            # Disable the legacy foot-stumble term.
            feet_stumble = 0.0
            # Default-pose holding is applied only at a zero command.  Keeping
            # both old zero-command terms strong double-counted the same error.
            stand_still = -1.0
            zero_command_leg_default = -1.0
            # A very small all-leg regularizer prevents runaway joint targets
            # but is weak enough for thigh/calf swing motion.
            joint_pos_penalty = -0.03
            # Pull the hips back to the inward default after each corrective
            # step, without demanding that a loaded wheel slide sideways while
            # it is still in contact with the ground.
            moving_hip_default = -0.10
            # Disable the older always-on L1 default-pose penalty.
            default_pos_linear = 0.0
            # Penalize only wheel speeds above max_wheel_speed (45 rad/s).
            wheel_speed = -2.0e-4
            # Penalize airborne wheel spin while moving and all wheel spin at rest.
            wheel_vel_penalty = -0.01
            # Penalize unnecessary loss of wheel contact. This discourages
            # lifting on easy terrain while still allowing obstacle traversal.
            wheel_contact_loss = -0.01
            # Disable the redundant legacy airborne-wheel-spin reward.
            wheel_spin_air = 0.0
            # At zero command, suppress policy wheel output.
            zero_command_wheel_action = -0.2
            # At zero command, suppress measured wheel velocity.
            zero_command_wheel_velocity = -0.02
            # Do not use a negative air-time scale: it can be reward-hacked by
            # rapidly tapping the wheels and caused model_4300 to thrash.
            feet_air_time = 0.0
            # While a wheel centre is moving relative to the body, pull it
            # toward clearance_height_target instead of dragging it sideways.
            foot_clearance = -0.20
            # Do not enforce a quadruped diagonal gait.
            diagonal_sync = 0.0
            # Do not add a separate hard-coded hip symmetry reward.
            hip_mirror_symmetry = 0.0
            # Do not require four-wheel contact through the legacy stand reward.
            stand_four_feet = 0.0
            # Upright posture is already handled by orientation.
            stand_orientation = 0.0


class YQZ000RoughCfgPPO(W1WRoughCfgPPO):
    class policy(W1WRoughCfgPPO.policy):
        # Start below the inherited 1.0 noise level so early random actions do
        # not slam the heavily loaded robot through its enlarged action range.
        init_noise_std = 0.6

    class algorithm(W1WRoughCfgPPO.algorithm):
        # Reduce persistent random exploration and allow a wider KL step so
        # the adaptive schedule does not collapse to its minimum learning rate
        # during the first few hundred iterations.
        # 0.01 drove model_4300's learned action std above 4.0.  The earlier
        # stable model converged near 0.31 with this conservative coefficient.
        entropy_coef = 0.003
        desired_kl = 0.02

    class runner(W1WRoughCfgPPO.runner):
        experiment_name = "yqz_rough"
        run_name = ""
