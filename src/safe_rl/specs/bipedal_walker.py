import temporal_logic.signal_tl as stl

SIGNALS = (
    'hull_angle',
    'hull_angularVelocity',
    'vel_x',
    'vel_y',
    'hip_joint_1_angle',
    'hip_joint_1_speed',
    'knee_joint_1_angle',
    'knee_joint_1_speed',
    'leg_1_ground_contact_flag',
    'hip_joint_2_angle',
    'hip_joint_2_speed',
    'knee_joint_2_angle',
    'knee_joint_2_speed',
    'leg_2_ground_contact_flag',
    'lidar0',
    'lidar1',
    'lidar2',
    'lidar3',
    'lidar4',
    'lidar5',
    'lidar6',
    'lidar7',
    'lidar8',
    'lidar9',
)

(hull_angle, hull_angularVelocity, vel_x, vel_y, hip_joint_1_angle, hip_joint_1_speed, knee_joint_1_angle,
 knee_joint_1_speed, leg_1_ground_contact_flag, hip_joint_2_angle, hip_joint_2_speed, knee_joint_2_angle,
 knee_joint_2_speed, leg_2_ground_contact_flag, lidar0, lidar1, lidar2, lidar3, lidar4, lidar5, lidar6, lidar7, lidar8,
 lidar9) = stl.signals(
    SIGNALS)

SPEC = stl.G(
    stl.And(
        # Be moving forward always
        (vel_x > 1),
        # Reduce hull tilt to ~ +- 12deg
        (abs(hull_angle) <= 0.2),
        # Reduce hull angular velocity
        (abs(hull_angularVelocity) < 2),
        # Prevents running
        stl.Implies((leg_1_ground_contact_flag > 0),
                    stl.And(
                        # Set foot in 10-16 timesteps
                        stl.F(leg_2_ground_contact_flag > 0, (10, 16)),
                        # Make sure to actually lift leg
                        stl.G(leg_2_ground_contact_flag <= 0, (1, 8)),
        )),
        stl.Implies((leg_2_ground_contact_flag > 0),
                    stl.And(
                        # Set foot in 10-16 timesteps
                        stl.F(leg_1_ground_contact_flag > 0, (10, 16)),
                        # Make sure to actually lift leg
                        stl.G(leg_1_ground_contact_flag <= 0, (1, 8)),
        )),
    )
)
