import copy
import csv
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform, TwistStamped, Twist, Vector3, Quaternion
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import pprint
import rosbag
import rospy
import tf.transformations
import tf2_py
import traceback

################################################################################
# CONSTANTS
################################################################################
# ed209
# IN_DIR = "/home/amaln/workspaces/amal_ws/data/2021_extended_action_space_dataset/raw/"
# OUT_DIR = "/home/amaln/workspaces/amal_ws/data/2021_extended_action_space_dataset/processed/"
# rosie-backpack
# IN_DIR = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study"
# OUT_DIR = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study/processed/"
# external harddrive
IN_DIR = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/"
OUT_DIR = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/processed/"

################################################################################
# TF Transformation Helper Functions
################################################################################

# ROS rotation convention, see https://answers.ros.org/question/53688/euler-angle-convention-in-tf/?answer=53693#post-id-53693
EULER_ORDER = 'rzyx'

def transform_to_matrix(transform_msg):
    m = tf.transformations.quaternion_matrix([
        transform_msg.rotation.x,
        transform_msg.rotation.y,
        transform_msg.rotation.z,
        transform_msg.rotation.w,
    ])
    m[0][3] = transform_msg.translation.x
    m[1][3] = transform_msg.translation.y
    m[2][3] = transform_msg.translation.z
    return m

def pose_to_matrix(pose_msg):
    m = tf.transformations.quaternion_matrix([
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
        pose_msg.orientation.w,
    ])
    m[0][3] = pose_msg.position.x
    m[1][3] = pose_msg.position.y
    m[2][3] = pose_msg.position.z
    return m

def quaternion_msg_to_euler(quaternion_msg):
    q = [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]
    x, y, z = tf.transformations.euler_from_quaternion(q, EULER_ORDER)
    return [x, y, z]

################################################################################
# Main Code
################################################################################

def parse_rosbag_name(filename):
    """
    Return the participant number, food name, and trial number
    for a given rosbag name.

    The naming convention is:

    {participant_num}-{food_name}-{trial_num}_{timestamp}.bag
    """
    print(filename)
    first_hyphen = filename.find("-")
    if first_hyphen == -1:
        raise Exception("Filename %s is malformed" % filename)
    participant_num = int(filename[:first_hyphen])
    second_hyphen = filename.find("-", first_hyphen+1)
    if second_hyphen == -1:
        raise Exception("Filename %s is malformed" % filename)
    food_name = filename[first_hyphen+1:second_hyphen]
    first_underscore = filename.find("_", second_hyphen+1)
    if first_underscore == -1:
        raise Exception("Filename %s is malformed" % filename)
    trial_num = int(filename[second_hyphen+1:first_underscore])

    return participant_num, food_name, trial_num

def get_initial_static_transforms(start_time, fork_tip_frame_id):
    """
    Return the static transforms from ForqueBody to fork_tip and from CameraBody to
    camera_link
    """
    start_time = rospy.Time(start_time)

    fork_transform = TransformStamped()
    fork_transform.header.stamp = start_time
    fork_transform.header.frame_id = "ForqueBody"
    fork_transform.child_frame_id = fork_tip_frame_id
    fork_transform.transform.translation.x = 0.13347163
    fork_transform.transform.translation.y = -0.02011249
    fork_transform.transform.translation.z =  0.1598728
    fork_transform.transform.rotation.x = 0.31048052
    fork_transform.transform.rotation.y = -0.11364137
    fork_transform.transform.rotation.z = 0.92150748
    fork_transform.transform.rotation.w = -0.20366851

    camera_transform = TransformStamped()
    camera_transform.header.stamp = start_time
    camera_transform.header.frame_id = "CameraBody"
    camera_transform.child_frame_id = "camera_link"
    camera_transform.transform.translation.x = 0.00922705
    camera_transform.transform.translation.y = 0.00726385
    camera_transform.transform.translation.z =  0.01964674
    camera_transform.transform.rotation.x = -0.10422086
    camera_transform.transform.rotation.y = 0.08057444
    camera_transform.transform.rotation.z = -0.69519346
    camera_transform.transform.rotation.w = 0.7061333

    return [fork_transform, camera_transform]


def create_directory_structure(participant_num, food_name, trial_num):
    """
    Creates the necessary folders for this trial if they don't exist. The
    convention is:

    {OUT_DIR}/subject{participant_num}_{food_name}/{trial_num}
    """
    participant_food_dir = os.path.join(OUT_DIR, "subject%d_%s" % (participant_num, food_name))
    if not os.path.exists(participant_food_dir):
        os.makedirs(participant_food_dir)

    trial_dir = os.path.join(participant_food_dir, "%d" % trial_num)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

def rotate_ft_data(force_data_raw, debug=False):
    """
    Rotates the F/T sensor data to align it with the way the F/T sensor is
    screwed onto the robot.
    """
    force_ts, force_xyzs = [], []
    torque_xyzs = []

    for force_msg in force_data_raw:
        force_ts.append(force_msg.header.stamp.to_sec())
        force_xyzs.append([force_msg.wrench.force.x, force_msg.wrench.force.y, force_msg.wrench.force.z])
        torque_xyzs.append([force_msg.wrench.torque.x, force_msg.wrench.torque.y, force_msg.wrench.torque.z])
    force_xyzs = np.array(force_xyzs).T # make them into column vectors
    torque_xyzs = np.array(torque_xyzs).T # make them into column vectors

    ANGLE = 2*np.pi/3
    rotation_matrix = tf.transformations.rotation_matrix(ANGLE, (0,0,1))[0:3,0:3]
    # print(rotation_matrix)
    rotated_forces = np.dot(rotation_matrix, force_xyzs)
    rotated_torques = np.dot(rotation_matrix, torque_xyzs)

    # Create the rotated ros msgs
    force_data = []
    for i in range(len(force_data_raw)):
        force_msg_raw = force_data_raw[i]
        force_msg = copy.deepcopy(force_msg_raw)
        force_msg.wrench.force.x = rotated_forces[0, i]
        force_msg.wrench.force.y = rotated_forces[1, i]
        # Since the rotation is around z, the z doesn't change
        force_msg.wrench.torque.x = rotated_torques[0, i]
        force_msg.wrench.torque.y = rotated_torques[1, i]
        force_data.append(force_msg)

    # Debugging
    if debug:
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True)
        for i in range(len(axes)):
            if i == 0:
                data_matrix = force_xyzs
            else:
                data_matrix = rotated_forces
            for j in range(len(axes[i])):
                axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
                axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
                axes[i][j].plot(force_ts, data_matrix[j, :], linestyle='-', c='k', marker='o', markersize=4)
                axes[i][j].set_xlabel("Elapsed Time (sec)")
                axes[i][j].set_ylabel("Force %s (N) %s" % ("XYZ"[j],"raw" if i == 0 else "rotated"))
                axes[i][j].grid(visible=True, which='both')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return force_data

def transform_stamped_to_csv_row(msg):
    """
    Converts the TransformStamped message to a CSV row for the following
    header: "Parent Frame", "Child Frame", "Transform X (m)", "Transform Y (m)",
    "Transform Z (m)", "Transform Roll (rad)", "Transform Pitch (rad)",
    "Transform Yaw (rad)",
    """
    row = []
    row.append(msg.header.frame_id)
    row.append(msg.child_frame_id)
    row.append(msg.transform.translation.x)
    row.append(msg.transform.translation.y)
    row.append(msg.transform.translation.z)
    yaw, pitch, roll = quaternion_msg_to_euler(msg.transform.rotation)
    row.append(roll)
    row.append(pitch)
    row.append(yaw)
    return row

def process_rosbag(filepath, participant_num, food_name, trial_num, first_rosbag=False, debug=False):
    """
    Reads in the rosbag and does the following:
    (1) Saves every RGB image to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}/{timestamp}_rgb.jpg
    (2) Saves every depth image to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}/{timestamp}_depth.png
    (3) Saves the camera location and the mouth location in the world frame to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}_static_transforms.csv
    (4) Saves every F/T reading and every fork pose reading to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}_wrenches_poses.csv
        Note that this requires a few steps of pre-processing:
        (a) Transform the center of the rigid body (recorded by OptiTrak)
            to the forktip frame.
        (b) Rotate the F/T sensor readings to align with the convention that, when looking
            from the fork handle to the forktines, +z is towards the forktines, +y is left,
            and +x is away from curvature of the forktines.
    (5) If first_rosbag is True, then store the camera parameters in the top-level
        directory.
    """

    base_dir = os.path.join(OUT_DIR, "subject%d_%s/" % (participant_num, food_name))

    # Get the topics
    camera_image_topic = "/camera/color/image_raw/compressed"
    camera_info_topic = "/camera/color/camera_info"
    camera_depth_topic = "/camera/aligned_depth_to_color/image_raw"
    tf_static_topic = "/tf_static"
    force_topic = "/forque/forqueSensor"
    # forque_body_topic = "/vrpn_client_node/ForqueBody/pose"
    tf_topic = "tf"
    topics = [
        camera_image_topic,
        camera_info_topic,
        camera_depth_topic,
        tf_static_topic,
        tf_topic,
        force_topic,
    ]
    desired_parent_frame="TableBody"
    forque_body_frame = "ForqueBody"
    fork_tip_frame_id="fork_tip"
    camera_color_optical_frame_id = "camera_color_optical_frame"
    ft_sensor_frame_id = "ft_sensor"
    mouth_body_frame = "MouthBody"

    # Open the bag
    bag = rosbag.Bag(filepath)
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()

    # Open a tf buffer, and populate it with static transforms
    tf_buffer = tf2_py.BufferCore(rospy.Duration((end_time-start_time)*2.0)) # times 2 is to give it enough space in case some messages were backdated
    initial_static_transforms = get_initial_static_transforms(start_time, fork_tip_frame_id)
    for transform in initial_static_transforms:
        tf_buffer.set_transform_static(transform, "default_authority")

    # Iterate over the msgs in the rosbag
    cv_bridge = CvBridge()
    has_seen_table = False
    camera_info = None
    force_data_raw = []
    forque_transform_data_raw = []
    for topic, msg, timestamp in bag.read_messages(topics=topics):
        if topic == camera_image_topic:
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ts = msg.header.stamp.to_nsec()
            cv2.imwrite(os.path.join(base_dir, "%d/%d_rgb.jpg" % (trial_num, ts)), img)
        elif topic == camera_depth_topic:
            img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            ts = msg.header.stamp.to_nsec()
            cv2.imwrite(os.path.join(base_dir, "%d/%d_depth.png" % (trial_num, ts)), img)
            if debug:
                # Verify that the cv2 doesn't change the units from mm when it saves the depth image
                assert(np.isclose(img, cv2.imread(os.path.join(base_dir, "%d/%d_depth.png" % (trial_num, ts)), cv2.IMREAD_ANYDEPTH)).all())
        elif topic == tf_static_topic:
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, "default_authority")
        elif topic == tf_topic:
            for transform in msg.transforms:
                if transform.child_frame_id == desired_parent_frame:
                    has_seen_table = True
                tf_buffer.set_transform(transform, "default_authority")
                # if transform.child_frame_id == desired_parent_frame or transform.child_frame_id == forque_body_frame:
                #     print(start_time, transform)
                if has_seen_table and transform.child_frame_id == forque_body_frame:
                    forktip_transform = tf_buffer.lookup_transform_core(desired_parent_frame,
                        fork_tip_frame_id, rospy.Time(0))
                    forque_transform_data_raw.append(forktip_transform)
        elif topic == force_topic:
            force_data_raw.append(msg)
        elif topic == camera_info_topic and camera_info is None:
            camera_info = msg

    # Rotate the F/T readings
    force_data = rotate_ft_data(force_data_raw, debug=False)

    # Store force data and forktip data in a CSV
    header = [
        "Time (sec)",
        "Force X (N)",
        "Force Y (N)",
        "Force Z (N)",
        "Torque X (Nm)",
        "Torque Y (Nm)",
        "Torque Z (Nm)",
        "Forktip Pose X (m)",
        "Forktip Pose Y (m)",
        "Forktip Pose Z (m)",
        "Forktip Pose Roll (rad)",
        "Forktip Pose Pitch (rad)",
        "Forktip Pose Yaw (rad)",
    ]
    data = []
    datasets = (force_data, forque_transform_data_raw)
    i = [0, 0] # indices into each of the datasets
    while i[0] < len(datasets[0]) or i[1] < len(datasets[1]):
        # Determine which dataset to use
        if i[0] == len(datasets[0]): # all remaining messages are forque pose
            dataset_to_increment = 1
        elif i[1] == len(datasets[1]): # all remaining messages are force data
            dataset_to_increment = 0
        else: # increment the dataset with the earlier timestamp
            ts_0 = datasets[0][i[0]].header.stamp.to_sec()
            ts_1 = datasets[1][i[1]].header.stamp.to_sec()
            if ts_0 <= ts_1:
                dataset_to_increment = 0
            else:
                dataset_to_increment = 1

        # Store the data in a CSV row
        msg = datasets[dataset_to_increment][i[dataset_to_increment]]
        row = [msg.header.stamp.to_sec()]
        if dataset_to_increment == 0:
            row.append(msg.wrench.force.x)
            row.append(msg.wrench.force.y)
            row.append(msg.wrench.force.z)
            row.append(msg.wrench.torque.x)
            row.append(msg.wrench.torque.y)
            row.append(msg.wrench.torque.z)
            for _ in range(6): row.append("")
        else:
            for _ in range(6): row.append("")
            row.append(msg.transform.translation.x)
            row.append(msg.transform.translation.y)
            row.append(msg.transform.translation.z)
            yaw, pitch, roll = quaternion_msg_to_euler(msg.transform.rotation)
            row.append(roll)
            row.append(pitch)
            row.append(yaw)
        data.append(row)

        # Increment the dataset
        i[dataset_to_increment] += 1
    with open(os.path.join(base_dir, "%d_wrenches_poses.csv" % (trial_num,)), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(data)

    # Store all static transforms in a separate CSV
    # Parent Frame to camera_color_optical_frame
    camera_optical_transform = tf_buffer.lookup_transform_core(desired_parent_frame,
        camera_color_optical_frame_id, rospy.Time(0))
    camera_optical_transform.header.frame_id = "world" # rename the world frame
    mouth_transform = tf_buffer.lookup_transform_core(desired_parent_frame,
        mouth_body_frame, rospy.Time(0))
    mouth_transform.header.frame_id = "world" # rename the world frame
    ft_sensor_transform = fork_transform = TransformStamped()
    ft_sensor_transform.header.stamp = start_time
    ft_sensor_transform.header.frame_id = fork_tip_frame_id
    ft_sensor_transform.child_frame_id = ft_sensor_frame_id
    ft_sensor_transform.transform.translation.x = 0.003
    ft_sensor_transform.transform.translation.y = 0.0013
    ft_sensor_transform.transform.translation.z = -0.065
    ft_sensor_transform.transform.rotation.x = 0
    ft_sensor_transform.transform.rotation.y = 0
    ft_sensor_transform.transform.rotation.z = 0
    ft_sensor_transform.transform.rotation.w = 1
    header = [
        "Parent Frame",
        "Child Frame",
        "Transform X (m)",
        "Transform Y (m)",
        "Transform Z (m)",
        "Transform Roll (rad)",
        "Transform Pitch (rad)",
        "Transform Yaw (rad)",
    ]
    with open(os.path.join(base_dir, "%d_static_transforms.csv" % (trial_num,)), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerow(transform_stamped_to_csv_row(camera_optical_transform))
        csvwriter.writerow(transform_stamped_to_csv_row(ft_sensor_transform))
        csvwriter.writerow(transform_stamped_to_csv_row(mouth_transform))

    # Camera Info
    if first_rosbag:
        header = [
            "Frame ID",
            "Width",
            "Height",
            "Distortion Model",
            "Fx",
            "Fy",
            "Cx",
            "Cy",
        ]
        with open(os.path.join(OUT_DIR, "camera_intrinsics.csv"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerow([
                camera_info.header.frame_id,
                camera_info.width,
                camera_info.height,
                camera_info.distortion_model,
                camera_info.K[0],
                camera_info.K[4],
                camera_info.K[2],
                camera_info.K[5],
            ])

if __name__ == "__main__":

    rosbags_to_skip = [
        "1-fries-1_2021-11-23-13-55-22.bag", # tf2.LookupException: "TableBody" passed to lookupTransform argument target_frame does not exist.
        "2-mashedpotato-2_2021-11-23-14-51-48.bag", # tf2.ExtrapolationException: Lookup would require extrapolation at time 1637707910.153995991, but only time 1637707910.203967571 is in the buffer, when looking up transform from frame [fork_tip] to frame [TableBody]
        "2-riceandbeans-4_2021-11-23-14-54-58.bag", # tf2.ExtrapolationException: Lookup would require extrapolation at time 1637708100.653999090, but only time 1637708100.663951159 is in the buffer, when looking up transform from frame [fork_tip] to frame [TableBody]
        "3-mashedpotato-2_2021-11-23-18-01-03.bag", # tf2.ExtrapolationException: Lookup would require extrapolation at time 1637719263.953940630, but only time 1637719263.963970900 is in the buffer, when looking up transform from frame [fork_tip] to frame [TableBody]
        "3-spinach-1_2021-11-23-18-06-44.bag", # tf2.LookupException: "TableBody" passed to lookupTransform argument target_frame does not exist.
        "3-jello-1_2021-11-23-17-58-27.bag", # tf2.ExtrapolationException: Lookup would require extrapolation at time 1637719108.423963070, but only time 1637719108.433976412 is in the buffer, when looking up transform from frame [fork_tip] to frame [TableBody]
        "4-sandwich-2_2021-11-23-16-00-22.bag", # tf2.ExtrapolationException: Lookup would require extrapolation at time 1637712022.933997631, but only time 1637712022.943970680 is in the buffer, when looking up transform from frame [fork_tip] to frame [TableBody]
        "4-noodles-1_2021-11-23-16-02-06.bag", # tf2.LookupException: "TableBody" passed to lookupTransform argument target_frame does not exist.
        "5-jello-1_2021-11-24-10-12-17.bag", # tf2.LookupException: "TableBody" passed to lookupTransform argument target_frame does not exist.
        "6-pizza-2_2021-11-24-12-43-37.bag", # tf2.ExtrapolationException: Lookup would require extrapolation at time 1637786617.673836946, but only time 1637786617.683793783 is in the buffer, when looking up transform from frame [fork_tip] to frame [TableBody]
    ]
    start_at_i = 330

    # Find all the rosbags in IN_DIR
    total_rosbags = 0
    rosbag_paths = {} # participant_num -> food_name -> trial_num -> filepath
    for root, subfolders, files in os.walk(IN_DIR):
        for filename in files:
            if filename.endswith(".bag") and "pilot" not in root and filename not in rosbags_to_skip:
                try:
                    participant_num, food_name, trial_num = parse_rosbag_name(filename)
                except Exception as e:
                    print(e)
                    print("Skipping rosbag %s" % filename)
                    continue
                if participant_num not in rosbag_paths:
                    rosbag_paths[participant_num] = {}
                if food_name not in rosbag_paths[participant_num]:
                    rosbag_paths[participant_num][food_name] = {}
                if trial_num in rosbag_paths[participant_num][food_name]:
                    raise Exception("Trial %d occurs multiple times: %s" % (trial_num, os.path.join(root, filename)))
                rosbag_paths[participant_num][food_name][trial_num] = os.path.join(root, filename)
                total_rosbags += 1
    pprint.pprint(rosbag_paths)

    # Process each rosbag
    processed_rosbags = -1
    for participant_num in rosbag_paths:
        for food_name in rosbag_paths[participant_num]:
            for trial_num in rosbag_paths[participant_num][food_name]:
                processed_rosbags += 1
                if processed_rosbags < start_at_i: continue
                filepath = rosbag_paths[participant_num][food_name][trial_num]
                print("%d/%d" % (processed_rosbags, total_rosbags), filepath)

                create_directory_structure(participant_num, food_name, trial_num)

                try:
                    process_rosbag(filepath, participant_num, food_name, trial_num, processed_rosbags==0)
                except Exception as e:
                    traceback.print_exc()
                    print(e)
