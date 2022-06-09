#!/usr/bin/env python

import rospy
import copy
import csv
from geometry_msgs.msg import PoseStamped, Pose, Point, TransformStamped, Transform, TwistStamped, Twist, Vector3, Quaternion
import numpy as np
import os
from std_msgs.msg import Header
import tf2_ros
import tf_conversions.posemath as pm
import tf.transformations

FOOD_REFERENCE_FRAME_COLUMNS = [
    "Food Reference Frame Translation X",
    "Food Reference Frame Translation Y",
    "Food Reference Frame Translation Z",
    "Food Reference Frame Rotation X",
    "Food Reference Frame Rotation Y",
    "Food Reference Frame Rotation Z",
]

PRE_GRASP_TARGET_OFFSET_COLUMNS = [
    "Pre-Grasp Target Offset X",
    "Pre-Grasp Target Offset Y",
    "Pre-Grasp Target Offset Z",
]

APPROACH_FRAME_COLUMNS = [
    "Approach Frame Rotation X",
    "Approach Frame Rotation Y",
    "Approach Frame Rotation Z",
]

ACTION_SCHEMA_COLUMNS = [
    "Pre-Grasp Initial Utensil Transform Translation X",
    "Pre-Grasp Initial Utensil Transform Translation Y",
    "Pre-Grasp Initial Utensil Transform Translation Z",
    "Pre-Grasp Initial Utensil Transform Rotation X",
    "Pre-Grasp Initial Utensil Transform Rotation Y",
    "Pre-Grasp Initial Utensil Transform Rotation Z",
    "Pre-Grasp Force Threshold",
    "Grasp In-Food Twist Linear X",
    "Grasp In-Food Twist Linear Y",
    "Grasp In-Food Twist Linear Z",
    "Grasp In-Food Twist Angular X",
    "Grasp In-Food Twist Angular Y",
    "Grasp In-Food Twist Angular Z",
    "Grasp Force Threshold",
    "Grasp Torque Threshold",
    "Grasp Duration",
    "Extraction Out-Of-Food Twist Linear X",
    "Extraction Out-Of-Food Twist Linear Y",
    "Extraction Out-Of-Food Twist Linear Z",
    "Extraction Out-Of-Food Twist Angular X",
    "Extraction Out-Of-Food Twist Angular Y",
    "Extraction Out-Of-Food Twist Angular Z",
    "Extraction Duration",
]

EULER_ORDER = 'rxyz'

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

def matrix_to_transform(m):
    transform_msg = Transform()

    q = tf.transformations.quaternion_from_matrix(m)
    transform_msg.translation.x = m[0][3]
    transform_msg.translation.y = m[1][3]
    transform_msg.translation.z = m[2][3]
    transform_msg.rotation.x = q[0]
    transform_msg.rotation.y = q[1]
    transform_msg.rotation.z = q[2]
    transform_msg.rotation.w = q[3]

    return transform_msg

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

def matrix_to_pose(m):
    pose_msg = Pose()

    q = tf.transformations.quaternion_from_matrix(m)
    pose_msg.position.x = m[0][3]
    pose_msg.position.y = m[1][3]
    pose_msg.position.z = m[2][3]
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]

    return pose_msg

def matrix_to_twist(m, duration):
    twist_msg = Twist()

    rx, ry, rz = tf.transformations.euler_from_matrix(m, EULER_ORDER)
    # print(m, ry, duration)
    twist_msg.linear.x = m[0][3]/duration
    twist_msg.linear.y = m[1][3]/duration
    twist_msg.linear.z = m[2][3]/duration
    twist_msg.angular.x = rx/duration
    twist_msg.angular.y = ry/duration
    twist_msg.angular.z = rz/duration

    return twist_msg

def quaternion_msg_to_euler(quaternion_msg, prev_euler=None):
    q = [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]
    x, y, z = tf.transformations.euler_from_quaternion(q, EULER_ORDER)
    # Fix wraparound for continuity. Only for x and z since in the xyz order y is [-pi/2, pi/2]
    if prev_euler is not None:
        threshold = 0.75
        if (prev_euler[0] > 0) and (x < 0) and ((prev_euler[0] - x) > 2*np.pi*threshold):
            # print("A", prev_euler, x, y, z)
            x += np.pi*2
        elif (prev_euler[0] < 0) and (x > 0) and ((x - prev_euler[0]) > 2*np.pi*threshold):
            # print("B", prev_euler, x, y, z)
            x -= np.pi*2
        if (prev_euler[2] > 0) and (z < 0) and ((prev_euler[2] - z) > 2*np.pi*threshold):
            # print("C", prev_euler, x, y, z)
            z += np.pi*2
        elif (prev_euler[2] < 0) and (z > 0) and ((z - prev_euler[2]) > 2*np.pi*threshold):
            # print("D", prev_euler, x, y, z)
            z -= np.pi*2
    return [x, y, z]

def euler_to_quaternion_msg(euler):
    # Revert wraparound to previous standard
    euler = copy.copy(euler)
    if euler[0] > np.pi:
        euler[0] -= np.pi*2
    elif euler[0] < np.pi:
        euler[0] += np.pi*2
    if euler[2] > np.pi:
        euler[2] -= np.pi*2
    elif euler[2] < np.pi:
        euler[2] += np.pi*2
    qx, qy, qz, qw = tf.transformations.quaternion_from_euler(*euler, axes=EULER_ORDER)
    msg = Quaternion()
    msg.x = qx
    msg.y = qy
    msg.z = qz
    msg.w = qw
    return msg

def linear_average_transform(t0, t1, alpha):
    if t0.header.frame_id != t1.header.frame_id:
        print("ERROR? linear_average_transform have different frame_ids %s and %s" % (t0.header.frame_id, t1.header.frame_id))
    if t0.child_frame_id != t1.child_frame_id:
        print("ERROR? linear_average_transform have different child_frame_ids %s and %s" % (t0.child_frame_id, t1.child_frame_id))
    retval = TransformStamped()
    retval.header.frame_id = t0.header.frame_id
    retval.header.stamp = rospy.Time((1-alpha)*t0.header.stamp.to_sec() + alpha*t1.header.stamp.to_sec())
    retval.child_frame_id = t0.child_frame_id
    retval.transform.translation.x = (1-alpha)*t0.transform.translation.x + alpha*t1.transform.translation.x
    retval.transform.translation.y = (1-alpha)*t0.transform.translation.y + alpha*t1.transform.translation.y
    retval.transform.translation.z = (1-alpha)*t0.transform.translation.z + alpha*t1.transform.translation.z
    t0_euler = quaternion_msg_to_euler(t0.transform.rotation)
    t1_euler = quaternion_msg_to_euler(t1.transform.rotation)
    retval_euler = [
        (1-alpha)*t0_euler[0] + alpha*t1_euler[0],
        (1-alpha)*t0_euler[1] + alpha*t1_euler[1],
        (1-alpha)*t0_euler[2] + alpha*t1_euler[2],
    ]
    retval.transform.rotation = euler_to_quaternion_msg(retval_euler)

    return retval

# NOTE: If I make any significant changes to this function I should probably
# make them to the same function in detect_food_bounding_box.py as well!!!
def apply_twist(start_transform, approach_frame, food_reference_frame, twist, duration, granularity):
    """
    Let F be the frame that start_transform is in. Then, approach_frame must have
    as its parent_frame F. And then the angular velocity of the twist will be
    interpreted in the start_transform frame, but the linear velocity will be
    interpreted in the approach_frame.
    """
    retval = []
    parent_to_start_transform_matrix = transform_to_matrix(start_transform.transform)
    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
    food_to_approach_matrix = transform_to_matrix(approach_frame.transform)
    start_transform_to_approach_matrix = tf.transformations.concatenate_matrices(tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_start_transform_matrix), parent_to_food_matrix), food_to_approach_matrix)

    for i in range(1, granularity+1):
        elapsed_time = duration*float(i)/granularity
        transform = copy.deepcopy(start_transform)
        transform.header.stamp += rospy.Duration(elapsed_time)
        # Apply the angular velocity in the start_transform frame
        elapsed_transform = tf.transformations.euler_matrix(
            twist.twist.angular.x*elapsed_time,
            twist.twist.angular.y*elapsed_time,
            twist.twist.angular.z*elapsed_time,
            EULER_ORDER)
        # # Apply the linear velocity in the start_transform frame
        # elapsed_transform[0][3] = twist.twist.linear.x*elapsed_time
        # elapsed_transform[1][3] = twist.twist.linear.y*elapsed_time
        # elapsed_transform[2][3] = twist.twist.linear.z*elapsed_time
        # Apply the linear velocity in the approach frame
        displacement = np.array([[twist.twist.linear.x], [twist.twist.linear.y], [twist.twist.linear.z]])*elapsed_time
        elapsed_transform[0:3, 3:] = np.dot(start_transform_to_approach_matrix[0:3,0:3], displacement)
        # if i == granularity:
        #     print("B: displacement in approach frame", displacement)
        #     print("B: displacement in start transform frame", elapsed_transform[0:3, 3:])
        #     print("B: transform inv", tf.transformations.inverse_matrix(start_transform_to_approach_matrix))
        #     print("B: start_transform", start_transform)
        #     print("B: approach_frame", approach_frame)
        #     print("B: food_to_approach_matrix", food_to_approach_matrix)

        final_matrix = tf.transformations.concatenate_matrices(parent_to_start_transform_matrix, elapsed_transform)
        transform.transform = matrix_to_transform(final_matrix)
        retval.append(transform)

    return retval

# NOTE: If I make any significant changes to this function I should probably
# make them to the same function in detect_food_bounding_box.py as well!!!
def get_predicted_forque_transform_data(
    action_start_time,
    contact_time,
    food_reference_frame, # geometry_msgs/TransformStamped
    pre_grasp_target_offset, # geometry_msgs/Vector3
    pre_grasp_initial_utensil_transform, # geometry_msgs/PoseStamped
    pre_grasp_force_threshold, # float, newtons
    approach_frame, # geometry_msgs/TransformStamped
    grasp_in_food_twist, # geometry_msgs/TwistStamped
    grasp_force_threshold, # float, newtons
    grasp_torque_threshold, # float, newston-meters
    grasp_duration, # float, secs
    extraction_out_of_food_twist, # geometry_msgs/TwistStamped
    extraction_duration, # float, secs
    fork_tip_frame_id,
    granularity=20,
):
    predicted_forque_transform_data = []

    # Pre-grasp
    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
    food_to_forque = pose_to_matrix(pre_grasp_initial_utensil_transform.pose)
    parent_to_forque_start_matrix = tf.transformations.concatenate_matrices(parent_to_food_matrix, food_to_forque)
    start_forque_transform = TransformStamped()
    start_forque_transform.header.frame_id = food_reference_frame.header.frame_id
    start_forque_transform.header.stamp = rospy.Time(action_start_time)
    start_forque_transform.child_frame_id = fork_tip_frame_id
    start_forque_transform.transform = matrix_to_transform(parent_to_forque_start_matrix)

    food_to_food_offset = Transform()
    food_to_food_offset.translation = pre_grasp_target_offset
    food_to_food_offset.rotation.w = 1
    food_to_food_offset_matrix = transform_to_matrix(food_to_food_offset)
    # print("pre_grasp_target_offset", pre_grasp_target_offset)
    # print("food_to_food_offset_matrix", food_to_food_offset_matrix)
    # print("parent_to_food_matrix", parent_to_food_matrix)
    parent_to_food_offset_matrix = tf.transformations.concatenate_matrices(parent_to_food_matrix, food_to_food_offset_matrix)
    # print("parent_to_food_offset_matrix", parent_to_food_offset_matrix)
    parent_to_forque_end_matrix = np.copy(parent_to_forque_start_matrix)
    parent_to_forque_end_matrix[0][3] = parent_to_food_offset_matrix[0][3]
    parent_to_forque_end_matrix[1][3] = parent_to_food_offset_matrix[1][3]
    parent_to_forque_end_matrix[2][3] = parent_to_food_offset_matrix[2][3]
    end_forque_transform = TransformStamped()
    end_forque_transform.header.frame_id = food_reference_frame.header.frame_id
    end_forque_transform.header.stamp = rospy.Time(contact_time)
    end_forque_transform.child_frame_id = fork_tip_frame_id
    end_forque_transform.transform = matrix_to_transform(parent_to_forque_end_matrix)

    for i in range(granularity+1):
        predicted_forque_transform = linear_average_transform(start_forque_transform, end_forque_transform, float(i)/granularity)
        predicted_forque_transform_data.append(predicted_forque_transform)

    # Grasp
    predicted_forque_transform_data += apply_twist(predicted_forque_transform_data[-1], approach_frame, food_reference_frame, grasp_in_food_twist, grasp_duration, granularity)

    # Extraction
    predicted_forque_transform_data += apply_twist(predicted_forque_transform_data[-1], approach_frame, food_reference_frame, extraction_out_of_food_twist, extraction_duration, granularity)

    return predicted_forque_transform_data

def get_predicted_forque_transform_data_raw(food_reference_frame, approach_frame,
    action_schema_point, action_schema_start_time, action_schema_contact_time,
    action_schema_extraction_time, action_schema_end_time, world_frame_id,
    food_frame_id, fork_tip_frame_id, approach_frame_id, num_symmetric_rotations=1):
    food_reference_frame_msg = TransformStamped(
        Header(0, rospy.get_rostime(), world_frame_id),
        food_frame_id,
        Transform(
            Vector3(*food_reference_frame[0:3]),
            Quaternion(*tf.transformations.quaternion_from_euler(*food_reference_frame[3:], axes=EULER_ORDER)),
        ),
    )
    parent_to_food_matrix = transform_to_matrix(food_reference_frame_msg.transform)
    pre_grasp_initial_utensil_transform = PoseStamped(
        Header(0, rospy.Time.from_sec(action_schema_start_time), food_frame_id),
        Pose(
            Point(*action_schema_point[3:6]),
            Quaternion(*tf.transformations.quaternion_from_euler(*action_schema_point[6:9], axes=EULER_ORDER)),
        ),
    )
    # Of the different symmetry options, pick the one that has the most positive x and z in the world frame. X is the right side, z is towards the user
    food_to_forque = pose_to_matrix(pre_grasp_initial_utensil_transform.pose)
    max_dist = None
    max_food_to_forque_rotated = None
    for i in range(num_symmetric_rotations):
        theta = 2*np.pi/num_symmetric_rotations*i
        theta_matrix = tf.transformations.rotation_matrix(theta, [0,0,1])
        food_to_forque_rotated = tf.transformations.concatenate_matrices(theta_matrix, food_to_forque)
        parent_to_forque_rotated = tf.transformations.concatenate_matrices(parent_to_food_matrix, food_to_forque_rotated)
        forque_offset_from_food = parent_to_forque_rotated[0:3,3] - parent_to_food_matrix[0:3,3]
        print("theta", theta, "forque_offset_from_food", forque_offset_from_food)
        # dist = (forque_offset_from_food[0] + forque_offset_from_food[2])/2 # a heuristic to ick the rotation with the most positive x and z
        dist = forque_offset_from_food[0]
        # dist = i == 2
        if max_dist is None or dist > max_dist:
            max_dist = dist
            max_food_to_forque_rotated = food_to_forque_rotated
    pre_grasp_initial_utensil_transform.pose = matrix_to_pose(max_food_to_forque_rotated)

    return get_predicted_forque_transform_data(
        action_schema_start_time,
        action_schema_contact_time,
        food_reference_frame_msg,
        Vector3(*action_schema_point[0:3]),
        pre_grasp_initial_utensil_transform,
        action_schema_point[9],
        TransformStamped(
            Header(0, rospy.get_rostime(), food_frame_id),
            approach_frame_id,
            Transform(
                Vector3(0, 0, 0),
                Quaternion(*tf.transformations.quaternion_from_euler(*approach_frame, axes=EULER_ORDER)),
            ),
        ),
        TwistStamped(
            Header(0, rospy.Time.from_sec(action_schema_contact_time), fork_tip_frame_id),
            Twist(
                Vector3(*action_schema_point[10:13]),
                Vector3(*action_schema_point[13:16]),
            ),
        ),
        action_schema_point[16],
        action_schema_point[17],
        action_schema_point[18],
        TwistStamped(
            Header(0, rospy.Time.from_sec(action_schema_extraction_time), fork_tip_frame_id),
            Twist(
                Vector3(*action_schema_point[19:22]),
                Vector3(*action_schema_point[22:25]),
            ),
        ),
        action_schema_point[25],
        fork_tip_frame_id,
        granularity=20,
    )

class VisualizeActionSchema:
    def __init__(self, rosbag_filename, action_schema_filepath, action_schema_without_target_offset=None,
    world_frame_id="TableBody", food_frame_id="detected_food", fork_tip_frame_id="fork_tip", approach_frame_id="approach_frame"):
        # Initialize the subscriber and publishers
        self.sub = rospy.Subscriber("forque_tip", PoseStamped, self.forque_tip_callback)
        self.pub = rospy.Publisher('forque_tip/predicted', PoseStamped, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Load the action_schema CSV
        self.action_schema_point = None
        with open(action_schema_filepath, 'r') as file:
            reader = csv.reader(file)
            header_to_i = None
            for row in reader:
                if header_to_i is None:
                    header_to_i = {row[i] : i for i in range(len(row))}
                    continue
                if row[header_to_i["Bag File Name"]] == rosbag_filename:
                    bag_start_time = float(row[header_to_i["Bag Start Time"]])
                    self.action_schema_start_time = float(row[header_to_i["Action Start Time"]])+bag_start_time
                    self.action_schema_contact_time = float(row[header_to_i["Action Contact Time"]])+bag_start_time
                    self.action_schema_extraction_time = float(row[header_to_i["Action Extraction Time"]])+bag_start_time
                    self.action_schema_end_time = float(row[header_to_i["Action End Time"]])+bag_start_time
                    self.food_reference_frame = [float(row[header_to_i[action_schema_column]]) for action_schema_column in FOOD_REFERENCE_FRAME_COLUMNS]
                    self.approach_frame = [float(row[header_to_i[action_schema_column]]) for action_schema_column in APPROACH_FRAME_COLUMNS]
                    self.action_schema_point = [float(row[header_to_i[action_schema_column]]) for action_schema_column in PRE_GRASP_TARGET_OFFSET_COLUMNS]
                    if action_schema_without_target_offset is None:
                        self.action_schema_point += [float(row[header_to_i[action_schema_column]]) for action_schema_column in ACTION_SCHEMA_COLUMNS]
                    else:
                        self.action_schema_point += action_schema_without_target_offset
                    break
        if self.action_schema_point is None:
            raise Exception("Bagfile %s not found in Action Schema CSV %s! Exiting." % (rosbag_filename, action_schema_filepath))

        # Compute the static transforms
        self.world_to_food_transform = TransformStamped(
            Header(0, rospy.get_rostime(), world_frame_id),
            food_frame_id,
            Transform(
                Vector3(*self.food_reference_frame[0:3]),
                Quaternion(*tf.transformations.quaternion_from_euler(*self.food_reference_frame[3:], axes=EULER_ORDER)),
            ),
        )
        self.food_to_approach_transform = TransformStamped(
            Header(0, rospy.get_rostime(), food_frame_id),
            approach_frame_id,
            Transform(
                Vector3(0, 0, 0),
                Quaternion(*tf.transformations.quaternion_from_euler(*self.approach_frame, axes=EULER_ORDER)),
            ),
        )
        self.has_sent_static_transforms = False

        # Predict the action schema point
        self.predicted_forque_transform_data = get_predicted_forque_transform_data_raw(self.food_reference_frame, self.approach_frame,
            self.action_schema_point, self.action_schema_start_time, self.action_schema_contact_time,
            self.action_schema_extraction_time, self.action_schema_end_time, world_frame_id,
            food_frame_id, fork_tip_frame_id, approach_frame_id)#, num_symmetric_rotations=4)#
        # print("self.predicted_forque_transform_data", self.predicted_forque_transform_data)
        self.predicted_forque_transform_data_i = 0

    def forque_tip_callback(self, msg):
        """
        Takes in the PoseStamped msg for the forque_tip, and published the predicted
        forque_tip pose at that time.
        """
        if not self.has_sent_static_transforms:
            self.tf_static_broadcaster.sendTransform(self.world_to_food_transform)
            self.tf_static_broadcaster.sendTransform(self.food_to_approach_transform)
            self.has_sent_static_transforms = True

        msg_ts = msg.header.stamp
        while self.predicted_forque_transform_data_i < len(self.predicted_forque_transform_data)-1:
            pred_msg_ts = self.predicted_forque_transform_data[self.predicted_forque_transform_data_i].header.stamp
            # print("msg_ts", msg_ts, "pred_msg_ts", pred_msg_ts)
            if pred_msg_ts > msg_ts:
                break
            self.predicted_forque_transform_data_i += 1
        pred_transform = self.predicted_forque_transform_data[self.predicted_forque_transform_data_i]
        pred_transform.child_frame_id = "fork_tip_predicted"
        pred_pose = PoseStamped(
            Header(msg.header.seq, msg.header.stamp, pred_transform.header.frame_id),
            Pose(
                Point(pred_transform.transform.translation.x, pred_transform.transform.translation.y, pred_transform.transform.translation.z),
                pred_transform.transform.rotation,
            ),
        )

        # Publish the food reference frame
        self.tf_broadcaster.sendTransform(pred_transform)
        self.pub.publish(pred_pose)
        pass

if __name__ == '__main__':

    rospy.init_node('visualize_action_schema')

    rosbag_filename = rospy.get_param("~rosbag_filename")
    action_schema_filepath = rospy.get_param("~action_schema_filepath")
    action_schema_without_target_offset = None
    # action_schema_without_target_offset = [ 1.42735648e-02,  5.68762707e-02,  1.25015154e-01,
    #      1.41960553e+00, -1.62101146e-01, -5.35208263e-01,
    #      2.15431287e+00,  7.34112120e-03,  7.94792992e-03,
    #      2.37635237e-02, -2.07871325e-02,  1.20615381e-01,
    #      1.83434186e-02,  2.88160084e+01,  1.12741995e-02,
    #      1.52637625e+00, -5.65014501e-02,  9.26339527e-03,
    #     -2.15628047e-01, -8.16513324e-01, -4.96882190e-01,
    #      6.23994377e-01,  1.84774630e-01]
    # action_schema_without_target_offset = [-1.79700635e-02,  7.20751466e-02,  1.15233797e-01,
    #      2.35034866e+00,  7.08066697e-03, -1.79041788e-01,
    #      2.22214468e+00,  1.21386121e-02,  2.36545605e-02,
    #      3.36445825e-02, -2.59529132e-01,  1.80172783e-01,
    #      3.71610979e-01,  3.67071706e+00,  3.56473236e-03,
    #      2.18756196e+00,  2.17225656e-02,  8.47805376e-02,
    #     -7.33201193e-02, -7.69230977e-01,  9.73354349e-02,
    #      4.11638598e-01,  4.17895855e-01]
    # action_schema_without_target_offset = [-4.48988659e-03,  7.42786683e-02,  1.28244574e-01,
    #      1.67107510e+00, -1.16585778e-01, -2.51065682e-01,
    #      2.39164018e+00,  5.37657931e-03,  3.30279162e-03,
    #      2.55605927e-02,  2.28795389e-02,  1.07412539e-01,
    #      3.50231639e-02,  5.43769928e+01,  1.35776429e-02,
    #      2.03296422e+00, -2.23919319e-02,  6.13663139e-02,
    #     -2.38569877e-01, -1.55086466e+00, -4.73278362e-01,
    #      7.21988940e-01,  1.62982238e-01]
    # action_schema_without_target_offset = [ 7.21984313e-03,  6.39600437e-02,  1.17457759e-01,
    #      2.18275922e+00, -1.88260411e-01, -4.77121385e-01,
    #      2.21320650e+00,  3.65286845e-03,  5.09877573e-03,
    #      2.30846683e-02, -3.17533308e-02,  1.28230485e-01,
    #      6.74867736e-02,  3.96227225e+01,  1.44112416e-02,
    #      1.86068464e+00, -5.39068716e-02,  2.55154870e-02,
    #     -2.33596646e-01, -1.29531505e+00, -5.88209988e-01,
    #      8.73130948e-01,  1.79170537e-01]
    # action_schema_without_target_offset = [ 1.53591619e-03,  5.80759567e-02,  1.18972715e-01,
    #      1.95759101e+00, -1.32696211e-01, -4.45190927e-01,
    #      2.16207493e+00,  4.74823196e-03,  8.65887716e-03,
    #      2.42304040e-02, -4.17596517e-02,  1.05129733e-01,
    #     -6.67664331e-02,  1.78447267e+01,  8.04140417e-03,
    #      1.31139864e+00, -7.04619055e-02,  2.02235877e-02,
    #     -2.05566248e-01, -8.56462335e-01, -4.90230995e-01,
    #      8.20459776e-01,  1.86229338e-01]
    # action_schema_without_target_offset = [ 2.46027995e-02,  6.79099995e-02,  1.38565260e-01,
    #      9.76452235e-01,  4.24348030e-02, -6.51389240e-01,
    #      3.48959855e+01,  1.81183333e-03, -2.51043599e-03,
    #     -2.90368829e-02, -5.68219237e-01,  5.66510196e-02,
    #      1.69579828e-01,  2.96268781e+01,  8.74396658e-03,
    #      6.25030239e-01, -5.95811425e-02,  1.13494224e-01,
    #     -2.58874154e-01, -1.09412233e+00, -7.35469557e-01,
    #      8.14858613e-01,  1.56648596e-01]
    # action_schema_without_target_offset = [-1.28470440e-01,  1.14870080e-01,  1.39477987e-01,
    #      2.81911550e+00,  1.40975928e-01, -2.18209034e-01,
    #      1.93475875e+00, -4.96126119e-03, -4.28161433e-03,
    #      5.54608983e-03,  1.26944870e-02,  2.70423483e-02,
    #      4.60676589e-02,  8.82781128e+01,  1.30901533e-02,
    #      5.31993246e+00,  4.29870912e-02,  5.31214206e-02,
    #     -3.20641191e-01, -2.24875021e+00,  3.84871512e-01,
    #     -3.50300924e-01,  1.20016336e-01]

    visualize_action_schema = VisualizeActionSchema(rosbag_filename, action_schema_filepath, action_schema_without_target_offset=action_schema_without_target_offset)

    rospy.spin()
