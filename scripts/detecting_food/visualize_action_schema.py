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
        """
        If action_schema_without_target_offset is None, it visualizes the extracted
        schema element for the specified bagfile. If action_schema_without_target_offset is
        not None, it appends the target offset from the specified rosbag's extracted
        action schema, but then uses the rest of the action schema that was passed to it.
        This is because the target offset part of the action schema should be perceived,
        whereas the rest should be specified.
        """
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
            food_frame_id, fork_tip_frame_id, approach_frame_id, num_symmetric_rotations=2)#)#
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
    action_schema_without_target_offset = None # get the action schema element for the specified bagfile.
    # # K Medoids Center 0 -- skewer with a long grasp
    # action_schema_without_target_offset = [-0.0241435154, 0.0627733441, 0.0903450192, 2.9335558522, -0.0668797969, -0.6043233543, 1.411675406, 0.0040550042, -0.0020559625, -0.0111952656, -0.0456289485, 0.1015461256, 0.1079535895, 18.1902203375, 0.0053723529, 2.1900441647, 0.0293917653, 0.0172552147, 0.2330012648, -1.4520816436, 0.1509430804, 0.4077787458, 0.169929266]
    # # K Medoids Center 1 -- kinda a slant rotation during grasp
    # action_schema_without_target_offset = [-0.0933095041, 0.0473945431, 0.0746568551, 2.612486284, 0.6774372583, -0.938307514, 1.8661167057, 0.0496440751, -0.0035888136, -0.0194992453, -0.2958625766, 0.2775733786, 0.4960116733, 4.7116844261, 0.0049858205, 0.7699754238, 0.0077459214, 0.0034566257, 0.1386343538, -0.5364469236, 0.0956165579, -0.1095267226, 0.3900365829]
    # # K Medoids Center 2 -- kinda a scoop
    # action_schema_without_target_offset = [0.0194575523, 0.0712459881, 0.0954177189, 2.6825735228, -0.1819881585, 0.1787181453, 2.1258280113, 0.0310197167, 0.0016830375, -0.0268012858, -0.3972107239, 0.1994809825, 0.285544059, 3.594568911, 0.0035010007, 1.1499814987, 0.0078253463, 0.006374296, 0.0581426333, -0.2350001343, 0.0952019699, 0.0056849928, 1.0199453831]
    # # K Medoids Center 3 -- pretty direct skewer
    # action_schema_without_target_offset = [0.0121543255, 0.0487665879, 0.0698090342, 2.9087563801, -0.2215589235, 0.0726021097, 1.9680640809, 0.0171551959, -0.0091361979, -0.0270055141, 0.0384959233, 0.2600723545, 0.049637002, 20.6284852895, 0.0032483905, 1.0700218678, -0.0554067812, 0.0075518517, 0.2068274077, -2.2970612737, 0.145287009, -0.0072780301, 0.2899940014]
    # # K Medoids Center 4 -- skewer, but rotate a bit on extraction. For food items that will fall off.
    # action_schema_without_target_offset = [-0.0270112245, 0.0187378178, 0.0879066299, 2.8642044805, 0.2871018582, -1.1611927634, 2.0707863055, -0.0013812467, -0.0197499396, -0.0287871, 0.0832228896, 0.0917362301, 0.0834890889, 35.3793314916, 0.0076060858, 1.2298948765, -0.0015740456, -0.0283020475, 0.1945183228, -1.5889343937, -0.3155210079, 0.5648036816, 0.3100457191]
    # # K Medoids Center 5 -- pretty standard skewer
    # action_schema_without_target_offset = [0.0473675761, 0.0621456676, 0.0971244872, 2.8033531736, -0.6184566254, 0.2703061197, 2.0181984644, 0.0073753093, 0.0041578542, -0.0183269882, 0.011943316, 0.1597240197, -0.0354191717, 30.9229988029, 0.0055181877, 1.8299505711, 0.0169685109, -0.025293421, 0.177595202, -1.4817376905, -1.033173501, 0.5955795989, 0.2799816132]
    # # Mean for mashed potatoes
    # action_schema_without_target_offset = [-0.0207533154, 0.0590092196, 0.0645446973, 2.4178163749, 0.1691904814, -0.2445237426, 1.9798474465, 0.0336443429, 0.0053413999, -0.0109308029, -0.3413072553, 0.2374914221, 0.6321235195, 3.7726047498, 0.0038493739, 1.1169676853, -0.0012243083, -0.0181489784, 0.1348582514, -0.5994386454, 0.2168043484, 0.1650639896, 0.5184955163]
    # # Mean for noodles
    # action_schema_without_target_offset = [-0.0036061555, 0.0452589689, 0.068923647, 2.3544101226, 0.0256324706, -0.2219648843, 2.347667815, 0.0518702027, 0.0192482036, -0.0070643911, -0.2545528163, 0.1924065908, 0.0132156723, 3.2765067318, 0.002627861, 2.8848222533, 0.0040679476, -0.011395859, 0.1742952075, -1.0920519617, 0.0548015088, 0.5238431571, 0.3777573109]

    visualize_action_schema = VisualizeActionSchema(rosbag_filename, action_schema_filepath, action_schema_without_target_offset=action_schema_without_target_offset)

    rospy.spin()
