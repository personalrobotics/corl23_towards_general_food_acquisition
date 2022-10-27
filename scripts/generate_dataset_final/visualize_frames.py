#!/usr/bin/env python
# This ROS node, written for ROS Noetic, takes in a participant num, food item,
# trial number, and image timestamp, and publishes the images and TF frames.
# The goal of this node is to visually see the TF frames superimposed with the
# image (subscribed to as a camera) in RVIZ.
import csv
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import TransformStamped
import os
import rospy
from sensor_msgs.msg import CameraInfo, Image
import tf.transformations
import tf2_ros

# ROS rotation convention, see https://answers.ros.org/question/53688/euler-angle-convention-in-tf/?answer=53693#post-id-53693
EULER_ORDER = 'rzyx'

if __name__ == "__main__":
    # Directory Structure Information
    base_dir = "/workspace/rosbags/expanded_action_space_study/processed/"
    participant_num = 1
    food_name = "chicken"
    trial_num = 3
    image_timestamp = 1637703870600778341

    # Initialize the node
    rospy.init_node('publish_forque_transform')

    # Read and publish the static transforms
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    transforms = []
    with open(os.path.join(base_dir, 'subject%d_%s/%d_static_transforms.csv' % (participant_num, food_name, trial_num)), 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        for row in reader:
            msg = TransformStamped()
            msg.header.frame_id = row[0]
            msg.child_frame_id = row[1]
            msg.transform.translation.x = float(row[2])
            msg.transform.translation.y = float(row[3])
            msg.transform.translation.z = float(row[4])
            q = tf.transformations.quaternion_from_euler(float(row[7]), float(row[6]), float(row[5]), EULER_ORDER)
            msg.transform.rotation.x = q[0]
            msg.transform.rotation.y = q[1]
            msg.transform.rotation.z = q[2]
            msg.transform.rotation.w = q[3]
            transforms.append(msg)

    # Publish the nearest fork_tip pose as a static transform
    with open(os.path.join(base_dir, 'subject%d_%s/%d_wrenches_poses.csv' % (participant_num, food_name, trial_num)), 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        for row in reader:
            timestamp = float(row[0])
            if row[7] is None or len(row[7]) == 0: # no pose data at this timestamp
                continue
            # Select the first timestamp after the image timestamp
            if timestamp >= image_timestamp/10.0**9:
                print(timestamp)
                msg = TransformStamped()
                msg.header.frame_id = "world"
                msg.child_frame_id = "fork_tip"
                msg.transform.translation.x = float(row[7])
                msg.transform.translation.y = float(row[8])
                msg.transform.translation.z = float(row[9])
                q = tf.transformations.quaternion_from_euler(float(row[12]), float(row[11]), float(row[10]), EULER_ORDER)
                msg.transform.rotation.x = q[0]
                msg.transform.rotation.y = q[1]
                msg.transform.rotation.z = q[2]
                msg.transform.rotation.w = q[3]
                transforms.append(msg)
                break
    broadcaster.sendTransform(transforms)

    # Generate the CameraInfo message
    camera_info = CameraInfo()
    with open(os.path.join(base_dir, 'camera_intrinsics.csv'), 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        row = next(reader)
        camera_info.header.frame_id = row[0]
        camera_info.width = int(row[1])
        camera_info.height = int(row[2])
        camera_info.distortion_model = row[3]
        fx, fy, cx, cy = [float(x) for x in row[4:]]
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Load the image
    bridge = CvBridge()
    img_cv = cv2.imread(os.path.join(base_dir, "subject%d_%s/%d/%d_rgb.jpg" % (participant_num, food_name, trial_num, image_timestamp)))
    img = bridge.cv2_to_imgmsg(img_cv, encoding="passthrough")
    img.header.frame_id = camera_info.header.frame_id

    # Keep publishing the Image and CameraInfo
    camera_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=1)
    img_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        t = rospy.get_rostime()
        camera_info.header.stamp = t
        img.header.stamp = t
        camera_info_pub.publish(camera_info)
        img_pub.publish(img)
        r.sleep()

    pass
