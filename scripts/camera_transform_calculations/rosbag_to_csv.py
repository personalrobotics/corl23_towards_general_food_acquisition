#!/usr/bin/env python
import csv
import cv2
from cv_bridge import CvBridge
import math
import numpy as np
import os
import pprint
import rosbag
import rospy
import tf.transformations as transformations

# NOTE: In the OptiTrak frame, if our eyes are the camera, +z is forward, +x is left, and +y is up.
if __name__ == "__main__":
    basedir = "/workspace/rosbags/expanded_action_space_study/"
    filenames = [
        "7-bagels-1_2021-11-24-14-51-50",
        "8-broccoli-5_2021-11-24-15-45-19", # doesn't work well
        "5-broccoli-4_2021-11-24-09-58-50",
        "1-chicken-1_2021-11-23-13-43-33",
        "9-chicken-4_2021-11-24-16-40-03", # doesn't work well
        "2-donutholes-1_2021-11-23-15-09-33",
        "6-doughnutholes-5_2021-11-24-12-42-28",
        "7-fries-5_2021-11-24-15-10-04", # doesn't work well
        "4-jello-1_2021-11-23-16-12-06",
        "6-jello-4_2021-11-24-12-46-39",
        "3-lettuce-4_2021-11-23-18-04-06",
        "4-mashedpotato-4_2021-11-23-16-10-04",
        "6-noodles-4_2021-11-24-12-39-57",
        "2-pizza-4_2021-11-23-15-19-49",
        "3-riceandbeans-5_2021-11-23-17-43-50",
        "1-sandwich-1_2021-11-23-14-00-49",
        "2-sandwich-5_2021-11-23-15-08-52",
        "1-spinach-4_2021-11-23-13-42-15",
    ]
    num_correspondences = 30 # Go for 2x the num you actually want, in case the correspondences for consecutive intervals are temporally adjacent

    # Define the topics
    camera_info_topic = "/camera/color/camera_info"
    camera_image_topic = "/camera/color/image_raw/compressed"
    camera_depth_topic = "/camera/aligned_depth_to_color/image_raw"
    forque_pose_topic = "/vrpn_client_node/ForqueBody/pose"
    camera_pose_topic = "/vrpn_client_node/CameraBody/pose"

    rospy.init_node('rosbag_to_csv')

    for filename in filenames:
        print(filename)

        # Open the rosbag and get its metadata
        bag = rosbag.Bag(os.path.join(basedir, filename+".bag"))
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        interval_size = (end_time - start_time) / num_correspondences

        # Get the points with the nearest temporal distance within each interval
        topics = [
            camera_info_topic,
            camera_image_topic,
            camera_depth_topic,
            forque_pose_topic,
            camera_pose_topic,
        ]
        interval_to_min_temporal_distance = {}
        interval_to_closest_camera_forque_messages = {}
        topic_to_most_recent_message = {}
        for topic, msg, timestamp in bag.read_messages(topics=topics):
            timestamp_float = timestamp.secs + timestamp.nsecs / 10**9
            topic_to_most_recent_message[topic] = (msg, timestamp_float)
            if topic == camera_image_topic:
                # print(msg.header)
                pass
            if topic == forque_pose_topic:
                if camera_image_topic in topic_to_most_recent_message and camera_pose_topic in topic_to_most_recent_message and camera_info_topic in topic_to_most_recent_message:
                    most_recent_camera_ts = topic_to_most_recent_message[camera_image_topic][1]
                    temporal_dist = timestamp_float - most_recent_camera_ts

                    interval = int(math.floor((timestamp_float - start_time) / interval_size))
                    if interval not in interval_to_min_temporal_distance or temporal_dist < interval_to_min_temporal_distance[interval]:
                        interval_to_min_temporal_distance[interval] = temporal_dist
                        if ((camera_image_topic in topic_to_most_recent_message) and
                           (camera_depth_topic in topic_to_most_recent_message) and
                           (forque_pose_topic in topic_to_most_recent_message) and
                           (camera_pose_topic in topic_to_most_recent_message) and
                           (camera_info_topic in topic_to_most_recent_message)):
                            interval_to_closest_camera_forque_messages[interval] = {}
                        # if camera_image_topic in topic_to_most_recent_message:
                            interval_to_closest_camera_forque_messages[interval][camera_image_topic] = topic_to_most_recent_message[camera_image_topic]
                        # if camera_depth_topic in topic_to_most_recent_message:
                            interval_to_closest_camera_forque_messages[interval][camera_depth_topic] = topic_to_most_recent_message[camera_depth_topic]
                        # if forque_pose_topic in topic_to_most_recent_message:
                            interval_to_closest_camera_forque_messages[interval][forque_pose_topic] = topic_to_most_recent_message[forque_pose_topic]
                        # if camera_pose_topic in topic_to_most_recent_message:
                            interval_to_closest_camera_forque_messages[interval][camera_pose_topic] = topic_to_most_recent_message[camera_pose_topic]
                        # if camera_info_topic in topic_to_most_recent_message:
                            interval_to_closest_camera_forque_messages[interval][camera_info_topic] = topic_to_most_recent_message[camera_info_topic]

        # pprint.pprint(interval_to_min_temporal_distance)
        # pprint.pprint(interval_to_closest_camera_forque_messages)
        bag.close()

        # Convert the ForqueBody pose to the forktip pose
        interval_to_forktip_position = {}
        # A times a vector in ForqueBody frame will give a vector in forktip frame
        A = np.array([
            [0.7242579781,	 0.4459343445,	-0.5259778548,	-0.003609486659],
            [-0.3048250761,	 0.8912113169,	 0.3359484526,	 0.00490096214 ],
            [0.6185878081,	-0.08298122464,	 0.7813997912,	-0.2091574497  ],
            [0,	             0,	             0,	             1             ],
        ])
        # A_inv times a vector in forktip frame gives the vector in ForqueBody frame
        A_inv = np.linalg.inv(A)
        for interval in interval_to_closest_camera_forque_messages:
            # B times a vector in ForqueBody frame will give the vector in OptiTrak frame
            B = transformations.quaternion_matrix(np.array([
                interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.orientation.x,
                interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.orientation.y,
                interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.orientation.z,
                interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.orientation.w,
            ]))
            B[0][3] = interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.position.x
            B[1][3] = interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.position.y
            B[2][3] = interval_to_closest_camera_forque_messages[interval][forque_pose_topic][0].pose.position.z
            # [0,0,0,1] in forktip frame is the position of the forktip.
            # Multiplying that by A_inv and B will give the position of the froktip
            # in the OptiTrak frame
            forktip_position = np.matmul(B, np.matmul(A_inv, [0, 0, 0, 1]))
            # print(forktip_position)
            interval_to_forktip_position[interval] = forktip_position.reshape((4,))

        # Save the images
        cv_bridge = CvBridge()
        for interval in interval_to_closest_camera_forque_messages:
            img_msg = interval_to_closest_camera_forque_messages[interval][camera_image_topic][0]
            img_cv = cv2.imdecode(np.frombuffer(img_msg.data, np.uint8), cv2.IMREAD_COLOR)
            img_filename = "%s_%d.jpg" % (filename, math.floor(interval_to_closest_camera_forque_messages[interval][camera_image_topic][1]*1000))
            cv2.imwrite(os.path.join(basedir, img_filename), img_cv)

            # depth_msg = interval_to_closest_camera_forque_messages[interval][camera_depth_topic][0]
            # depth_cv = np.array(cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'), dtype=np.float32)
            # depth_min = np.min(depth_cv)
            # depth_max = np.max(depth_cv)
            # depth_cv -= depth_min
            # depth_cv /= (depth_max - depth_min)
            # depth_cv *= 255
            # # print(depth_cv)
            # depth_colormap = cv2.applyColorMap(depth_cv.astype(np.uint8), cv2.COLORMAP_AUTUMN)
            # # print(np.frombuffer(depth_msg.data, np.uint8))
            # # depth_cv = cv2.imdecode(np.frombuffer(depth_msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            # # print(depth_cv)
            # depth_filename = "%s_%d_depth.jpg" % (filename, math.floor(interval_to_closest_camera_forque_messages[interval][camera_depth_topic][1]*1000))
            # cv2.imwrite(os.path.join(basedir, depth_filename), depth_colormap)

        # Write the correspondences out to a CSV
        header = [
            "Camera Image Timestamp",
            "Forktip X (RealSense)",
            "Forktip Y (RealSense)",
            "ForqueBody Pose Timestamp",
            "Forktip Position X (OptiTrak)",
            "Forktip Position Y (OptiTrak)",
            "Forktip Position Z (OptiTrak)",
            "CameraBody Pose Timestamp",
            "RealSense Position X (OptiTrak)",
            "RealSense Position Y (OptiTrak)",
            "RealSense Position Z (OptiTrak)",
            "RealSense Orientation X (OptiTrak)",
            "RealSense Orientation Y (OptiTrak)",
            "RealSense Orientation Z (OptiTrak)",
            "RealSense Orientation W (OptiTrak)",
            "CameraInfo Timestamp",
            "RealSense Fx",
            "RealSense Fy",
            "RealSense Cx",
            "RealSense Cy",
        ]
        with open(os.path.join(basedir, filename+"_raw.csv"), "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            for interval in interval_to_closest_camera_forque_messages:
                row = [
                    interval_to_closest_camera_forque_messages[interval][camera_image_topic][1],
                    None, # To be filled in manually
                    None, # To be filled in manually
                    interval_to_closest_camera_forque_messages[interval][forque_pose_topic][1],
                    interval_to_forktip_position[interval][0],
                    interval_to_forktip_position[interval][1],
                    interval_to_forktip_position[interval][2],
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][1],
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.position.x,
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.position.y,
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.position.z,
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.orientation.x,
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.orientation.y,
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.orientation.z,
                    interval_to_closest_camera_forque_messages[interval][camera_pose_topic][0].pose.orientation.w,
                    interval_to_closest_camera_forque_messages[interval][camera_info_topic][1],
                    interval_to_closest_camera_forque_messages[interval][camera_info_topic][0].K[0],
                    interval_to_closest_camera_forque_messages[interval][camera_info_topic][0].K[4],
                    interval_to_closest_camera_forque_messages[interval][camera_info_topic][0].K[2],
                    interval_to_closest_camera_forque_messages[interval][camera_info_topic][0].K[5],
                ]
                csvwriter.writerow(row)