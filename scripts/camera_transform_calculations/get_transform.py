#!/usr/bin/env python
import csv
import cv2
import math
import numpy as np
import os
import pandas as pd
import pprint
import rosbag
import rospy
import tf.transformations as transformations

if __name__ == "__main__":
    basedir = "/workspace/rosbags/expanded_action_space_study/"
    filenames = [
        "6-noodles-4_2021-11-24-12-39-57",
        # "9-chicken-4_2021-11-24-16-40-03", # doesn't work well
        "1-spinach-4_2021-11-23-13-42-15",
        "3-riceandbeans-5_2021-11-23-17-43-50",
        # "8-broccoli-5_2021-11-24-15-45-19", # doesn't work well
        # "7-fries-5_2021-11-24-15-10-04", # doesn't work well
        "5-broccoli-4_2021-11-24-09-58-50",
    ]

    rospy.init_node('get_transform')

    # rosrun tf tf_echo camera_color_optical_frame camera_link
    camera_internal_transform = transformations.quaternion_matrix([0.505, -0.496, 0.500, 0.498])
    camera_internal_transform[0,3] = 0.015


    rvec, tvec = None, None
    got_first_rvec = False

    transforms = []

    for filename in filenames:
        print(filename)
        data = pd.read_csv(os.path.join(basedir, filename+".csv"))
        correspondances = data[~data['Forktip X (RealSense)'].isnull()]
        object_points = []
        image_points = []
        camera_matrix = None
        for _, row in correspondances.iterrows():
            object_points.append([
                row["Forktip Position X (OptiTrak)"],
                row["Forktip Position Y (OptiTrak)"],
                row["Forktip Position Z (OptiTrak)"],
            ])
            image_points.append([
                row["Forktip X (RealSense)"],
                row["Forktip Y (RealSense)"],
            ])
            if camera_matrix is None:
                camera_matrix =np.array([
                    [row["RealSense Fx"], 0, row["RealSense Cx"]],
                    [0, row["RealSense Fy"], row["RealSense Cy"]],
                    [0, 0, 1],
                ])
        object_points = np.array(object_points)
        image_points = np.array(image_points)

        # Get the rotation and translation of the OptiTrak origin in the RealSense optical frame
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None)#, rvec, tvec, got_first_rvec)
        if not success:
            raise Exception("solvePnP Failed! %s" % success)
        got_first_rvec = True
        rotation_matrix = cv2.Rodrigues(rvec)[0]
        print(rotation_matrix, tvec)

        # Invert it to get the RealSense optical frame pose in the OptiTrak frame
        realsense_rotation = rotation_matrix.transpose()
        realsense_translation = np.matmul(-realsense_rotation, tvec)
        realsense_transform = np.identity(4)
        realsense_transform[0:3,0:3] = realsense_rotation
        realsense_transform[0:3,3:] = realsense_translation
        print(realsense_transform)

        # D times a vector in the RealSense optical frame gives the vector in the OptiTrak frame
        D = realsense_transform

        total_transform = np.zeros((7,))
        num_transforms = 0
        for _, row in correspondances.iterrows():
            # C times a vector in CameraBody frame will give the vector in OptiTrak frame
            C = transformations.quaternion_matrix(np.array([
                row["RealSense Orientation X (OptiTrak)"],
                row["RealSense Orientation Y (OptiTrak)"],
                row["RealSense Orientation Z (OptiTrak)"],
                row["RealSense Orientation W (OptiTrak)"],
            ]))
            C[0][3] = row["RealSense Position X (OptiTrak)"]
            C[1][3] = row["RealSense Position Y (OptiTrak)"]
            C[2][3] = row["RealSense Position Z (OptiTrak)"]

            # E is the final transform we're looking for.
            # # E times a vector in CameraBody frame will give the vector in RealSense optical frame
            # E = np.matmul(np.linalg.inv(camera_internal_transform), np.matmul(np.linalg.inv(D), C))

            # E times a vector in camera_link frame will give the vector in CameraBody frame
            E = np.matmul(np.linalg.inv(C), np.matmul(D, camera_internal_transform))
            quat = transformations.quaternion_from_matrix(E)
            trans = E[0:3,3]
            # print(E)
            total_transform += np.concatenate((trans, quat))
            num_transforms += 1

        final_transform = total_transform / num_transforms
        print(final_transform)
        transforms.append(final_transform)

    print("AGGREGATE!")
    aggregate_transform = np.mean(transforms, axis=0)
    print(aggregate_transform)
    # print(np.std(transforms, axis=0))

    aggregateFourByFour = transformations.quaternion_matrix(aggregate_transform[3:])
    aggregateFourByFour[0:3,3] = aggregate_transform[0:3]
    print(aggregateFourByFour)
