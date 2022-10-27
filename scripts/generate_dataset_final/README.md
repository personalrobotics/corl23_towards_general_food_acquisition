1. CSV Data Sequence

Each trial's data is stored in a file called `<trial_num>_wrenches_poses.csv`. Each row of this file includes a timestamp, and either a wrench message or a pose message. The data is in the following order, with the first row being a header.

| Time(s) | F_x(N) | F_y(N) | F_z(N) | T_x(Nm) | T_y(Nm) | T_z(Nm) | P_x(m) | P_y(m) | P_z(m) | O_x(rad) | O_y(rad) | O_z(rad) |

F_x, F_y, F_z: Forces detected based on the (left-handed) sensor coordinate.
T_x, T_y, T_z: Torques detected based on the (right-handed) sensor coordinate.
P_x, P_y, P_z: Positions of the forktip in world coordinates.
O_x, O_y, O_z: Euler angles of the forktip in world coordinates.

2. CSV Data Coordinate Frames

The forces and torques are in the local coordinate frame of the F/T sensor. The poses and velocities are in the world frame. The file `<trial_num>_static_transforms.csv` contains static transforms from the: (a) world frame to the camera's optical frame; (b) world frame to the mouth that users were moving the fork towards; and (c) fork_tip frame to the force torque sensor frame. Note that to add (c) to a continuous transform tree, you have to publish the fork_tip pose from the aforementioned CSV as a transform from the world frame to the fork_tip frame.

The image included in the dataset, `frames.png`, shows the frames overlaid onto one of the camera images. Red is +x, green is +y, and blue is +z.

3. CSV Data Rotations

The orientaion follows ZYX Euler convention.

4. Depth Images

Each depth image is stored in the format `<timestamp>_depth.png`. We have a sequence of depth images for each feeding trial for every subject and food item. Although depth images may appear black, when you read them as 16-bit images the pixel values will indicate distance in mm from the camera. A sample command to read these images in python is:

	```
	import cv2
	cv2.imread("/absolute_path_to_parent_folder/subject1_mashedpotato/4/1637704790144165039_depth.png", cv2.IMREAD_ANYDEPTH)
	```

5. RGB Images

Each rgb image is stored in the format `<timestamp>_rgb.jpg`. We have a sequence of rgb images for each feeding trial for every subject and food item.

6. Notes

- Not all users necessarily have all trial numbers (e.g., due to mistrials).
- The sample code, `visualize_frames.py` is written as a ROS Node and has an associated RVIZ configuration file, `visualize_frames.rviz`. Running it requires having ROS Noetic, [creating a catkin workspace](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment), and [creating a ROS package](http://wiki.ros.org/ROS/Tutorials/CreatingPackage) to put the file into. However, even without running it, reading the code could help provide ideas for how to use the dataset. Note that the visualization of the frames included in this dataset, `frames.png`, was generated using this ROS node and RVIZ configuration.
