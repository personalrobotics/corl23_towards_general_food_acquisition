# Camera Transform Calculations

This calculation had a couple of steps:

## Computing the Transform

### rosbag_to_csv.py

This file took in one filename at a time (see lines 15-21), and converted it to a CSV where each row corresponds to a RealSense camera image. The reason for converting the rosbag to a CSV is that we need to manually label the position of the forkip center in (x,y) coordinates in the stored image. This script also stored the images.

### Add (x,y) points for the forktip center
The next step was to open the stored images, manually find the (x,y) coordinates of the forktip center (in those images where it is clear and visible), and add those datapoints to the CSV.

### get_transform.py

The final step involved loading all the labeled lines of the CSVs, running [solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) on it, and doing matrix transformations to get the transform from the CameraBody tf frame to the camera_link tf frame.

## Publishing the transform

`roslaunch feeding_study_cleanup camera_transform.launch`

Open RVIZ and open the config `rviz/expanded_action_space.rviz`

`rosbag play [NAME_OF_ROSBAG]`

This should show the TF tree (you might have to un-check and re-check the TF viz every time you re-play a rosbag), the camera, and the forktip pose.
