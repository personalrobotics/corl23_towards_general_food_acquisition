# from *file* import *method*

# Trajectory ID
# Calculated start time
# Calculated end time
# Cumulative time lost tracking
# Position of mouth in table frame
# Position of camera in table frame
# Transform from camera optical frame to table frame
#   For N points:
#       Get fork tip pose from optitrak frame
#       Use the fixed transform between the optitrak pose and the forktip pose
#       Get 2D fork top pose in pixel frame
#       Solve PnP returns: transform from camera optical frame to the Optitrak frame (invert that)
#       Use above and transform from the camera optitrak pose to create a transform from the realsense optitrak frame and the realsense optical frame
