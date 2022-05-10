from matplotlib.cbook import to_filehandle
import rosbag, sys, csv
import time
import os #for file management make directory
import rospy
import numpy as np
import tf_conversions.posemath as pm
import PyKDL

time_lost_threshold = 0.1
distance_threshold = 0.01 #TODO: what are the units here? Assuming it's meters
time_duration_threshold = 0.5

#verify correct input arguments: 1 or 2
if (len(sys.argv) == 1):
    listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]	#get list of only bag files in current dir.
    numberOfFiles = str(len(listOfBagFiles))
    print("reading all " + numberOfFiles + " bagfiles in current directory: \n")
    for f in listOfBagFiles:
        print(f)
    print("\n press ctrl+c in the next 3 seconds to cancel \n")
    time.sleep(3)
else:
    print("bad argument(s): " + str(sys.argv))	#shouldnt really come up
    sys.exit(1)

with open("information.csv", 'w+') as csvfile:
    filewriter = csv.writer(csvfile, delimiter = ',')
    headers = ["file_name","bag_start_time","calculated_start_time","time_lost_tracking","end_time", "total_duration"]	#first column header
    filewriter.writerow(headers)
    count = 0
    for bagFile in listOfBagFiles:
        count += 1
        print("reading file " + str(count) + " of  " + numberOfFiles + ": " + bagFile)
        #access bag
        bag = rosbag.Bag(bagFile)
        bagContents = bag.read_messages()
        bagName = bag.filename


        #get list of topics from the bag
        listOfTopics = []
        for topic, msg, t in bagContents:
            if topic not in listOfTopics:
                listOfTopics.append(topic)

        table_pose = None
        camera_pose = None
        best_t = rospy.Time()
        start_time = 0;
        for topicName in listOfTopics:
            if topicName == '/vrpn_client_node/TableBody/pose':
                # set the table pose the first time
                for subtopic, msg, t in bag.read_messages(topicName):
                    if (table_pose == None):
                        table_pose = msg.pose
                        break;
            elif topicName == '/vrpn_client_node/CameraBody/pose':
                for subtopic, msg, t in bag.read_messages(topicName):
                    if (camera_pose == None):
                        camera_pose = msg.pose
                        break;
        print(table_pose)
        print(camera_pose)
        # transform so we have physical camera in table frame
        table_pykdl_frame = pm.fromMsg(table_pose).Inverse();
        camera_pykdl_frame = pm.fromMsg(camera_pose)
        camera_to_table_transform = (table_pykdl_frame * camera_pykdl_frame)
        print (camera_to_table_transform)
         
        # create realsense in optitrack frame
        vector = PyKDL.Vector(0.00922705, 0.00726385, 0.01964674)
        rotation = PyKDL.Rotation.Quaternion(-0.10422086, 0.08057444, -0.69519346, 0.7061333)
        camera_realsense_pykdl_frame = PyKDL.Frame(rotation,vector)

        realsense_to_table_transform = (camera_to_table_transform * camera_realsense_pykdl_frame)
        print(realsense_to_table_transform)
        camera_position_x, camera_position_y, camera_position_z = realsense_to_table_transform.p
        camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w = realsense_to_table_transform.M.GetQuaternion()

        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")