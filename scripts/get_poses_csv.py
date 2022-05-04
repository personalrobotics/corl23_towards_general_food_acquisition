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
    headers = ["file_name", "table_position_x", "table_position_y", "table_position_z", "table_orientation_x", "table_orientation_y", "table_orientation_z", "table_orientation_w",
            "mouth_position_x", "mouth_position_y", "mouth_position_z", "mouth_orientation_x", "mouth_orientation_y",
           "mouth_orientation_z", "mouth_orientation_w", "camera_position_x", "camera_position_y", "camera_position_z",
           "camera_orientation_x", "camera_orientation_y", "camera_orientation_z", "camera_orientation_w"]
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
        mouth_pose = None
        camera_pose = None
        best_t = rospy.Time()
        start_time = 0;
        current = None
        previous = None
        time_lost = 0
        mouth_position_x, mouth_position_y, mouth_position_z, mouth_orientation_x, mouth_orientation_y, mouth_orientation_z, mouth_orientation_w = [None]*7
        camera_position_x, camera_position_y, camera_position_z, camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w = [None]*7
        table_position_x, table_position_y, table_position_z, table_orientation_x, table_orientation_y, table_orientation_z, table_orientation_w = [None] *7
        for topicName in listOfTopics:
            #print("Extracting topic: " + topicName)

            if topicName == '/vrpn_client_node/TableBody/pose':
                # set the table pose using a msg 75% through the bag
                msg_list = list(bag.read_messages(topicName))
                index_percentile_75th = int(0.75*len(msg_list))
                i_subtopic, i_msg, i_t = msg_list[index_percentile_75th]
                if (table_pose == None):
                    table_pose = i_msg.pose
            elif topicName == '/vrpn_client_node/MouthBody/pose':
                # set the mouth pose using a msg 75% through the bag
                msg_list = list(bag.read_messages(topicName))
                index_percentile_75th = int(0.75*len(msg_list))
                i_subtopic, i_msg, i_t = msg_list[index_percentile_75th]
                if (mouth_pose == None):
                    mouth_pose = i_msg.pose
            elif topicName == '/vrpn_client_node/CameraBody/pose':
                # set the camera pose using a msg 75% through the bag
                msg_list = list(bag.read_messages(topicName))
                index_percentile_75th = int(0.75*len(msg_list))
                i_subtopic, i_msg, i_t = msg_list[index_percentile_75th]
                if (camera_pose == None):
                    camera_pose = i_msg.pose
                

        #table
        table_pykdl_frame = pm.fromMsg(table_pose)
        table_position_x, table_position_y, table_position_z = table_pykdl_frame.p
        table_orientation_x, table_orientation_y, table_orientation_z, table_orientation_w = table_pykdl_frame.M.GetQuaternion()
        #mouth
        mouth_pykdl_frame = pm.fromMsg(mouth_pose)
        mouth_position_x, mouth_position_y, mouth_position_z = mouth_pykdl_frame.p
        mouth_orientation_x, mouth_orientation_y, mouth_orientation_z, mouth_orientation_w = mouth_pykdl_frame.M.GetQuaternion()
        #camera
        camera_pykdl_frame = pm.fromMsg(camera_pose)
        camera_position_x, camera_position_y, camera_position_z = camera_pykdl_frame.p
        camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w = camera_pykdl_frame.M.GetQuaternion()
 

     
        filewriter.writerow([bagName,table_position_x, table_position_y, table_position_z,table_orientation_x, table_orientation_y, table_orientation_z, table_orientation_w,mouth_position_x, mouth_position_y, mouth_position_z, mouth_orientation_x, mouth_orientation_y,
           mouth_orientation_z, mouth_orientation_w, camera_position_x, camera_position_y, camera_position_z, 
           camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w])                
        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")