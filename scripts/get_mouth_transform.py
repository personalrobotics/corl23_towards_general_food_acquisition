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
        mouth_pose = None
        best_t = rospy.Time()
        start_time = 0;
        for topicName in listOfTopics:
            if topicName == '/vrpn_client_node/TableBody/pose':
                # set the table pose the first time
                for subtopic, msg, t in bag.read_messages(topicName):
                    if (table_pose == None):
                        table_pose = msg.pose
                        break;
            elif topicName == '/vrpn_client_node/MouthBody/pose':
                for subtopic, msg, t in bag.read_messages(topicName):
                    if (mouth_pose == None):
                        mouth_pose = msg.pose
                        break;
        print(table_pose)
        print(mouth_pose)
        table_pykdl_frame = pm.fromMsg(table_pose).Inverse();
        mouth_pykdl_frame = pm.fromMsg(mouth_pose)
        mouth_to_table_transform = (table_pykdl_frame * mouth_pykdl_frame)
        print (mouth_to_table_transform)

        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")