from ctypes import sizeof
import rosbag, sys, csv
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
import rosbag
import rospy
import numpy as np
from matplotlib import pyplot as plt

from rospy_message_converter import message_converter

#thresholds
time_lost_threshold = 0.1
distance_threshold = 0.01 #TODO: what are the units here?
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
    headers = ["file_name","start_time","time_lost_tracking","end_time", "total_duration"]	#first column header
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

        # pull from the forquebody topic
            # find the first five seconds of continuous tracking
                # determine how frequently messages are sent
                # calculate the number of messages that should be sent in five seconds
                # use sequence numbers and timestamps to figure out how many messages have been sent in a time period
            # beginning of those five seconds is our start time
        table_pose = None
        best_t = rospy.Time()
        timestamps = []
        start_time = 0;
        #timestamps.append(best_t.to_sec())
        for topicName in listOfTopics:
            if topicName == 'vrpn_client_node/TableBody/pose':
                # set the table pose the first time
                for subtopic, msg, t in bag.read_messages(topicName):
                    if (table_pose == None):
                        table_pose = msg.Pose
                        break;
            elif topicName == '/vrpn_client_node/ForqueBody/pose':
                msg_list = list(bag.read_messages(topicName))
                for i in range(len(msg_list)):	# for each instant in time that has data for topicName
                    if (i == 0):
                        i_subtopic, i_msg, i_t = msg_list[i]
                        start_time = i_t.to_sec()
                    for j in range(i+1,len(msg_list)):
                        # pull this and previous message
                        curr_subtopic, curr_msg, curr_t = msg_list[j]
                        prev_subtopic, prev_msg, prev_t = msg_list[j-1]

                        # determine if time was lost tracking
                        if curr_t.to_sec() - prev_t.to_sec() > time_lost_threshold:
                            break

                        # determine if forque moved too much
                        p0 = np.array([curr_msg.pose.position.x,curr_msg.pose.position.y,curr_msg.pose.position.z])  
                        p1 = np.array([prev_msg.pose.position.x,prev_msg.pose.position.y,prev_msg.pose.position.z])
                        if (np.linalg.norm(p0 - p1) > distance_threshold):
                            break

                        # pull ith message
                        i_subtopic, i_msg, i_t = msg_list[i]
                        if (i == 0):
                            start_time = i_t.to_sec()

                        # did we hit our time threshold
                        if (curr_t.to_sec() - i_t.to_sec() >= time_duration_threshold):
                            # check distance between i and j
                            p0 = np.array([curr_msg.pose.position.x,curr_msg.pose.position.y,curr_msg.pose.position.z])  
                            p1 = np.array([i_msg.pose.position.x,i_msg.pose.position.y,i_msg.pose.position.z])
                            if (np.linalg.norm(p0 - p1) > distance_threshold):
                                break
                            if (curr_t.to_sec() > best_t.to_sec()):
                                # overwrite the best time
                                best_t = curr_t
                                timestamps.append(best_t.to_sec()-start_time)
                        
        plt.title(bagName) 
                # Show a legend on the plot 
                #Saving the plot as an image
        plt.hist(timestamps, 100, facecolor='blue')

        plt.savefig(bagName+'.png')
        #plt.show()        
        print(best_t.to_nsec())
        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")
