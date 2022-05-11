from ctypes import sizeof
import rosbag, sys
import time
import os #for file management make directory
import rosbag
import rospy
from matplotlib import pyplot as plt


#thresholds
height_threshold = 0.03 # assuming in newtons

#verify correct input arguments: 1 or 2
if (len(sys.argv) > 2):
	print("invalid number of arguments:   " + str(len(sys.argv)))
	print("should be 2: 'get_metadata_csv.py' and 'bagName'")
	print("or just 1  : 'get_metadata_csv.py'")
	sys.exit(1)
elif (len(sys.argv) == 2):
    listOfBagFiles = [sys.argv[1]]	#get list of only bag files in current dir.
    numberOfFiles = str(len(listOfBagFiles))
    print("reading only 1 bagfile: " + str(listOfBagFiles[0]))
elif (len(sys.argv) == 1):
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
    best_t = rospy.Time()
    timestamps = []
    heights = []
    start_time = 0;
    #timestamps.append(best_t.to_sec())
    for topicName in listOfTopics:
        if topicName == '/vrpn_client_node/TableBody/pose':
                # set the table pose using a msg 75% through the bag
                msg_list = list(bag.read_messages(topicName))
                index_percentile_75th = int(0.75*len(msg_list))
                i_subtopic, i_msg, i_t = msg_list[index_percentile_75th]
                if (table_pose == None):
                    table_pose = i_msg.pose

    for topicName in listOfTopics:
        if topicName == '/vrpn_client_node/ForqueBody/pose':
            # set the table pose the first time
            for subtopic, msg, t in bag.read_messages(topicName):
                #print(msg)
                height = msg.pose.position.y - table_pose.position.y  
                print(height)
                timestamps.append(t.to_sec())
                heights.append(height)
            
                        
    plt.title(bagName) 
            # Show a legend on the plot 
                #Saving the plot as an image
    plt.plot(timestamps, heights)
    plt.savefig(bagName+'.png')
    plt.show()        
    bag.close()
print("Done reading all " + numberOfFiles + " bag files.")
