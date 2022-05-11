from ctypes import sizeof
import rosbag, sys, csv
import time
import os #for file management make directory
import rosbag
import rospy
import numpy as np
from matplotlib import pyplot as plt


#thresholds
force_threshold = 2.0 # assuming in newtons

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
    norms = []
    start_time = 0;
    #timestamps.append(best_t.to_sec())
    for topicName in listOfTopics:
        if topicName == '/forque/forqueSensor':
            # set the table pose the first time
            for subtopic, msg, t in bag.read_messages(topicName):
                #print(msg)
                force_array = np.array([msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z])  
                l2 = np.linalg.norm(force_array,2)
                timestamps.append(t.to_sec())
                if (l2 < force_threshold):
                    norms.append(l2)
                else: # used to tune what the threshold should be
                    norms.append(l2)

            
                        
    plt.title(bagName) 
            # Show a legend on the plot 
                #Saving the plot as an image
    plt.plot(timestamps, norms)
    plt.savefig(bagName+'.png')
    plt.show()        
    bag.close()
print("Done reading all " + numberOfFiles + " bag files.")
