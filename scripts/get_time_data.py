import rosbag, sys, csv
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
import rosbag
import numpy as np

# For now, manually specify which topic contains the image messages.

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
        
        for topicName in listOfTopics:
            if topicName == '/vrpn_client_node/ForqueBody/pose':
                print("Extracting topic: " + topicName)
                #Create a new CSV file for each topic
                    
                firstIteration = True	#allows header row
                current = None
                previous = None
                time_lost = 0
                current
                start_time = 0 # value should never be zero if reading a valid bag file
                for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
                    #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                    #	- put it in the form of a list of 2-element lists
                    current = t
                    if previous == None:  # first real value
                        start_time = current.to_sec()
                        previous = t
                    else:  
                        difference = current.to_sec() - previous.to_sec()
                        if difference >= 0.1:
                            time_lost += difference
                        previous = current
                            
                filewriter.writerow([bagName,start_time,time_lost,current.to_sec(),current.to_sec()-start_time])                
        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")