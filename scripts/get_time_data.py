import rosbag, sys, csv
import time
import os #for file management make directory
import rosbag


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

        
        for topicName in listOfTopics:
            if topicName == '/vrpn_client_node/ForqueBody/pose':
                print("Extracting topic: " + topicName)
                current = None
                previous = None
                time_lost = 0
                current
                start_time = 0 # value should never be zero if reading a valid bag file
                for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
                    current = t
                    if previous == None:  # first real value
                        start_time = current.to_sec()
                        previous = t
                    else:  
                        difference = current.to_sec() - previous.to_sec()
                        if difference >= 0.1:
                            time_lost += difference
                        previous = current
                # todo maybe: change name of bag file to just the participant, trial, and food            
                filewriter.writerow([bagName,start_time,time_lost,current.to_sec(),current.to_sec()-start_time])                
        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")