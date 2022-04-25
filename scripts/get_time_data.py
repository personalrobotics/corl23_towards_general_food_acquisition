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

count = 0
for bagFile in listOfBagFiles:
    count += 1
    print("reading file " + str(count) + " of  " + numberOfFiles + ": " + bagFile)
    #access bag
    bag = rosbag.Bag(bagFile)
    bagContents = bag.read_messages()
    bagName = bag.filename


	#create a new directory
    folder = bagName.rstrip('.bag')
    try:	#else already exists
        os.makedirs(folder)
    except:
        pass
    shutil.copyfile(bagName, folder + '/' + bagName)


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
            filename = folder + '/' + topicName.replace('/', '_') + '.csv'
            with open(filename, 'w+') as csvfile:
                filewriter = csv.writer(csvfile, delimiter = ',')
                firstIteration = True	#allows header row
                current = None
                previous = None
                time_lost = 0
                current
                for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
                    #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                    #	- put it in the form of a list of 2-element lists
                    current = t
                    #write the first row from the first element of each pair
                    if firstIteration:	# header
                        headers = ["filename","timelosttracking","endtime"]	#first column header
                        #for pair in instantaneousListOfData:
                        #    headers.append(pair[0])
                        filewriter.writerow(headers)
                        firstIteration = False
                    elif previous == None:  # first real value
                        previous = t
                    else:  
                        difference = current.to_sec() - previous.to_sec()
                        if difference >= 0.1:
                            time_lost += difference
                        previous = current
                    
                filewriter.writerow([bagName,time_lost,current.to_nsec])                
    bag.close()
print("Done reading all " + numberOfFiles + " bag files.")