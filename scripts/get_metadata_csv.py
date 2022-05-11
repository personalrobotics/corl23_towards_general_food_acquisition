import rosbag, sys, csv
import time
import os #for file management make directory
import rospy
import numpy as np
import tf_conversions.posemath as pm
import PyKDL


time_lost_threshold = 0.1
distance_threshold = 0.01 # 1cm
time_duration_threshold = 0.3 
updated_start_time_max_duration = 4 # sec
height_threshold = 0.03 # 3cm
force_threshold = 2.1 # assuming in newtons
force_duration_threshold = 0.3
updated_force_time_max_duration = 1


listOfBagFiles = []
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

with open("information.csv", 'w+') as csvfile:
    filewriter = csv.writer(csvfile, delimiter = ',')
    headers = ["file_name","bag_start_time","calculated_start_time","end_time","time_lost_tracking", "total_duration",
            "mouth_position_x", "mouth_position_y", "mouth_position_z", "mouth_orientation_x", "mouth_orientation_y",
           "mouth_orientation_z", "mouth_orientation_w", "camera_position_x", "camera_position_y", "camera_position_z",
           "camera_orientation_x", "camera_orientation_y", "camera_orientation_z", "camera_orientation_w"]
    filewriter.writerow(headers)
    count = 0
    for bagFile in listOfBagFiles:
        # if "sandwich" not in bagFile: continue
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
        bag_start_time = 0;
        current = None
        previous = None
        time_lost = 0
        force_threshold_timestamp = None
        mouth_position_x, mouth_position_y, mouth_position_z, mouth_orientation_x, mouth_orientation_y, mouth_orientation_z, mouth_orientation_w = [None]*7
        camera_position_x, camera_position_y, camera_position_z, camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w = [None]*7

        for topicName in listOfTopics:
            if topicName == '/forque/forqueSensor':
                msg_list = list(bag.read_messages(topicName))
                for i in range(len(msg_list)):	# for each instant in time that has data for topicName
                    print("i ", i)
                    i_subtopic, i_msg, i_t = msg_list[i]
                    force_array_i = np.array([i_msg.wrench.force.x,i_msg.wrench.force.y,i_msg.wrench.force.z])  
                    l2_i = np.linalg.norm(force_array_i,2)

                    if (l2_i > force_threshold):
                        for j in range(i+1,len(msg_list)):	# for each instant in time that has data for topicName
                            j_subtopic, j_msg, j_t = msg_list[j]
                            force_array_j = np.array([j_msg.wrench.force.x,j_msg.wrench.force.y,j_msg.wrench.force.z])  
                            l2_j = np.linalg.norm(force_array_j,2)
                                # break due to threshold not reached
                            if l2_j < force_threshold:
                                break
                            if (j_t.to_sec() - i_t.to_sec() > force_duration_threshold):
                                if force_threshold_timestamp == None:
                                    force_threshold_timestamp = i_t
                                    print("set force threshold timestamp", force_threshold_timestamp.to_sec())
                                if (j_t.to_sec() - force_threshold_timestamp.to_sec() < updated_force_time_max_duration):
                                    force_threshold_timestamp = i_t
                print("set force threshold timestamp", force_threshold_timestamp.to_sec())
            elif topicName == '/vrpn_client_node/TableBody/pose':
                # set the table pose using a msg 75% through the bag
                msg_list = list(bag.read_messages(topicName))
                index_percentile_75th = int(0.75*len(msg_list))
                i_subtopic, i_msg, i_t = msg_list[index_percentile_75th]
                if (table_pose == None):
                    table_pose = i_msg.pose
                                      
        #bag.close()

        for topicName in listOfTopics:
            #print("Extracting topic: " + topicName)
            if topicName == '/vrpn_client_node/MouthBody/pose':
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
            elif topicName == '/vrpn_client_node/ForqueBody/pose':
                msg_list = list(bag.read_messages(topicName))
                force_threshold_reached = False
                for i in range(len(msg_list)):	# for each instant in time that has data for topicName
                    i_subtopic, i_msg, i_t = msg_list[i]
                    print("i", i, i_t.to_sec())
                    if (i == 0):
                        bag_start_time = i_t.to_sec()
                    if (table_pose == None): break

                    # skip all messages until we are above some height threshold
                    if (i_msg.pose.position.y - table_pose.position.y < height_threshold): continue

                    
                    current = i_t
                    if previous == None:  # first real value
                        previous = i_t
                    else:  
                        difference = current.to_sec() - previous.to_sec()
                        if difference >= 0.1:
                            time_lost += difference
                        previous = current

                    if not force_threshold_reached:
                        for j in range(i+1,len(msg_list)):
                            # pull this and previous message
                            curr_subtopic, curr_msg, curr_t = msg_list[j]
                            prev_subtopic, prev_msg, prev_t = msg_list[j-1]
                            if (curr_t > force_threshold_timestamp):
                                force_threshold_reached = True
                                break;

                            # determine if time was lost tracking
                            if curr_t.to_sec() - prev_t.to_sec() > time_lost_threshold:
                                print("    j", j, curr_t.to_sec())
                                print("    Break due to time lost tracking")
                                break

                            # determine if forque moved too much
                            p0 = np.array([curr_msg.pose.position.x,curr_msg.pose.position.y,curr_msg.pose.position.z])  
                            p1 = np.array([prev_msg.pose.position.x,prev_msg.pose.position.y,prev_msg.pose.position.z])
                            if (np.linalg.norm(p0 - p1) > distance_threshold):
                                print("    j", j, curr_t.to_sec())
                                print("    break due to distance threshold")
                                break

                            # did we hit our time threshold
                            if (curr_t.to_sec() - i_t.to_sec() >= time_duration_threshold):
                                # check distance between i and j
                                p0 = np.array([curr_msg.pose.position.x,curr_msg.pose.position.y,curr_msg.pose.position.z])  
                                p1 = np.array([i_msg.pose.position.x,i_msg.pose.position.y,i_msg.pose.position.z])
                                if (np.linalg.norm(p0 - p1) > distance_threshold):
                                    print("    j", j, curr_t.to_sec())
                                    print("    break due to distance threshold from start")
                                    break
                                if ((curr_t.to_sec() > best_t.to_sec())):
                                    if best_t == rospy.Time():
                                        print("    j", j, curr_t.to_sec())
                                        print("    set best time initial")
                                        best_t = curr_t
                                        time_lost = 0

                            # if j == len(msg_list)-1:
                            #     print("    j", j, curr_t.to_sec())
                            #     print("    reached end of list")
                    else: 
                        print("    broke due force threshold reached")

        if (table_pose != None):
            table_pykdl_frame = pm.fromMsg(table_pose).Inverse();

            if (mouth_pose != None):
                # transform so that we get mouth in table frame
                mouth_pykdl_frame = pm.fromMsg(mouth_pose)
                mouth_to_table_transform = (table_pykdl_frame * mouth_pykdl_frame)
                # extract in the form of pose and quaternion
                # TODO: verify using Amal's code
                mouth_position_x, mouth_position_y, mouth_position_z = mouth_to_table_transform.p
                mouth_orientation_x, mouth_orientation_y, mouth_orientation_z, mouth_orientation_w = mouth_to_table_transform.M.GetQuaternion()

            if (camera_pose != None):
                # transform so we have physical camera in table frame
                camera_pykdl_frame = pm.fromMsg(camera_pose)
                camera_to_table_transform = (table_pykdl_frame * camera_pykdl_frame)
                # create a transform
                vector = PyKDL.Vector(0.00922705, 0.00726385, 0.01964674)
                rotation = PyKDL.Rotation.Quaternion(-0.10422086, 0.08057444, -0.69519346, 0.7061333)
                camera_realsense_pykdl_frame = PyKDL.Frame(rotation,vector)
                # transform so we have realsense in table frame
                realsense_to_table_transform = (camera_to_table_transform * camera_realsense_pykdl_frame)
                # extract in the form of pose and quaternion
                camera_position_x, camera_position_y, camera_position_z = realsense_to_table_transform.p
                camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w = realsense_to_table_transform.M.GetQuaternion()


        # TODO: maybe change name of bag file to just the participant, trial, and food for ease of reading
        # ["file_name","bag_start_time","calculated_start_time","end_time","time_lost_tracking", "total_duration",
        #    "mouth_position_x", "mouth_position_y", "mouth_position_z", "mouth_orientation_x", "mouth_orientation_y",
        #   "mouth_orientation_z", "mouth_orientation_w"]
        if (best_t == None): best_t = rospy.Time()   
        if (current == None): current = rospy.Time()   
  
        filewriter.writerow([bagName,bag_start_time,best_t.to_sec(),current.to_sec(),time_lost,current.to_sec()-bag_start_time,mouth_position_x, mouth_position_y, mouth_position_z, mouth_orientation_x, mouth_orientation_y,
           mouth_orientation_z, mouth_orientation_w, camera_position_x, camera_position_y, camera_position_z, 
           camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w])                
        bag.close()
print("Done reading all " + numberOfFiles + " bag files.")