import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import pprint
import rosbag
import tf.transformations
import tf2_py

# CONSTANTS
IN_DIR = "/home/amaln/workspaces/amal_ws/data/2021_extended_action_space_dataset/raw/"
OUT_DIR = "/home/amaln/workspaces/amal_ws/data/2021_extended_action_space_dataset/processed/"

def parse_rosbag_name(filename):
	"""
	Return the participant number, food name, and trial number 
	for a given rosbag name.

	The naming convention is:

	{participant_num}-{food_name}-{trial_num}_{timestamp}.bag
	"""
	first_hyphen = filename.find("-")
	if first_hyphen == -1:
		raise Exception("Filename %s is malformed" % filename)
	participant_num = int(filename[:first_hyphen])
	second_hyphen = filename.find("-", first_hyphen+1)
	if second_hyphen == -1:
		raise Exception("Filename %s is malformed" % filename)
	food_name = filename[first_hyphen+1:second_hyphen]
	first_underscore = filename.find("_", second_hyphen+1)
	if first_underscore == -1:
		raise Exception("Filename %s is malformed" % filename)
	trial_num = int(filename[second_hyphen+1:first_underscore])

	return participant_num, food_name, trial_num

def get_initial_static_transforms(start_time, fork_tip_frame_id):
    """
    Return the static transforms from ForqueBody to fork_tip and from CameraBody to
    camera_link
    """
    start_time = rospy.Time(start_time)

    fork_transform = TransformStamped()
    fork_transform.header.stamp = start_time
    fork_transform.header.frame_id = "ForqueBody"
    fork_transform.child_frame_id = fork_tip_frame_id
    fork_transform.transform.translation.x = 0.13347163
    fork_transform.transform.translation.y = -0.02011249
    fork_transform.transform.translation.z =  0.1598728
    fork_transform.transform.rotation.x = -0.13918934
    fork_transform.transform.rotation.y = 0.2998962
    fork_transform.transform.rotation.z = -0.50758908
    fork_transform.transform.rotation.w = 0.79561947

    camera_transform = TransformStamped()
    camera_transform.header.stamp = start_time
    camera_transform.header.frame_id = "CameraBody"
    camera_transform.child_frame_id = "camera_link"
    camera_transform.transform.translation.x = 0.00922705
    camera_transform.transform.translation.y = 0.00726385
    camera_transform.transform.translation.z =  0.01964674
    camera_transform.transform.rotation.x = -0.10422086
    camera_transform.transform.rotation.y = 0.08057444
    camera_transform.transform.rotation.z = -0.69519346
    camera_transform.transform.rotation.w = 0.7061333

    return [fork_transform, camera_transform]


def create_directory_structure(participant_num, food_name, trial_num):
	"""
	Creates the necessary folders for this trial if they don't exist. The 
	convention is:

	{OUT_DIR}/subject{participant_num}_{food_name}/{trial_num}
	"""
	participant_food_dir = os.path.join(OUT_DIR, "subject%d_%s" % (participant_num, food_name))
	if not os.path.exists(participant_food_dir):
		os.makedirs(participant_food_dir)

	trial_dir = os.path.join(participant_food_dir, "%d" % trial_num)
	if not os.path.exists(trial_dir):
		os.makedirs(trial_dir)
	

def process_rosbag(filepath, participant_num, food_name, trial_num, debug=False):
	"""
	Reads in the rosbag and does the following:
	(1) Saves every RGB image to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}/{timestamp}_rgb.jpg
	(2) Saves every depth image to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}/{timestamp}_depth.png
	(3) Saves the camera location and the mouth location in the world frame to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}_static_transforms.csv
	(4) Saves every F/T reading and every fork pose reading to {OUT_DIR}/{participant_num}_{food_name}/{trial_num}.csv
		Note that this requires a few steps of pre-processing:
		(a) Transform the center of the rigid body (recorded by OptiTrak)
			to the forque frame (the frame of the F/T sensor).
		(b) Rotate the F/T sensor readings to align with the convention that, when looking 
			from the fork handle to the forktines, +z is towards the forktines, +y is left, 
			and +x is away from curvature of the forktines.
					
	"""

	base_dir = os.path.join(OUT_DIR, "subject%d_%s/" % (participant_num, food_name))

	# Open a tf buffer, and populate it with static transforms
    tf_buffer = tf2_py.BufferCore(rospy.Duration((end_time-start_time)*2.0)) # times 2 is to give it enough space in case some messages were backdated
    initial_static_transforms = get_initial_static_transforms(start_time, fork_tip_frame_id)
    for transform in initial_static_transforms:
        tf_buffer.set_transform_static(transform, "default_authority")

	# Get the topics
	camera_image_topic = "/camera/color/image_raw/compressed"
	camera_depth_topic = "/camera/aligned_depth_to_color/image_raw"
	tf_static_topic = "/tf_static"
	forque_topic = "/forque/forqueSensor"
    forque_body_topic = "/vrpn_client_node/ForqueBody/pose"
	tf_topic = "tf"
	topics = [
		camera_image_topic,
		camera_depth_topic,
		tf_static_topic,
		tf_topic,
		forque_topic,
		forque_body_topic,
	]
	desired_parent_frame="TableBody",

	# Open the bag
	bag = rosbag.Bag(filepath)

	# Iterate over the msgs in the rosbag
	cv_bridge = CvBridge()
	for topic, msg, timestamp in bag.read_messages(topics=topics):
		if topic == camera_image_topic:
			img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			ts = msg.header.stamp.to_nsec()
			cv2.imwrite(os.path.join(base_dir, "%d/%d_rgb.jpg" % (trial_num, ts)), img)
		elif topic == camera_depth_topic:
			img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
			ts = msg.header.stamp.to_nsec()
			cv2.imwrite(os.path.join(base_dir, "%d/%d_depth.png" % (trial_num, ts)), img)
			if debug:
				# Verify that the cv2 doesn't change the units from mm when it saves the depth image
				assert(np.isclose(img, cv2.imread(os.path.join(base_dir, "%d/%d_depth.png" % (trial_num, ts)), cv2.IMREAD_ANYDEPTH)).all())
		 elif topic == tf_static_topic:
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, "default_authority")
        elif topic == tf_topic:
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "default_authority")


if __name__ == "__main__":

	# Find all the rosbags in IN_DIR
	rosbag_paths = {} # participant_num -> food_name -> trial_num -> filepath
	for root, subfolders, files in os.walk(IN_DIR):
		for filename in files:
			if filename.lower().endswith(".bag") and "pilot" not in root:
				participant_num, food_name, trial_num = parse_rosbag_name(filename)
				if participant_num not in rosbag_paths: 
					rosbag_paths[participant_num] = {}
				if food_name not in rosbag_paths[participant_num]: 
					rosbag_paths[participant_num][food_name] = {}
				if trial_num in rosbag_paths[participant_num][food_name]: 
					raise Exception("Trial %d occurs multiple times: %s" % (trial_num, os.path.join(root, filename)))
				rosbag_paths[participant_num][food_name][trial_num] = os.path.join(root, filename)
	pprint.pprint(rosbag_paths)

	# Process each rosbag
	for participant_num in rosbag_paths:
		for food_name in rosbag_paths[participant_num]:
			for trial_num in rosbag_paths[participant_num][food_name]:
				filepath = rosbag_paths[participant_num][food_name][trial_num]

				create_directory_structure(participant_num, food_name, trial_num)

				process_rosbag(filepath, participant_num, food_name, trial_num)

	pass