# Standard Imports
import argparse
from collections import namedtuple
import copy
import csv
import os
import pprint
import random
import time
import traceback

# Third-Party Imports
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.ndimage import median_filter
import transformations

# Create named tuples to contain the data
TrialDataPaths = namedtuple('TrialDataPaths', ['img_dir', 'static_transforms', 'wrenches_poses'])
ActionSchemaPoint = namedtuple('ActionSchemaPoint', [
    "Trial_Start_Time",
    "Action_Start_Time",
    "Action_Contact_Time",
    "Action_Extraction_Time",
    "Action_End_Time",
    "Trial_Duration",
    "Food_Reference_Frame_Translation_X",
    "Food_Reference_Frame_Translation_Y",
    "Food_Reference_Frame_Translation_Z",
    "Food_Reference_Frame_Rotation_X",
    "Food_Reference_Frame_Rotation_Y",
    "Food_Reference_Frame_Rotation_Z",
    "Pre_Grasp_Target_Offset_X",
    "Pre_Grasp_Target_Offset_Y",
    "Pre_Grasp_Target_Offset_Z",
    "Pre_Grasp_Initial_Utensil_Transform_Translation_X",
    "Pre_Grasp_Initial_Utensil_Transform_Translation_Y",
    "Pre_Grasp_Initial_Utensil_Transform_Translation_Z",
    "Pre_Grasp_Initial_Utensil_Transform_Rotation_X",
    "Pre_Grasp_Initial_Utensil_Transform_Rotation_Y",
    "Pre_Grasp_Initial_Utensil_Transform_Rotation_Z",
    "Pre_Grasp_Force_Threshold",
    "Approach_Frame_Rotation_X",
    "Approach_Frame_Rotation_Y",
    "Approach_Frame_Rotation_Z",
    "Grasp_In_Food_Twist_Linear_X",
    "Grasp_In_Food_Twist_Linear_Y",
    "Grasp_In_Food_Twist_Linear_Z",
    "Grasp_In_Food_Twist_Angular_X",
    "Grasp_In_Food_Twist_Angular_Y",
    "Grasp_In_Food_Twist_Angular_Z",
    "Grasp_Force_Threshold",
    "Grasp_Torque_Threshold",
    "Grasp_Duration",
    "Extraction_Out_Of_Food_Twist_Linear_X",
    "Extraction_Out_Of_Food_Twist_Linear_Y",
    "Extraction_Out_Of_Food_Twist_Linear_Z",
    "Extraction_Out_Of_Food_Twist_Angular_X",
    "Extraction_Out_Of_Food_Twist_Angular_Y",
    "Extraction_Out_Of_Food_Twist_Angular_Z",
    "Extraction_Duration",
])
WrenchStamped = namedtuple('WrenchStamped', ['time', 'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z'])
TransformStamped = namedtuple('TransformStamped', ['parent_frame', 'child_frame', 'time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
PoseStamped = namedtuple('PoseStamped', ['frame_id', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
TwistStamped = namedtuple('TwistStamped', ['frame_id', 'time', 'linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
Vector3 = namedtuple('Vector3', ['x', 'y', 'z'])
CameraIntrinsics = namedtuple('CameraIntrinsics', ['width', 'height', 'fx', 'fy', 'cx', 'cy'])

# NOTE: IF you change these, you'll have to change the order in which R/P/Y is
# inputted into the functions below!
EULER_ORDER_INPUT = 'rzyx'
EULER_ORDER = 'rxyz'

def transform_to_matrix(transform):
    m = transformations.euler_matrix(transform.roll, transform.pitch, transform.yaw, EULER_ORDER)
    m[0][3] = transform.x
    m[1][3] = transform.y
    m[2][3] = transform.z
    return m

def matrix_to_transform(m, parent_frame=None, child_frame=None, time=None):
    roll, pitch, yaw = transformations.euler_from_matrix(m, EULER_ORDER)
    transform = TransformStamped(
        parent_frame=parent_frame,
        child_frame=child_frame,
        time=time,
        x=m[0][3],
        y=m[1][3],
        z=m[2][3],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )
    return transform

def pose_to_matrix(pose):
    return transform_to_matrix(pose)

def matrix_to_pose(m, frame_id=None):
    roll, pitch, yaw = transformations.euler_from_matrix(m, EULER_ORDER)
    pose = PoseStamped(
        frame_id=frame_id,
        x=m[0][3],
        y=m[1][3],
        z=m[2][3],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )
    return pose

def matrix_to_twist(m, duration, frame_id=None, time=None):
    rx, ry, rz = transformations.euler_from_matrix(m, EULER_ORDER)
    twist_stamped = TwistStamped(
        frame_id=frame_id,
        time=time,
        linear_x=m[0][3]/duration,
        linear_y=m[1][3]/duration,
        linear_z=m[2][3]/duration,
        angular_x=rx/duration,
        angular_y=ry/duration,
        angular_z=rz/duration,
    )

    return twist_stamped

def convert_input_euler(roll, pitch, yaw):
    m = transformations.euler_matrix(yaw, pitch, roll, EULER_ORDER_INPUT)
    roll, pitch, yaw = transformations.euler_from_matrix(m, EULER_ORDER)
    return roll, pitch, yaw

def fix_euler_continuity(roll, pitch, yaw, prev_roll=None, prev_pitch=None, prev_yaw=None):
    if prev_roll is None or prev_pitch is None or prev_yaw is None:
        return roll, pitch, yaw
    threshold = 0.75
    if (prev_roll > 0) and (roll < 0) and ((prev_roll - roll) > 2*np.pi*threshold):
        roll += np.pi*2
    elif (prev_roll < 0) and (roll > 0) and ((roll - prev_roll) > 2*np.pi*threshold):
        roll -= np.pi*2
    if (prev_yaw > 0) and (yaw < 0) and ((prev_yaw - yaw) > 2*np.pi*threshold):
        yaw += np.pi*2
    elif (prev_yaw < 0) and (yaw > 0) and ((yaw - prev_yaw) > 2*np.pi*threshold):
        yaw -= np.pi*2
    return roll, pitch, yaw


def get_trials(raw_data_path):
    """
    Returns a nested dictionary of all trials in the raw data path. Specifically,
    the dictionary is of the form:
    subject_num -> food_name -> trial_num -> trial_data_paths

    Specifically, it expects the following structure:
    > your_raw_data_folder/
    >     subject{subject_num}_{food_name}/
    >         {trial_num}/
    >             {timestamp}_depth.png
    >             {timestamp}_rgb.jpg
    >         {trial_num}_static_transforms.csv
    >         {trial_num}_wrenches_poses.csv

    Parameters
    ----------
    raw_data_path : str
        The path to the folder containing the raw data

    Returns
    -------
    trials : dict
        A nested dictionary of all trials in the raw data path
    total_trials : int
        The total number of trials in the raw data path
    """
    # Get all the folders in the raw data path
    folders = os.listdir(raw_data_path)

    # Initialize the trials dictionary
    trials = {}
    total_trials = 0

    # Iterate through the folders
    for folder in folders:
        # Get the subject number and food name
        if "subject" not in folder or "_" not in folder:
            continue
        subject_num_superstring, food_name = folder.split("_")
        subject_num = int(subject_num_superstring.replace("subject", ""))

        # Initialize the trials dictionary
        if subject_num not in trials:
            trials[subject_num] = {}
        if food_name not in trials[subject_num]:
            trials[subject_num][food_name] = {}

        # Get the path to the subject folder
        subject_path = os.path.join(raw_data_path, folder)

        # Get the trial folders
        trial_folders = os.listdir(subject_path)

        # Iterate through the trial folders
        for trial_folder in trial_folders:
            if not os.path.isdir(os.path.join(subject_path, trial_folder)):
                continue
            try:
                # Convert the trial number to an integer
                trial_num = int(trial_folder)
            except ValueError:
                # If the trial number is not an integer, skip it
                continue

            # Check that the CSVs exist
            static_transforms_path = os.path.join(subject_path, f"{trial_num}_static_transforms.csv")
            wrenches_poses_path = os.path.join(subject_path, f"{trial_num}_wrenches_poses.csv")
            if not os.path.exists(static_transforms_path) or not os.path.exists(wrenches_poses_path):
                continue

            # Add the trial to the trials dictionary
            trials[subject_num][food_name][trial_num] = TrialDataPaths(
                img_dir=os.path.join(subject_path, trial_folder),
                static_transforms=static_transforms_path,
                wrenches_poses=wrenches_poses_path
            )
            total_trials += 1

    return trials, total_trials

def get_camera_intrinsics(camera_intrinsics_path):
    """
    Returns the camera intrinsics from the camera intrinsics CSV file.

    Parameters
    ----------
    camera_intrinsics_path : str
        The path to the camera intrinsics CSV file

    Returns
    -------
    camera_intrinsics : CameraIntrinsics
        The camera intrinsics
    """
    # Open the CSV file
    with open(camera_intrinsics_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Iterate through the rows
        for row in csvreader:
            # Skip the header
            if row[0] == "Frame ID":
                continue

            # Get the data
            width = int(row[1])
            height = int(row[2])
            fx = float(row[4])
            fy = float(row[5])
            cx = float(row[6])
            cy = float(row[7])

            return CameraIntrinsics(width, height, fx, fy, cx, cy)

def create_csv_writer(filepath="data/action_schema_data.csv"):
    """
    Creates a CSV writer for the action schema data.

    Parameters
    ----------
    filepath : str
        The path to the CSV file to write to. This is relative to this script's
        parent folder.

    Returns
    -------
    csv_writer : csv.writer
        The CSV writer
    """
    # Create the CSV header
    csv_header = [
        "Save Timestamp",
        "Participant",
        "Food",
        "Trial",
        # "Bag File Name",
    ] + list(ActionSchemaPoint._fields)

    # Open the CSV writer
    csvfile_path = os.path.join(os.path.dirname(__file__), "..", filepath)
    os.makedirs(os.path.dirname(csvfile_path), exist_ok=True)
    csvfile = open(csvfile_path, "w")
    csvwriter = csv.writer(csvfile)

    # Write the header
    csvwriter.writerow(csv_header)

    return csvwriter, csvfile

def write_action_schema_point_to_csv(csvwriter, action_schema_point, subject_num, food_name, trial_num):
    """
    Writes an action schema point to the CSV file.

    Parameters
    ----------
    csvwriter : csv.writer
        The CSV writer
    action_schema_point : ActionSchemaPoint
        The action schema point
    subject_num : int
        The subject number
    food_name : str
        The food name
    trial_num : int
        The trial number

    Returns
    -------
    None
    """
    csvwriter.writerow([
        time.time(),
        subject_num,
        food_name,
        trial_num,
    ] + [
        action_schema_point.__getattribute__(field)
        for field in ActionSchemaPoint._fields
    ])

def read_wrenches_poses(wrenches_poses_path, parent_frame, forktip_frame):
    """
    Read the wrenches and poses from the wrenches_poses CSV file.

    Parameters
    ----------
    wrenches_poses_path : str
        The path to the wrenches_poses CSV file

    Returns
    -------
    force_data : list of WrenchStamped
        The force data
    forque_transform_data : list of TransformStamped
        The forque transform data
    """
    # Initialize the force and forque transform data
    force_data = []
    forque_transform_data = []

    # Open the CSV file
    with open(wrenches_poses_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Iterate through the rows
        for row in csvreader:
            # Skip the header
            if row[0] == "Time (sec)":
                continue

            # Get the data
            time = float(row[0])
            if len(row[1]) > 0:
                force_x = float(row[1])
                force_y = float(row[2])
                force_z = float(row[3])
                torque_x = float(row[4])
                torque_y = float(row[5])
                torque_z = float(row[6])
                force_data.append(WrenchStamped(time, force_x, force_y, force_z, torque_x, torque_y, torque_z))
            elif len(row[7]) > 0:
                x = float(row[7])
                y = float(row[8])
                z = float(row[9])
                roll = float(row[10])
                pitch = float(row[11])
                yaw = float(row[12])
                roll, pitch, yaw = convert_input_euler(roll, pitch, yaw)

                # The dataset defines the forktip frame with +x going down and +y going to the left.
                # However, our robot URDF defines the forktip frame with +x going to the left and +y
                # going up. Hence, we rotate the forktip frame by +90 degrees about the z-axis.
                m0 = transformations.euler_matrix(roll, pitch, yaw, EULER_ORDER)
                m1 = transformations.euler_matrix(0, 0, np.pi/2, EULER_ORDER)
                roll, pitch, yaw = transformations.euler_from_matrix(transformations.concatenate_matrices(m0, m1), EULER_ORDER)
                
                forque_transform_data.append(TransformStamped(parent_frame, forktip_frame, time, x, y, z, roll, pitch, yaw))
            else:
                raise Exception(f"Invalid row in {wrenches_poses_path}: {row}")

    return force_data, forque_transform_data

def read_static_transforms(static_transforms_path):
    """
    Read the static transforms from the static_transforms CSV file.

    Parameters
    ----------
    static_transforms_path : str
        The path to the static_transforms CSV file
    
    Returns
    -------
    transforms : list of TransformStamped
        The static transforms
    """
    # Initialize the transforms
    transforms = []

    # Open the CSV file
    with open(static_transforms_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Iterate through the rows
        for row in csvreader:
            # Skip the header
            if row[0] == "Parent Frame":
                continue

            # Get the data
            parent_frame = row[0]
            child_frame = row[1]
            x = float(row[2])
            y = float(row[3])
            z = float(row[4])
            roll = float(row[5])
            pitch = float(row[6])
            yaw = float(row[7])
            roll, pitch, yaw = convert_input_euler(roll, pitch, yaw)
            transforms.append(TransformStamped(parent_frame, child_frame, None, x, y, z, roll, pitch, yaw))

    return transforms

def get_forque_distance_to_mouth(forque_transform_data_raw, parent_to_mouth_transform):
    """
    Get the forque's distance to the mouth.

    Parameters
    ----------
    forque_transform_data_raw : list of TransformStamped
        The raw forque transform data (e.g., parent frame to forktip frame)
    parent_to_mouth_transform : TransformStamped
        The transform from the forque's parent frame to the mouth

    Returns
    -------
    forque_distance_to_mouth : list of float
        The forque's distance to the mouth
    """
    forque_distance_to_mouth = []

    # Get the forque's distance to the mouth
    mouth_to_parent_matrix = transformations.inverse_matrix(transform_to_matrix(parent_to_mouth_transform))
    for parent_to_forktip_transform in forque_transform_data_raw:
        mouth_to_forktip_transform = transformations.concatenate_matrices(
            mouth_to_parent_matrix,
            transform_to_matrix(parent_to_forktip_transform),
        )
        distance = np.linalg.norm(mouth_to_forktip_transform[0:3, 3])
        forque_distance_to_mouth.append(distance)

    return forque_distance_to_mouth

def median_filter_hz(data, expected_hz=50.0, smoothening_window_hz=3.0):
    """
    Run a median filter on data to smoothen it. The data is expected to come in
    at a rate of `expected_hz` Hz, and the median filter will be run on windows
    with `expected_hz/smoothening_window_hz` num datapoints.

    Parameters
    ----------
    data : list
        The data to smoothen
    expected_hz : float
        The expected frequency of the data
    smoothening_window_hz : float
        The size of the smoothening window in Hz

    Returns
    -------
    filtered_data : list
        The smoothened data
    """
    return median_filter(np.array(data), size=int(expected_hz/smoothening_window_hz))

def smoothen_transforms(transform_data_raw, expected_hz=50.0, smoothening_window_hz=3.0):
    """
    Applies a median filter to the transforms to smoothen them. Specifically, it takes
    windows of size expected_hz/smoothening_window_hz and applies a median filter to that
    window.

    Parameters
    ----------
    transform_data_raw : list of TransformStamped
        The raw transform data
    expected_hz : float
        The expected frequency of the transforms
    smoothening_window_hz : float
        The size of the smoothening window in Hz
    
    Returns
    -------
    smoothened_transform_data : list of TransformStamped
        The smoothened transform data
    """
    smoothened_transform_data = []

    # Get actual data
    xs, ys, zs, rolls, pitches, yaws = [], [], [], [], [], []
    prev_euler = [None, None, None]
    for transform_stamped in transform_data_raw:
        xs.append(transform_stamped.x)
        ys.append(transform_stamped.y)
        zs.append(transform_stamped.z)
        roll, pitch, yaw = fix_euler_continuity(transform_stamped.roll, transform_stamped.pitch, transform_stamped.yaw, *prev_euler)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
        prev_euler = [roll, pitch, yaw]

    # Filter the data
    xs_filtered = median_filter_hz(xs, expected_hz=expected_hz, smoothening_window_hz=smoothening_window_hz)
    ys_filtered = median_filter_hz(ys, expected_hz=expected_hz, smoothening_window_hz=smoothening_window_hz)
    zs_filtered = median_filter_hz(zs, expected_hz=expected_hz, smoothening_window_hz=smoothening_window_hz)
    rolls_filtered = median_filter_hz(rolls, expected_hz=expected_hz, smoothening_window_hz=smoothening_window_hz)
    pitches_filtered = median_filter_hz(pitches, expected_hz=expected_hz, smoothening_window_hz=smoothening_window_hz)
    yaws_filtered = median_filter_hz(yaws, expected_hz=expected_hz, smoothening_window_hz=smoothening_window_hz)

    # Create the smoothened data
    for i in range(len(transform_data_raw)):
        transform_stamped = transform_data_raw[i]
        smoothened_transform_data.append(TransformStamped(
            transform_stamped.parent_frame,
            transform_stamped.child_frame,
            transform_stamped.time,
            xs_filtered[i],
            ys_filtered[i],
            zs_filtered[i],
            rolls_filtered[i],
            pitches_filtered[i],
            yaws_filtered[i],
        ))

    return smoothened_transform_data

def get_blue_range(colors, blue_ratio, min_blue):
    """
    Given a list of RGB colors, gives the range of colors that are "blues",
    where "blue" is defined as a color where the B element is more than blue_ratio
    times the R and G components

    Parameters
    ----------
    colors : list of (int, int, int)
        The colors
    blue_ratio : float
        The ratio of B to R and G that defines a blue color
    min_blue : int
        The minimum B value that defines a blue color

    Returns
    -------
    blue_lo : np.array
        The lower bound of the blue colors
    blue_hi : np.array
        The upper bound of the blue colors
    """
    blue_lo = [255, 255, 255]
    blue_hi = [0, 0, 0]
    for r, g, b in colors:
        if b > min_blue and b > blue_ratio*r and b > blue_ratio*g:
            if r > 0 and r < blue_lo[0]:
                blue_lo[0] = r-1
            if g > 0 and g < blue_lo[1]:
                blue_lo[1] = g-1
            if b > 0 and b < blue_lo[2]:
                blue_lo[2] = b-1
            if r < 255 and r > blue_hi[0]:
                blue_hi[0] = r+1
            if g < 255 and g > blue_hi[1]:
                blue_hi[1] = g+1
            if b < 255 and b > blue_hi[2]:
                blue_hi[2] = b+1
    blue_lo = np.array(blue_lo)
    blue_hi = np.array(blue_hi)
    return blue_lo, blue_hi

def remove_background(image,
    x_min=160, y_min=60,
    black_lo=np.array([0, 0, 0]), black_hi=np.array([100, 100, 100]),
    glare_lo=np.array([2, 110, 185]), glare_hi=np.array([255, 255, 255]),
    k=3,
    blue_ratio=1.75, min_blue=85,
    min_blue_proportion=0.5,
    image_name=None, save_process=False, out_dir=None):
    """
    The function has several steps, that were fine-tuned for the type of images
    captured in the Nov 21 Bite Acquisition Study.
        1) Crop the image at (x_min, y_min) onwards. This removes the forque and
           the floor.
        2) Inpaint black portions of the image to remove the cord.
        3) Inpaint white / light-blue portions of the image to remove glare on
           the plate.
        4) Run k-means with 3 centers to simplify the colors in the image.
        5) Mask out the blue colors in the simplified image (where "blue colors"
           are any colors where the B value is more than blue_ratio times the R
           and G values and greater than min_blue).
        6) Get the largest convex hull of the mask that contains > min_blue_proportion blue.
        7) Color everything *outside* that convex hull in the original image blue.
        8) Return that image.

    Note that the default parameters were selected by manually analyzing several
    of the images.
    """

    # Crop the image
    cropped_image = image[y_min:, x_min:]

    # Remove the black cord
    image_no_cord = cv2.inpaint(cropped_image, cv2.inRange(cropped_image, black_lo, black_hi), 3, cv2.INPAINT_TELEA)

    # Remove the glare
    image_no_glare = cv2.inpaint(image_no_cord, cv2.inRange(image_no_cord, glare_lo, glare_hi), 3, cv2.INPAINT_TELEA)

    # Run k-means to simplify the colors in the image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(np.float32(image_no_glare.reshape((-1, 3))), k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    simplified_image = res.reshape((image_no_glare.shape))

    # Get blue_lo and blue_hi
    blue_lo, blue_hi = get_blue_range(center, blue_ratio, min_blue)

    # Mask out the blue in the image
    mask = cv2.inRange(simplified_image, blue_lo, blue_hi)
    mask = 255-mask

    # Find the largest convex hull that contains mostly blue
    contours = cv2.findContours(255-mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else countous[1]
    best_hull, best_hull_size, best_hull_proportion = None, None, None
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])

        # Generate the Hull Mask
        hull_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(hull_mask, [hull], 0, (255), 1)

        # Get the proportion of pixels that are blue within the hull
        hull_size = np.count_nonzero(hull_mask == 255)
        hull_blue_size = np.count_nonzero(np.logical_and(mask == 255, hull_mask == 255))
        proportion = float(hull_blue_size) / hull_size

        # Store the largest one that is mostly blue
        if (proportion < min_blue_proportion and (best_hull_proportion is None or proportion > best_hull_proportion)) or proportion >= min_blue_proportion:
            if best_hull_size is None or hull_size > best_hull_size:
                best_hull_size = hull_size
                best_hull = hull
                best_hull_proportion = proportion

    # Save the process if specified
    if save_process and (image_name is None or out_dir is not None):
        result = cropped_image.copy()
        color = (255, 0, 0)
        cv2.drawContours(result, [best_hull], 0, color, thickness=3)
        plt.imshow(np.concatenate((
            cropped_image,
            image_no_cord,
            image_no_glare,
            simplified_image,
            np.repeat(mask.reshape((mask.shape[0], mask.shape[1], 1)), 3, axis=2),
            result,
        ), axis=1))
        if image_name is None:
            plt.show()
        else:
            if out_dir is not None:
                plt.savefig(os.path.join(out_dir, image_name+"_remove_background_process.png"))
                plt.clf()
        del result

    # Make all the points outside of the hull blue
    best_hull[:, :, 0] += x_min
    best_hull[:, :, 1] += y_min
    best_hull_mask = np.zeros(image.shape[0:2], np.uint8)
    cv2.fillPoly(best_hull_mask, pts=[best_hull], color=255)
    retval = image.copy()
    retval[best_hull_mask == 0] = (blue_lo + blue_hi) / 2

    # Finally, slightly blur the hull border to make it smoother
    best_hull_outline_mask = np.zeros(image.shape[0:2], np.uint8)
    cv2.drawContours(best_hull_outline_mask, [best_hull], 0, 255, thickness=20)
    retval_blurred = cv2.inpaint(retval, best_hull_outline_mask, 20, cv2.INPAINT_TELEA)

    return retval_blurred

def get_food_bounding_box(image,
    glare_lo=np.array([2, 110, 185]),
    glare_hi=np.array([255, 255, 255]),
    k=2,
    blue_ratio=1.75, min_blue=85,
    shrink_border_by=5,
    min_area=500, max_area=50000, #max_area=35000,#
    image_name=None, save_process=False, out_dir=None):
    """
    The function has several steps, that were fine-tuned for the type of images
    captured in the Nov 21 Bite Acquisition Study.
        1) Inpaint white / light-blue portions of the image to remove glare on
           the plate.
        4) Run k-means with 2-3 centers (depending on food item) to simplify
           the colors in the image.
        5) Mask out the blue colors in the simplified image (where "blue colors"
           are any colors where the B value is more than blue_ratio times the R
           and G values and greater than min_blue).
        6) Narrow the mask slightly to separate adjacent food items.
        7) Get the contours, fit rotated rectangles to them.
        8) Treat every rectangle with area between min_area and max_area as a
           food item.

    Note that the default parameters were selected by manually analyzing several
    of the images.
    """

    # Remove the glare
    image_no_glare = cv2.inpaint(image, cv2.inRange(image, glare_lo, glare_hi), 3, cv2.INPAINT_TELEA)

    # Run K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(np.float32(image_no_glare.reshape((-1, 3))), k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    # print(center)
    res = center[label.flatten()]
    simplified_image = res.reshape((image_no_glare.shape))

    # Get blue_lo and blue_hi
    blue_lo, blue_hi = get_blue_range(center, blue_ratio, min_blue)

    # Mask out the blue in the image
    mask = cv2.inRange(simplified_image, blue_lo, blue_hi)
    mask = 255-mask

    # If two discrete food items are slightly touching, we want to treat them
    # differently. Hence, we get the edges and set them to the background color
    image_only_food = image_no_glare.copy()
    image_only_food[mask == 0] = (255, 255, 255)
    edges = cv2.Canny(cv2.cvtColor(image_only_food, cv2.COLOR_RGB2GRAY), 100, 300)
    edges = cv2.dilate(edges, np.ones((2, 2),np.uint8), iterations=1)
    mask[edges != 0] = 0

    # Get the countours of this mask
    contours = cv2.findContours(255-mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else countous[1]
    # Get the best fit rectangle and ellipse
    # minRect = {}
    minEllipse = {}
    for i, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        center, size, angle = rect
        area = size[0]*size[1]
        # print(area)
        if area >= min_area and area <= max_area:
            # minRect[i] = rect
            if c.shape[0] > 5:
                ellipse = cv2.fitEllipse(c)
                # Only add the ellipse if its center is in-bounds
                if ellipse[0][0] >= 0 and ellipse[0][0] < image.shape[1] and ellipse[0][1] >= 0 and ellipse[0][1] < image.shape[0]:
                    minEllipse[i] = ellipse

    if save_process and (image_name is None or out_dir is not None):
        result = image.copy()
        for i in minEllipse:
            color = (255, 0, 0)
            cv2.drawContours(result, contours, i, color)
            # ellipse
            if c.shape[0] > 5:
                cv2.ellipse(result, minEllipse[i], color, 2)
        plt.imshow(np.concatenate((
            image,
            image_no_glare,
            simplified_image,
            np.repeat(mask.reshape((mask.shape[0], mask.shape[1], 1)), 3, axis=2),
            np.repeat(edges.reshape((edges.shape[0], edges.shape[1], 1)), 3, axis=2),
            result,
        ), axis=1))
        if image_name is None:
            plt.show()
        else:
            if out_dir is not None:
                plt.savefig(os.path.join(out_dir, image_name+"_bounding_box_process.png"))
                plt.clf()
        del result

    del image_only_food

    return list(minEllipse.values())

def fit_table(gray_raw, depth_raw, y_min=60,
    _hough_accum=1.5, _hough_min_dist=100, _hough_param1=100, _hough_param2=70,
    _hough_min=103, _hough_max=140, _table_buffer=50, image_name=None,
    save_process=False, out_dir=None):
    """
    Find table plane.
    gray(image matrix): grayscale image of plate
    depth(image matrix): depth image of plate

    Returns:
    ----------
    plate_uv: (u, v) of plate center
    plate_r: radius of plate in px
    table: depth image of just the table
    height: new depth image, filling in pixels below the table with the table
    """
    gray = gray_raw[y_min:, :]
    depth = depth_raw[y_min:, :]

    # Detect Largest Circle (Plate)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, _hough_accum, _hough_min_dist,
        param1=_hough_param1, param2=_hough_param2, minRadius=_hough_min, maxRadius=_hough_max)
    if circles is None:
        return None, None, None
    circles = np.round(circles[0, :]).astype("int")
    plate_uv = (0, 0)
    plate_r = 0
    for (x,y,r) in circles:
        # print("Radius: " + str(r))
        if r > plate_r:
            plate_uv = (x, y)
            plate_r = r

    # Create Mask for Depth Image
    plate_mask = np.zeros(depth.shape)
    cv2.circle(plate_mask, plate_uv, plate_r + _table_buffer, 1.0, -1)
    cv2.circle(plate_mask, plate_uv, plate_r, 0.0, -1)
    depth_mask = (depth * (plate_mask).astype("uint16")).astype(float)

    # Noise removal
    kernel = np.ones((6,6), np.uint8)
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel, iterations = 3)
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations = 3)

    # Remove Outliers
    depth_var = np.abs(depth_mask - np.mean(depth_mask[depth_mask > 0]))
    depth_std = np.std(depth_mask[depth_mask > 0])
    depth_mask[depth_var > 2.0*depth_std] = 0

    # Fit Plane: depth = a*u + b*v + c
    d_idx = np.where(depth_mask > 0)
    d = depth_mask[d_idx].astype(float)
    coeffs = np.hstack((np.vstack(d_idx).T, np.ones((len(d_idx[0]), 1)))).astype(float)
    b, a, c = np.linalg.lstsq(coeffs, d)[0]

    # Create Table Depth Image
    u = np.linspace(0, depth_raw.shape[1], depth_raw.shape[1], False)
    v = np.linspace(0, depth_raw.shape[0], depth_raw.shape[0], False)
    U, V = np.meshgrid(u, v)
    table = a*U + b*V + c
    table = table.astype("uint16")

    # New Height
    height = np.clip(depth_raw, None, table)
    height[np.where(depth_raw == 0)] = 0

    # Noise removal
    kernel = np.ones((2,2), np.uint8)
    height = cv2.morphologyEx(height, cv2.MORPH_OPEN, kernel, iterations = 2)
    height = cv2.morphologyEx(height, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Debugging
    if save_process and (image_name is None or out_dir is not None):
        plt.imshow(np.concatenate((
            gray.astype('float32')/255.0*depth.max(),
            plate_mask*depth.max(),
            depth,
            depth_mask.astype("uint16"),
            table[y_min:, :],
            height[y_min:, :],
        ), axis=1), cmap='gray')
        if image_name is None:
            plt.show()
        else:
            if out_dir is not None:
                plt.savefig(os.path.join(out_dir, image_name+"_table_height_process.png"))
                plt.clf()

    return (plate_uv[0], plate_uv[1]+y_min), plate_r, table, height

def get_food_origin_frames(bounding_ellipses, table, parent_to_camera_transform,
    depth_image, camera_intrinsics):
    """
    Converts detected bounding ellipses of food items in (u,v) space to a 6D
    transform relative to a parent_frame
    """
    food_frame_transforms = []
    ellipsoid_radii = []

    fx = camera_intrinsics.fx
    fy = camera_intrinsics.fy
    cx = camera_intrinsics.cx
    cy = camera_intrinsics.cy

    # Get the parent to camera transform
    parent_to_camera_matrix = transform_to_matrix(parent_to_camera_transform)

    # Cameras are z-forward, x-right
    for i in range(len(bounding_ellipses)):
        ellipse = bounding_ellipses[i]
        center_float, size, angle = ellipse # (u, v) dimensions
        center = [int(center_float[0]), int(center_float[1])]
        angle = angle*np.pi/180.0 # convert to radians
        z = table[center[1], center[0]] / 1000.0 # height is in mm
        food_depth = depth_image[center[1], center[0]] / 1000.0 # height is in mm

        # Get the origin and size of the ellipse
        ellipse_cx = (center[0] - cx) * z / fx
        ellipse_cy = (center[1] - cy) * z / fy
        # NOTE: I think the computation of is not necessarily correct because
        # fx and fy are different and the ellipse is rotated. But anyway this
        # may be good enough.
        ellipse_size_w = size[0] * z / fx
        ellipse_size_h = size[1] * z / fy

        # Determine the major axis
        if ellipse_size_w > ellipse_size_h:
            rotation = angle
            ellipsoid_radius_x = ellipse_size_w/2.0
            ellipsoid_radius_y = ellipse_size_h/2.0
        else:
            rotation = angle + np.pi/2.0
            ellipsoid_radius_x = ellipse_size_h/2.0
            ellipsoid_radius_y = ellipse_size_w/2.0
        # Adjust the angle to face the top of the image, where the human is
        if rotation < np.pi:
            rotation += np.pi
        ellipsoid_radius_z = (z - food_depth) / 2.0

        # To get the orientation of the food relative to the camera orientation,
        # first rotate 180 degrees around x, then rotate around z -rotation.
        Rx = transformations.rotation_matrix(np.pi, (1,0,0))
        Rz = transformations.rotation_matrix(-rotation, (0,0,1))
        camera_to_food_matrix = transformations.concatenate_matrices(Rx, Rz)
        camera_to_food_matrix[0][3] = ellipse_cx
        camera_to_food_matrix[1][3] = ellipse_cy
        camera_to_food_matrix[2][3] = z

        # Chain the transforms together and put it all together
        final_transform = matrix_to_transform(transformations.concatenate_matrices(parent_to_camera_matrix, camera_to_food_matrix))

        food_frame_transforms.append(final_transform)
        ellipsoid_radii.append((ellipsoid_radius_x, ellipsoid_radius_y, ellipsoid_radius_z))

    return food_frame_transforms, ellipsoid_radii

def get_forktip_to_table_distances(
    forque_transform_data, parent_to_camera_transform, depth_image, camera_intrinsics, plate_uv, plate_r):

    forktip_to_table_distances, forktip_to_table_timestamps = [], []

    camera_projection_matrix = np.zeros((3,3))
    camera_projection_matrix[0,0] = camera_intrinsics.fx
    camera_projection_matrix[1,1] = camera_intrinsics.fy
    camera_projection_matrix[0,2] = camera_intrinsics.cx
    camera_projection_matrix[1,2] = camera_intrinsics.cy
    camera_projection_matrix[2,2] = 1

    # Get the parent to camera transform
    parent_to_camera_matrix = transform_to_matrix(parent_to_camera_transform)
    camera_to_parent_matrix = transformations.inverse_matrix(parent_to_camera_matrix)

    plate_u = plate_uv[0]
    plate_v = plate_uv[1]

    for forque_transform in forque_transform_data:
        parent_to_forque_matrix = transform_to_matrix(forque_transform)
        camera_to_forque_matrix = transformations.concatenate_matrices(camera_to_parent_matrix, parent_to_forque_matrix)

        # Get the forktip location in pixels
        # print(camera_projection_matrix, camera_to_forque_matrix)
        [[forque_u], [forque_v], [forque_z]] = np.dot(camera_projection_matrix, camera_to_forque_matrix[0:3,3:])
        forque_u /= forque_z
        forque_v /= forque_z
        # forque_v = camera_intrinsics.height - forque_v  # OpenCv has +y going down

        # Check if the forktip is out of bounds of the camera
        if ((forque_u < 0) or (forque_u > camera_intrinsics.width) or (forque_v < 0) or (forque_v > camera_intrinsics.height)):
            continue

        # Check if the forktip is above the plate
        if ((forque_u - plate_u)**2 + (forque_v - plate_v)**2)**0.5 <= plate_r:
            plate_depth = depth_image[int(forque_v), int(forque_u)] / 1000.0 # convert from mm
            if plate_depth > 0: # Deal with unperceived parts of the image
                forktip_to_table_distances.append(plate_depth - camera_to_forque_matrix[2,3])
                forktip_to_table_timestamps.append(forque_transform.time)

    forktip_to_table_distances = median_filter_hz(forktip_to_table_distances)
    forktip_to_table_timestamps = np.array(forktip_to_table_timestamps)

    return forktip_to_table_distances, forktip_to_table_timestamps

def get_deliminating_timestamps(force_data, forque_transform_data, forque_distance_to_mouth,
    forktip_to_table_distances, forktip_to_table_timestamps,
    stationary_duration=0.5, height_threshold=0.05,
    distance_to_mouth_threshold=0.35, distance_threshold=0.05,
    force_proportion=1.0/3,
    distance_from_min_height=0.07, time_from_liftoff=2.0, extraction_epsilon=0.01,
    image_name=None, save_process=False, out_dir=None):
    """
    force_data is a list of WrenchStamped messages with the F/T data
    forque_transform_data is a list of TransformStamped messages with the forktip pose

    Define contact time to be the first time the forktip to table distance is <= 0

    Define start as the end of the max-y stationary_duration sec long interval
    where the forque position is within a ball of radii distance_threshold,
    the y is height_threshold above the table, and the forque is distance_to_mouth_threshold
    from the mouth, and time is before contact_time.

    Define extraction time to be the latest time the fork was within
    extraction_epsilon of its minimum point, within time_from_liftoff sec of
    the last time the fork was distance_from_min_height above its min height.

    Define end as the last time the fork was distance_from_min_height
    above its min height
    """
    bag_start_time = min(force_data[0].time, forque_transform_data[0].time)

    # Crop the data at the fork's min distance to the mouth
    forque_closest_to_mouth_i = np.argmin(forque_distance_to_mouth)
    forque_transform_data = forque_transform_data[:forque_closest_to_mouth_i+1]
    forque_distance_to_mouth = forque_distance_to_mouth[:forque_closest_to_mouth_i+1]
    forque_closest_to_mouth_time = forque_transform_data[forque_closest_to_mouth_i].time
    force_data_i = np.argmax((np.array([wrench_stamped.time for wrench_stamped in force_data]) > forque_closest_to_mouth_time))
    if force_data_i == 0: force_data_i = len(force_data) # in case the closest to mouth is the last time
    force_data = force_data[:force_data_i]

    # Get the min y of the forque
    forque_y_data = []
    forque_time_data = []
    for transform_stamped in forque_transform_data:
        forque_y_data.append(transform_stamped.y)
        forque_time_data.append(transform_stamped.time - bag_start_time)
    forque_y_data = np.array(forque_y_data)
    forque_time_data = np.array(forque_time_data)
    min_height = forque_y_data.min()

    contact_i = np.argmax(forktip_to_table_distances <= 0)
    if contact_i == 0: # If the forktip is never <= 0
        contact_i = np.argmin(forktip_to_table_distances)
    contact_time = forktip_to_table_timestamps[contact_i] - bag_start_time

    # Get start time
    max_distances = []
    max_distances_times = []
    action_start_time = None
    min_movement = None
    action_start_time_fallback = 0.0
    min_movement_fallback = None
    highest_y_of_start_time_contenders = None
    for i in range(len(forque_transform_data)):
        i_time = forque_transform_data[i].time - bag_start_time
        if i_time >= contact_time: #latest_timestamp_close_to_table:
            break
        # Check the height condition
        if forque_transform_data[i].y < min_height + height_threshold:
            continue
        # Check the distance to mouth condition
        if forque_distance_to_mouth[i] <= distance_to_mouth_threshold:
            continue
        # Get a range of length stationary_duration
        success = False
        for j in range(i+1, len(forque_transform_data)):
            j_time = forque_transform_data[j].time - bag_start_time
            if j_time >= contact_time: #latest_timestamp_close_to_table:
                break
            # if (j_time) > 4:
            #     print("Distance to Mouth", forque_distance_to_mouth[j])
            # Check the height condition
            if forque_transform_data[j].y < min_height + height_threshold:
                break
            # Check the distance to mouth condition
            if forque_distance_to_mouth[j] <= distance_to_mouth_threshold:
                # if (j_time) > 4:
                #     print("Within threshold!")
                break
            if (j_time - i_time) > stationary_duration and (j-i) > 2:
                success = True
                break
        # We found a stationary_duration length range that satisfies the height
        # and distance from mouth thresholds!
        if success:
            points = []
            for k in range(i,j):
                points.append([
                    forque_transform_data[k].x,
                    forque_transform_data[k].y,
                    forque_transform_data[k].z,
                ])
            points = np.array(points)
            # print("Got matching range from %f to %f" % (i_time - bag_start_time, j_time - bag_start_time), points)
            distances = np.linalg.norm(points - points.mean(axis=0), axis=1)
            max_y = np.max(points[:,1])
            # print("Max Distance: ", distances.max())
            max_distance = distances.max()
            max_distances.append(max_distance)
            max_distances_times.append(j_time)
            if max_distance <= distance_threshold:
                if highest_y_of_start_time_contenders is None or max_y >= highest_y_of_start_time_contenders:
                    action_start_time = j_time
                    min_movement = max_distance
                    highest_y_of_start_time_contenders = max_y
            # As a fallback, if no segment has the fork moving less than distance_threshold, take the segment with least motion
            if min_movement_fallback is None or max_distance < min_movement_fallback:
                action_start_time_fallback = j_time
                min_movement_fallback = max_distance

    if action_start_time is None:
        action_start_time = action_start_time_fallback

    # Get the contact time
    force_magnitude = []
    force_time_data = []
    for wrench_stamped in force_data:
        fx = wrench_stamped.force_x
        fy = wrench_stamped.force_y
        fz = wrench_stamped.force_z
        force_magnitude.append((fx**2.0 + fy**2.0 + fz**2.0)**0.5)
        force_time_data.append(wrench_stamped.time - bag_start_time)
    force_magnitude = np.array(force_magnitude)
    force_time_data = np.array(force_time_data)
    # max_force = np.max(force_magnitude)
    # max_force_time = force_time_data[np.argmax(force_magnitude)]
    # contact_i = np.argmax(np.logical_and(force_magnitude >= force_proportion*max_force, force_time_data > action_start_time))
    # contact_time = force_time_data[contact_i]

    # Get the extraction time
    points_close_to_table = np.logical_and(forque_y_data <= min_height + distance_from_min_height, forque_time_data > contact_time)
    latest_point_close_to_table_i = len(points_close_to_table) - 1 - np.argmax(points_close_to_table[::-1])
    action_end_time = forque_time_data[latest_point_close_to_table_i]
    points_close_to_liftoff = np.logical_and(
        np.logical_and(
            forque_time_data >= action_end_time - time_from_liftoff,
            points_close_to_table,
        ),
        # NOTE: ideally add the below in! Will prevent action end time and extraction time from being the same
        # np.logical_and(
        #     forque_time_data < action_end_time,
        #     forque_time_data > contact_time,
        # ),
        forque_time_data > contact_time,
    )
    masked_y_data = np.where(points_close_to_liftoff, forque_y_data, np.Inf)
    min_height_after_contact = np.min(masked_y_data)
    points_within_epsilon_of_min_height_after_contact = np.logical_and(masked_y_data >= min_height_after_contact, masked_y_data <= (min_height_after_contact + extraction_epsilon))
    extraction_i = len(points_within_epsilon_of_min_height_after_contact) - 1 - np.argmax(points_within_epsilon_of_min_height_after_contact[::-1])
    # extraction_i = np.argmin(masked_y_data)
    extraction_time = forque_time_data[extraction_i]

    # Debugging
    if save_process and (image_name is None or out_dir is not None):
        fig, axes = plt.subplots(5, 1, figsize=(6, 25), sharex=True)
        axes[0].plot([transform_stamped.time - bag_start_time for transform_stamped in forque_transform_data], forque_distance_to_mouth)
        axes[0].set_xlabel("Elapsed Time (sec)")
        axes[0].set_ylabel("Distance to Mouth (m)")
        axes[0].axhline(distance_to_mouth_threshold, linestyle='--', c='k')
        axes[1].plot(max_distances_times, max_distances)
        axes[1].set_xlabel("Elapsed Time (sec)")
        axes[1].set_ylabel("Max Amount Moved (m) in %f Contender Sec Interval" % stationary_duration)
        # axes[1].set_ylim(0.0, np.max(max_distances)*1.5)
        axes[1].axhline(distance_threshold, linestyle='--', c='k')
        axes[2].plot(force_time_data, force_magnitude)
        axes[2].set_xlabel("Elapsed Time (sec)")
        axes[2].set_ylabel("Force Magnitude (newtons)")
        axes[3].plot(forque_time_data, forque_y_data)
        axes[3].set_xlabel("Elapsed Time (sec)")
        axes[3].set_ylabel("Forktip Y (m)")
        axes[3].axhline(min_height + height_threshold, linestyle='--', c='k')
        axes[3].axhline(min_height + distance_from_min_height, linestyle='--', c='k')
        axes[4].plot([t - bag_start_time for t in forktip_to_table_timestamps], forktip_to_table_distances)
        axes[4].set_xlabel("Elapsed Time (sec)")
        axes[4].set_ylabel("Forktip Distance to Initial Depth Reading (m)")
        axes[4].axhline(0.0, linestyle='--', c='k')
        for i in range(len(axes)):
            axes[i].axvline(action_start_time, linestyle='--', c='r')
            axes[i].axvline(contact_time, linestyle='--', c='g')
            axes[i].axvline(extraction_time, linestyle='--', c='b')
            axes[i].axvline(action_end_time, linestyle='--', c='k')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if image_name is None:
            plt.show()
        else:
            if out_dir is not None:
                plt.savefig(os.path.join(out_dir, image_name+"_get_time_process.png"))
                plt.clf()

    return action_start_time, contact_time, extraction_time, action_end_time

def get_action_schema_elements(
    food_origin_frames, action_start_time, contact_time,
    extraction_time, end_time, force_data, forque_transform_data,
    fork_tip_frame_id,
    pre_grasp_initial_transform_linear_velocity_window=0.5, pre_grasp_initial_transform_distance=0.1,
    pre_grasp_ft_proportion=0.5,
    approach_frame_id="approach_frame",
    grasp_ft_proportion=0.5,
):
    """
    Parameters:
        - pre_grasp_initial_transform_linear_velocity_window: Num secs over which to get the movement near contact
        - pre_grasp_initial_transform_distance: Distance (m) away from contact position to extrapolate the robot's motion
        - pre_grasp_ft_proportion: proportion of the max force during grasp to set as the pre_grasp force threshold
        - grasp_ft_proportion: Same but for the force at which to transition from grasp to extraction

    Returns:
        - food_reference_frame (TransformStamped)
            - Origin (x,y) is the center of the food's bounding ellipse, z is
              level with the table. For rotation, +Z is out of the table and +X
              is along the major axis, facing the user.
        - pre_grasp_target_offset (Vector3)
            - Computed as the difference between the forktip (x,y,z) at
              contact_time and the food_reference_frame's origin.
        - pre_grasp_initial_utensil_transform (PoseStamped)
            - The utensil's 6D pose at action_start_time in the food_reference_frame
        - pre_grasp_force_threshold (float, newtons)
            - Defined as pre_grasp_ft_proportion of the max torque between start_time
              and contact_time.
        - approach_frame (TransformStamped)
            - Transform from the food frame to the approach frame, which has the same
              origin and is oriented where +x points away from the fork at the
              pre-grasp initial transform.
        - grasp_in_food_twist (TwistStamped)
            - Take the utensil's 6D pose at contact_time, subtract if from the
              fork's  6D pose at extraction_time, and divide it by the duration.
            - NOTE: Doesn't work for multiple twirls with noodles.
        - grasp_force_threshold (float, newtons)
            - Defined as grasp_ft_proportion of the max force between contact_time
              and extraction_time.
        - grasp_torque_threshold (float, newston-meters)
            - Defined as grasp_ft_proportion of the max torque between contact_time
              and extraction_time.
        - grasp_duration (float, secs)
            - extraction_time - contact_time
        - extraction_out_of_food_twist (TwistStamped)
            - Take the utensil's 6D pose at extraction_time, subtract if from the
              fork's  6D pose at end_time, and divide it by the duration.
        - extraction_duration (float, secs)
            - end_time - extraction_time
    """
    # First, find food_reference_frame and pre_grasp_target_offset by
    # determining which of the food_origin_frames the fork is closest to at contact.
    forque_transform_at_start = None
    forque_transform_at_beginning_of_contact_window = None
    forque_transform_at_contact = None
    forque_transform_at_extraction = None
    forque_transform_at_end = None
    for forque_transform in forque_transform_data:
        # print(forque_transform.time, action_start_time, contact_time, extraction_time, end_time )
        # start
        if forque_transform.time > action_start_time and forque_transform_at_start is None:
            forque_transform_at_start = forque_transform
        # contact
        if forque_transform.time > contact_time - pre_grasp_initial_transform_linear_velocity_window and forque_transform_at_beginning_of_contact_window is None:
            forque_transform_at_beginning_of_contact_window = forque_transform
        if forque_transform.time > contact_time and forque_transform_at_contact is None:
            forque_transform_at_contact = forque_transform
            min_dist = None
            min_dist_i = None
            for i in range(len(food_origin_frames)):
                food_origin_frame = food_origin_frames[i]
                dist = (
                    (forque_transform.x - food_origin_frame.x)**2.0 +
                    (forque_transform.y - food_origin_frame.y)**2.0 +
                    (forque_transform.z - food_origin_frame.z)**2.0)**0.5
                # print(food_origin_frame, dist)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_dist_i = i
                    food_reference_frame = food_origin_frame
                    parent_to_food_matrix = transform_to_matrix(food_reference_frame)
                    parent_to_forque_matrix = transform_to_matrix(forque_transform)
                    food_offset_matrix = transformations.concatenate_matrices(transformations.inverse_matrix(parent_to_food_matrix), parent_to_forque_matrix)
                    pre_grasp_target_offset = Vector3(food_offset_matrix[0][3], food_offset_matrix[1][3], food_offset_matrix[2][3])
                    # NOTE: The error with the below code is that its in the TableBody frame!
                    # pre_grasp_target_offset = Vector3()
                    # pre_grasp_target_offset.x = forque_transform.x - food_origin_frame.x
                    # pre_grasp_target_offset.y = forque_transform.y - food_origin_frame.y
                    # pre_grasp_target_offset.z = forque_transform.z - food_origin_frame.z
            # print("Min distance food reference frame is %d" % min_dist_i)
            # print(food_reference_frame)
            # print("parent_to_food_matrix", parent_to_food_matrix)
            # print("pre_grasp_target_offset", pre_grasp_target_offset)
            # print("parent_to_forque_matrix", parent_to_forque_matrix)
        # extraction
        if forque_transform.time > extraction_time and forque_transform_at_extraction is None:
            forque_transform_at_extraction = forque_transform
        # end
        if forque_transform.time > end_time and forque_transform_at_end is None:
            forque_transform_at_end = forque_transform
            break
    if forque_transform_at_end is None: # If the end time is the bag end time
        forque_transform_at_end = forque_transform_data[-1]

    # Compute pre_grasp_initial_utensil_transform, by getting the movement near contact and extrapolating that a fixed distance from the contact position
    d_position_near_contact = np.array([
        forque_transform_at_beginning_of_contact_window.x - forque_transform_at_contact.x,
        forque_transform_at_beginning_of_contact_window.y - forque_transform_at_contact.y,
        forque_transform_at_beginning_of_contact_window.z - forque_transform_at_contact.z,
    ])
    d_position_near_contact /= np.linalg.norm(d_position_near_contact)
    fork_start_transform = TransformStamped(
        None,
        None,
        None,
        d_position_near_contact[0] * pre_grasp_initial_transform_distance + forque_transform_at_contact.x,
        d_position_near_contact[1] * pre_grasp_initial_transform_distance + forque_transform_at_contact.y,
        d_position_near_contact[2] * pre_grasp_initial_transform_distance + forque_transform_at_contact.z,
        forque_transform_at_contact.roll,
        forque_transform_at_contact.pitch,
        forque_transform_at_contact.yaw,
    )

    parent_to_food_matrix = transform_to_matrix(food_reference_frame)
    parent_to_fork_start_matrix = transform_to_matrix(fork_start_transform)
    pre_grasp_initial_utensil_transform_matrix = transformations.concatenate_matrices(transformations.inverse_matrix(parent_to_food_matrix), parent_to_fork_start_matrix)
    pre_grasp_initial_utensil_transform = matrix_to_pose(pre_grasp_initial_utensil_transform_matrix, frame_id=food_reference_frame.child_frame)

    # Get pre_grasp_force_threshold
    max_force = None
    for wrench_stamped in force_data:
        if wrench_stamped.time >= action_start_time and wrench_stamped.time <= contact_time:
            fx = wrench_stamped.force_x
            fy = wrench_stamped.force_y
            fz = wrench_stamped.force_z
            force_magnitude = (fx**2.0 + fy**2.0 + fz**2.0)**0.5
            if max_force is None or force_magnitude > max_force:
                max_force = force_magnitude
    pre_grasp_force_threshold = pre_grasp_ft_proportion*max_force

    # Get Approach Frame
    fork_to_food_vector = Vector3(pre_grasp_initial_utensil_transform.x, pre_grasp_initial_utensil_transform.y, pre_grasp_initial_utensil_transform.z) # since food is at the origin
    angle_to_rotate = np.arctan2(-fork_to_food_vector.y, -fork_to_food_vector.x) # we want to point away from the fork
    food_frame_to_approach_frame_matrix = transformations.rotation_matrix(angle_to_rotate, [0,0,1])
    approach_frame = matrix_to_transform(
        m=food_frame_to_approach_frame_matrix,
        parent_frame=food_reference_frame.child_frame,
        child_frame=approach_frame_id,
        time=food_reference_frame.time,
    )
    approach_to_parent_matrix = transformations.concatenate_matrices(transformations.inverse_matrix(food_frame_to_approach_frame_matrix), transformations.inverse_matrix(parent_to_food_matrix))

    # Get grasp_in_food_twist and grasp_duration
    grasp_duration = extraction_time - contact_time
    # Get the fork pose at extraction_time in the frame of fork pose at contact time
    parent_to_fork_contact_matrix = transform_to_matrix(forque_transform_at_contact)
    approach_to_fork_contact_matrix = transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_contact_matrix)
    parent_to_fork_extraction_matrix = transform_to_matrix(forque_transform_at_extraction)
    # approach_to_fork_extraction_matrix = transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_extraction_matrix)
    # print(approach_to_fork_contact_matrix, approach_to_fork_extraction_matrix)
    fork_contact_to_extraction_transform_matrix = transformations.concatenate_matrices(transformations.inverse_matrix(parent_to_fork_contact_matrix), parent_to_fork_extraction_matrix)
    # Translation in appraoch frame
    # print("A: displacement in fork contact frame", fork_contact_to_extraction_transform_matrix[0:3,3:])
    fork_contact_to_extraction_transform_matrix[0:3,3:] = np.dot(approach_to_fork_contact_matrix[0:3,0:3], fork_contact_to_extraction_transform_matrix[0:3,3:])
    # print("A: displacement in approach frame", fork_contact_to_extraction_transform_matrix[0:3,3:])
    # print("A: transform", approach_to_fork_contact_matrix)
    # print("A: forque_transform_at_contact", forque_transform_at_contact)
    # print("A: approach_frame", approach_frame)
    # print("A: approach_to_parent_matrix", approach_to_parent_matrix)
    # fork_contact_to_extraction_transform_matrix[0,3] = approach_to_fork_extraction_matrix[0,3] - approach_to_fork_contact_matrix[0,3]
    # fork_contact_to_extraction_transform_matrix[1,3] = approach_to_fork_extraction_matrix[1,3] - approach_to_fork_contact_matrix[1,3]
    # fork_contact_to_extraction_transform_matrix[2,3] = approach_to_fork_extraction_matrix[2,3] - approach_to_fork_contact_matrix[2,3]
    grasp_in_food_twist = matrix_to_twist(
        fork_contact_to_extraction_transform_matrix,
        grasp_duration,
        frame_id="linear_%s_angular_%s" % (approach_frame_id, fork_tip_frame_id),
        time=forque_transform_at_contact.time,
    )

    # Get grasp_force_threshold, grasp_torque_threshold
    max_force, max_torque = None, None
    for wrench_stamped in force_data:
        if wrench_stamped.time >= contact_time and wrench_stamped.time <= extraction_time:
            fx = wrench_stamped.force_x
            fy = wrench_stamped.force_y
            fz = wrench_stamped.force_z
            force_magnitude = (fx**2.0 + fy**2.0 + fz**2.0)**0.5
            if max_force is None or force_magnitude > max_force:
                max_force = force_magnitude
            tx = wrench_stamped.torque_x
            ty = wrench_stamped.torque_y
            tz = wrench_stamped.torque_z
            torque_magnitude = (tx**2.0 + ty**2.0 + tz**2.0)**0.5
            if max_torque is None or torque_magnitude > max_torque:
                max_torque = torque_magnitude
    grasp_force_threshold = grasp_ft_proportion*max_force
    grasp_torque_threshold = grasp_ft_proportion*max_torque

    # Get extraction_out_of_food_twist and extraction_duration
    extraction_duration = end_time - extraction_time
    # Get the fork pose at end_time in the frame of fork pose at extraction_time
    parent_to_fork_extraction_matrix = transform_to_matrix(forque_transform_at_extraction)
    approach_to_fork_extraction_matrix = transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_extraction_matrix)
    parent_to_fork_end_matrix = transform_to_matrix(forque_transform_at_end)
    # approach_to_fork_end_matrix = transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_end_matrix)
    fork_extraction_to_end_transform_matrix = transformations.concatenate_matrices(transformations.inverse_matrix(parent_to_fork_extraction_matrix), parent_to_fork_end_matrix)
    # Translation in appraoch frame
    fork_extraction_to_end_transform_matrix[0:3,3:] = np.dot(approach_to_fork_extraction_matrix[0:3,0:3], fork_extraction_to_end_transform_matrix[0:3,3:])
    # fork_extraction_to_end_transform_matrix[0,3] = approach_to_fork_end_matrix[0,3] - approach_to_fork_extraction_matrix[0,3]
    # fork_extraction_to_end_transform_matrix[1,3] = approach_to_fork_end_matrix[1,3] - approach_to_fork_extraction_matrix[1,3]
    # fork_extraction_to_end_transform_matrix[2,3] = approach_to_fork_end_matrix[2,3] - approach_to_fork_extraction_matrix[2,3]
    extraction_out_of_food_twist = matrix_to_twist(
        fork_extraction_to_end_transform_matrix,
        extraction_duration,
        frame_id="linear_%s_angular_%s" % (approach_frame_id, fork_tip_frame_id),
        time=forque_transform_at_extraction.time,
    )

    return (
        food_reference_frame, # geometry_msgs/TransformStamped
        pre_grasp_target_offset, # geometry_msgs/Vector3
        pre_grasp_initial_utensil_transform, # geometry_msgs/PoseStamped
        pre_grasp_force_threshold, # float, newtons
        approach_frame, # geometry_msgs/TransformStamped
        grasp_in_food_twist, # geometry_msgs/TwistStamped
        grasp_force_threshold, # float, newtons
        grasp_torque_threshold, # float, newston-meters
        grasp_duration, # float, secs
        extraction_out_of_food_twist, # geometry_msgs/TwistStamped
        extraction_duration, # float, secs
    )

def linear_average_transform(t0, t1, alpha):
    if t0.parent_frame != t1.parent_frame:
        print("ERROR? linear_average_transform have different frame_ids %s and %s" % (t0.parent_frame, t1.parent_frame))
    if t0.child_frame != t1.child_frame:
        print("ERROR? linear_average_transform have different child_frame_ids %s and %s" % (t0.child_frame, t1.child_frame))
    retval = TransformStamped(
        parent_frame=t0.parent_frame,
        child_frame=t0.child_frame,
        time=(1-alpha)*t0.time + alpha*t1.time,
        x=(1-alpha)*t0.x + alpha*t1.x,
        y=(1-alpha)*t0.y + alpha*t1.y,
        z=(1-alpha)*t0.z + alpha*t1.z,
        roll=(1-alpha)*t0.roll + alpha*t1.roll,
        pitch=(1-alpha)*t0.pitch + alpha*t1.pitch,
        yaw=(1-alpha)*t0.yaw + alpha*t1.yaw,
    )
    return retval

def apply_twist(start_transform, approach_frame, food_reference_frame, twist, duration, granularity):
    """
    Let F be the frame that start_transform is in. Then, approach_frame must have
    as its parent_frame F. And then the angular velocity of the twist will be
    interpreted in the start_transform frame, but the linear velocity will be
    interpreted in the approach_frame.
    """
    retval = []
    parent_to_start_transform_matrix = transform_to_matrix(start_transform)
    parent_to_food_matrix = transform_to_matrix(food_reference_frame)
    food_to_approach_matrix = transform_to_matrix(approach_frame)
    start_transform_to_approach_matrix = transformations.concatenate_matrices(transformations.concatenate_matrices(transformations.inverse_matrix(parent_to_start_transform_matrix), parent_to_food_matrix), food_to_approach_matrix)

    for i in range(1, granularity+1):
        elapsed_time = duration*float(i)/granularity
        # Apply the angular velocity in the start_transform frame
        elapsed_transform = transformations.euler_matrix(
            twist.angular_x*elapsed_time,
            twist.angular_y*elapsed_time,
            twist.angular_z*elapsed_time,
            EULER_ORDER,
        )
        # Apply the linear velocity in the approach frame
        displacement = np.array([[twist.linear_x], [twist.linear_y], [twist.linear_z]])*elapsed_time
        elapsed_transform[0:3, 3:] = np.dot(start_transform_to_approach_matrix[0:3,0:3], displacement)

        final_matrix = transformations.concatenate_matrices(parent_to_start_transform_matrix, elapsed_transform)
        transform = matrix_to_transform(
            final_matrix,
            parent_frame=start_transform.parent_frame,
            child_frame=start_transform.child_frame,
            time=start_transform.time + elapsed_time,
        )
        retval.append(transform)

    return retval

def get_predicted_forque_transform_data(
    action_start_time,
    contact_time,
    food_reference_frame, # geometry_msgs/TransformStamped
    pre_grasp_target_offset, # geometry_msgs/Vector3
    pre_grasp_initial_utensil_transform, # geometry_msgs/PoseStamped
    pre_grasp_force_threshold, # float, newtons
    approach_frame, # geometry_msgs/TransformStamped
    grasp_in_food_twist, # geometry_msgs/TwistStamped
    grasp_force_threshold, # float, newtons
    grasp_torque_threshold, # float, newston-meters
    grasp_duration, # float, secs
    extraction_out_of_food_twist, # geometry_msgs/TwistStamped
    extraction_duration, # float, secs
    fork_tip_frame_id,
    granularity=20,
):
    predicted_forque_transform_data = []

    # Pre-grasp
    parent_to_food_matrix = transform_to_matrix(food_reference_frame)
    food_to_forque = pose_to_matrix(pre_grasp_initial_utensil_transform)
    parent_to_forque_start_matrix = transformations.concatenate_matrices(parent_to_food_matrix, food_to_forque)
    start_forque_transform = matrix_to_transform(
        parent_to_forque_start_matrix,
        parent_frame=food_reference_frame.parent_frame,
        child_frame=fork_tip_frame_id,
        time=action_start_time,
    )

    food_to_food_offset_matrix = np.eye(4)
    food_to_food_offset_matrix[0][3] = pre_grasp_target_offset.x
    food_to_food_offset_matrix[1][3] = pre_grasp_target_offset.y
    food_to_food_offset_matrix[2][3] = pre_grasp_target_offset.z
    parent_to_food_offset_matrix = transformations.concatenate_matrices(parent_to_food_matrix, food_to_food_offset_matrix)
    parent_to_forque_end_matrix = np.copy(parent_to_forque_start_matrix)
    parent_to_forque_end_matrix[0][3] = parent_to_food_offset_matrix[0][3]
    parent_to_forque_end_matrix[1][3] = parent_to_food_offset_matrix[1][3]
    parent_to_forque_end_matrix[2][3] = parent_to_food_offset_matrix[2][3]
    end_forque_transform = matrix_to_transform(
        parent_to_forque_end_matrix,
        parent_frame=food_reference_frame.parent_frame,
        child_frame=fork_tip_frame_id,
        time=contact_time,
    )

    for i in range(granularity+1):
        predicted_forque_transform = linear_average_transform(start_forque_transform, end_forque_transform, float(i)/granularity)
        predicted_forque_transform_data.append(predicted_forque_transform)

    # Grasp
    predicted_forque_transform_data += apply_twist(predicted_forque_transform_data[-1], approach_frame, food_reference_frame, grasp_in_food_twist, grasp_duration, granularity)

    # Extraction
    predicted_forque_transform_data += apply_twist(predicted_forque_transform_data[-1], approach_frame, food_reference_frame, extraction_out_of_food_twist, extraction_duration, granularity)

    return predicted_forque_transform_data

def graph_forktip_motion(forque_transform_data, forque_transform_data_raw, predicted_forque_transform_data, first_timestamp,
    action_start_time, contact_time, extraction_time, action_end_time,
    image_name=None, out_dir=None):

    # Get actual data
    ts, xs, ys, zs, rolls, pitches, yaws = [], [], [], [], [], [], []
    prev_euler = [None, None, None]
    for msg in forque_transform_data:
        ts.append(msg.time - first_timestamp)
        xs.append(msg.x)
        ys.append(msg.y)
        zs.append(msg.z)
        roll, pitch, yaw = fix_euler_continuity(msg.roll, msg.pitch, msg.yaw, *prev_euler)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
        prev_euler = [roll, pitch, yaw]

    # Get raw data
    ts_raw, xs_raw, ys_raw, zs_raw, rolls_raw, pitches_raw, yaws_raw = [], [], [], [], [], [], []
    for msg in forque_transform_data_raw:
        ts_raw.append(msg.time - first_timestamp)
        xs_raw.append(msg.x)
        ys_raw.append(msg.y)
        zs_raw.append(msg.z)
        rolls_raw.append(msg.roll)
        pitches_raw.append(msg.pitch)
        yaws_raw.append(msg.yaw)

    # Get the predicted data
    pred_ts, pred_xs, pred_ys, pred_zs, pred_rolls, pred_pitches, pred_yaws = [], [], [], [], [], [], []
    prev_euler = [None, None, None]
    for msg in predicted_forque_transform_data:
        pred_ts.append(msg.time - first_timestamp)
        pred_xs.append(msg.x)
        pred_ys.append(msg.y)
        pred_zs.append(msg.z)
        roll, pitch, yaw = fix_euler_continuity(msg.roll, msg.pitch, msg.yaw, *prev_euler)
        pred_rolls.append(roll)
        pred_pitches.append(pitch)
        pred_yaws.append(yaw)
        prev_euler = [roll, pitch, yaw]

    graph_data = [
        [
            [ts, xs, "Forktip X (m)", pred_ts, pred_xs, ts_raw, xs_raw],
            [ts, ys, "Forktip Y (m)", pred_ts, pred_ys, ts_raw, ys_raw],
            [ts, zs, "Forktip Z (m)", pred_ts, pred_zs, ts_raw, zs_raw],
        ],
        [
            [ts, rolls, "Forktip Roll (rad)", pred_ts, pred_rolls, ts_raw, rolls_raw],
            [ts, pitches, "Forktip Pitch (rad)", pred_ts, pred_pitches, ts_raw, pitches_raw],
            [ts, yaws, "Forktip Yaw (rad)", pred_ts, pred_yaws, ts_raw, yaws_raw],
        ],
    ]
    if out_dir is not None or image_name is None:
        fig, axes = plt.subplots(len(graph_data), len(graph_data[0]), figsize=(20, 10), sharex=True)
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                # axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
                # axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
                axes[i][j].plot(graph_data[i][j][0], graph_data[i][j][1], linestyle='-', c='k', marker='o', markersize=4)
                axes[i][j].plot(graph_data[i][j][5], graph_data[i][j][6], linestyle='-', c='b', alpha=0.5)
                axes[i][j].plot(graph_data[i][j][3], graph_data[i][j][4], linestyle='--', c='r')
                axes[i][j].set_xlabel("Elapsed Time (sec)")
                axes[i][j].set_ylabel(graph_data[i][j][2])
                axes[i][j].axvline(action_start_time, linestyle='--', c='r')
                axes[i][j].axvline(contact_time, linestyle='--', c='g')
                axes[i][j].axvline(extraction_time, linestyle='--', c='b')
                axes[i][j].axvline(action_end_time, linestyle='--', c='k')
                # axes[i][j].grid(visible=True, which='both')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if image_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_dir, image_name+"_fork_pose.png"))
            plt.clf()

def process_trial(trial_data_paths, subject_num, food_name, trial_num, camera_intrinsics, verbose=False, out_dir=None):
    """
    Extract an action schema element from the trial data.

    Parameters
    ----------
    trial_data_paths : TrialDataPaths
        The paths to the trial data
    subject_num : int
        The subject number
    food_name : str
        The food name
    trial_num : int
        The trial number
    camera_intrinsics : CameraIntrinsics
        The camera intrinsics
    verbose : bool
        Whether or not to output verbose logs
    out_dir : str
        The directory to save the process images to

    Returns
    -------
    was_error : bool
        Whether or not there was an error
    action_schema_point : ActionSchemaPoint
        The action schema point
    """
    was_error = False
    action_schema_point = None

    # The below frames should match the static_transforms CSV file
    parent_frame = "world"
    camera_frame = "camera_color_optical_frame"
    forktip_frame = "fork_tip"
    mouth_frame = "MouthBody"
    expected_forque_hz = 50.0

    # Get the wrenches and poses
    force_data, forque_transform_data_raw = read_wrenches_poses(trial_data_paths.wrenches_poses, parent_frame, forktip_frame)
    forque_timstamp_data = [forque_transform_datapoint.time for forque_transform_datapoint in forque_transform_data_raw]

    # Get the transforms
    transforms = read_static_transforms(trial_data_paths.static_transforms)
    parent_to_camera_transform, parent_to_mouth_transform = None, None
    for transform in transforms:
        if transform.parent_frame == parent_frame and transform.child_frame == camera_frame:
            parent_to_camera_transform = transform
        elif transform.parent_frame == parent_frame and transform.child_frame == mouth_frame:
            parent_to_mouth_transform = transform
    if parent_to_camera_transform is None or parent_to_mouth_transform is None:
        raise Exception("Malformed static transforms file!")

    # Get the fork distances to the mouth
    forque_distance_to_mouth = get_forque_distance_to_mouth(forque_transform_data_raw, parent_to_mouth_transform)

    # Get the first image
    image_timestamps = []
    for filename in os.listdir(trial_data_paths.img_dir):
        try:
            image_timestamps.append(int(filename.split("_")[0]))
        except ValueError:
            continue
    image_timestamps.sort()
    first_camera_image_timestamp = None
    first_depth_image_timestamp = None
    for image_timestamp in image_timestamps:
        if os.path.exists(os.path.join(trial_data_paths.img_dir, f"{image_timestamp}_rgb.jpg")):
            first_camera_image_timestamp = image_timestamp
        if os.path.exists(os.path.join(trial_data_paths.img_dir, f"{image_timestamp}_depth.png")):
            first_depth_image_timestamp = image_timestamp
        if first_camera_image_timestamp is not None and first_depth_image_timestamp is not None:
            break
    first_camera_image = cv2.imread(os.path.join(trial_data_paths.img_dir, f"{first_camera_image_timestamp}_rgb.jpg"))
    first_camera_image = cv2.cvtColor(first_camera_image, cv2.COLOR_BGR2RGB)
    first_depth_image = cv2.imread(os.path.join(trial_data_paths.img_dir, f"{first_depth_image_timestamp}_depth.png"), cv2.IMREAD_ANYDEPTH)

    # Get the trial start and end times
    start_time = min(force_data[0].time, forque_timstamp_data[0], image_timestamps[0])
    end_time = max(force_data[-1].time, forque_timstamp_data[-1], image_timestamps[-1])

    # Smoothen the detected forque transforms
    forque_transform_data = smoothen_transforms(forque_transform_data_raw)
    # if verbose:
    #     print("Smoothened forque transforms:")
    #     pprint.pprint(forque_transform_data)

    # Remove the background of the first image
    image_name = f"subject{subject_num}_{food_name}_trial{trial_num}"
    image_without_background = remove_background(first_camera_image, image_name=image_name, save_process=True, out_dir=out_dir)
    if out_dir is not None:
        plt.imshow(image_without_background)
        plt.savefig(os.path.join(out_dir, image_name+"_removed_background.png"))
        plt.clf()

    # Get the food bounding boxes. Broccoli and jello need one more cluster because
    # its color(s) makes it harder to distinguish from the plate
    bounding_ellipses = get_food_bounding_box(
        image_without_background,
        k=4 if "broc" in image_name or "jello" in image_name else 3,
        image_name=image_name, save_process=True, out_dir=out_dir)
    if out_dir is not None:
        image_with_bounding_ellipses = first_camera_image.copy()
        color=(255, 0, 0)
        for ellipse in bounding_ellipses:
            cv2.ellipse(image_with_bounding_ellipses, ellipse, color, 2)
        plt.imshow(image_with_bounding_ellipses)
        plt.savefig(os.path.join(out_dir, image_name+"_bounding_box.png"))
        plt.clf()
        del image_with_bounding_ellipses

    # Fit the table
    plate_uv, plate_r, table, depth_image_clipped = fit_table(
        cv2.cvtColor(first_camera_image, cv2.COLOR_RGB2GRAY),
        first_depth_image, image_name=image_name, save_process=True, out_dir=out_dir)
    if len(bounding_ellipses) == 0:
        print("ERROR: No bounding ellipses! Resorting to the table")
        bounding_ellipses = [[plate_uv, (plate_r, plate_r), 0]]
    if out_dir is not None:
        image_with_plate = first_camera_image.copy()
        color=(255, 0, 0)
        cv2.circle(image_with_plate, plate_uv, plate_r, color, 2)
        plt.imshow(image_with_plate)
        plt.savefig(os.path.join(out_dir, image_name+"_with_plate.png"))
        plt.clf()
        del image_with_plate

    # Get the food origin frame for each food item
    food_origin_frames, ellipsoid_radii = get_food_origin_frames(bounding_ellipses, table, 
        parent_to_camera_transform, first_depth_image, camera_intrinsics)
    # if args.verbose:
    #     print("Food Origin Frames:")
    #     pprint.pprint(food_origin_frames)

    # Get the distance from the forktip to the table
    forktip_to_table_distances, forktip_to_table_timestamps = get_forktip_to_table_distances(
        forque_transform_data, parent_to_camera_transform, depth_image_clipped, camera_intrinsics, plate_uv, plate_r)

    if len(forktip_to_table_distances) == 0:
        print("ERROR, the user never picked up the fork!")
        was_error = True
    else:

        # Get the contact time
        action_start_time, contact_time, extraction_time, action_end_time = get_deliminating_timestamps(
            force_data, forque_transform_data, forque_distance_to_mouth, forktip_to_table_distances, forktip_to_table_timestamps,
            image_name=image_name, save_process=True, out_dir=out_dir)

        # Determine whether too much time was lost tracking to make this trial invalid
        forque_timstamp_data = np.array(forque_timstamp_data)
        msgs_during_action = np.logical_and((forque_timstamp_data >= start_time+action_start_time), (forque_timstamp_data <= start_time+action_end_time))
        num_msgs_during_action = np.sum(msgs_during_action)
        avg_hz = num_msgs_during_action/(action_end_time-action_start_time)
        msg_diffs = np.diff(forque_timstamp_data)[msgs_during_action[1:]]
        np.insert(msg_diffs, 0, start_time+action_start_time)
        np.insert(msg_diffs, msg_diffs.shape[0], start_time+action_end_time)
        max_interval_between_messages = np.max(msg_diffs)
        if args.verbose:
            print("avg_hz", avg_hz, "max_interval_between_messages", max_interval_between_messages)
        if (avg_hz <= 0.5*expected_forque_hz) or (max_interval_between_messages >= 25/expected_forque_hz):
            print("TOO MUCH TIME LOST TRACKING, avg_hz=%f,max_interval_between_messages=%f,  SKIPPING" % (avg_hz, max_interval_between_messages))
            was_error = True
        else:


            # Extract Action Schema Components
            (
                food_reference_frame, # TransformStamped
                pre_grasp_target_offset, # Vector3
                pre_grasp_initial_utensil_transform, # PoseStamped
                pre_grasp_force_threshold, # float, newtons
                approach_frame, # TransformStamped
                grasp_in_food_twist, # TransformStamped
                grasp_force_threshold, # float, newtons
                grasp_torque_threshold, # float, newston-meters
                grasp_duration, # float, secs
                extraction_out_of_food_twist, # geometry_msgs/TwistStamped
                extraction_duration, # float, secs
            ) = get_action_schema_elements(
                food_origin_frames, start_time+action_start_time, start_time+contact_time,
                start_time+extraction_time, start_time+action_end_time, force_data, forque_transform_data,
                fork_tip_frame_id=forktip_frame,
            )

            # Get the predicted fork pose over time based on the action schema
            predicted_forque_transform_data = get_predicted_forque_transform_data(
                start_time+action_start_time,
                start_time+contact_time,
                food_reference_frame, # TransformStamped
                pre_grasp_target_offset, # Vector3
                pre_grasp_initial_utensil_transform, # PoseStamped
                pre_grasp_force_threshold, # float, newtons
                approach_frame, # TransformStamped
                grasp_in_food_twist, # TwistStamped
                grasp_force_threshold, # float, newtons
                grasp_torque_threshold, # float, newston-meters
                grasp_duration, # float, secs
                extraction_out_of_food_twist, # TwistStamped
                extraction_duration, # float, secs
                fork_tip_frame_id=forktip_frame,
            )

            # Graph The Fork Pose
            graph_forktip_motion(
                forque_transform_data, forque_transform_data_raw, predicted_forque_transform_data, start_time, action_start_time, contact_time,
                extraction_time, action_end_time, image_name=image_name, out_dir=out_dir)

            action_schema_point = ActionSchemaPoint(
                start_time,
                action_start_time,
                contact_time,
                extraction_time,
                action_end_time,
                end_time - start_time,
                food_reference_frame.x,
                food_reference_frame.y,
                food_reference_frame.z,
                food_reference_frame.roll,
                food_reference_frame.pitch,
                food_reference_frame.yaw,
                pre_grasp_target_offset.x,
                pre_grasp_target_offset.y,
                pre_grasp_target_offset.z,
                pre_grasp_initial_utensil_transform.x,
                pre_grasp_initial_utensil_transform.y,
                pre_grasp_initial_utensil_transform.z,
                pre_grasp_initial_utensil_transform.roll,
                pre_grasp_initial_utensil_transform.pitch,
                pre_grasp_initial_utensil_transform.yaw,
                pre_grasp_force_threshold,
                approach_frame.roll,
                approach_frame.pitch,
                approach_frame.yaw,
                grasp_in_food_twist.linear_x,
                grasp_in_food_twist.linear_y,
                grasp_in_food_twist.linear_z,
                grasp_in_food_twist.angular_x,
                grasp_in_food_twist.angular_y,
                grasp_in_food_twist.angular_z,
                grasp_force_threshold,
                grasp_torque_threshold,
                grasp_duration,
                extraction_out_of_food_twist.linear_x,
                extraction_out_of_food_twist.linear_y,
                extraction_out_of_food_twist.linear_z,
                extraction_out_of_food_twist.angular_x,
                extraction_out_of_food_twist.angular_y,
                extraction_out_of_food_twist.angular_z,
                extraction_duration,
            )

    del first_camera_image
    del first_depth_image
    del image_without_background

    return was_error, action_schema_point

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", help="The absolute path to the folder containing the raw data")
    parser.add_argument("--verbose", help="Output verbose logs", action="store_true")
    parser.add_argument("--output_img_path", help="If set, save output images to this path (relative to the repository root)")
    parser.add_argument("--start_trial_i", help="If set, skip all trials up to this one")
    parser.add_argument("--end_trial_i", help="If set, skip all trials after to this one")
    args = parser.parse_args()

    # Get the trials
    print("Getting trials...")
    trials, total_trials = get_trials(args.raw_data_path)
    print(f"Found {total_trials} total trials across {len(trials)} subjects.")
    if args.verbose:
        pprint.pprint(trials)

    # Get the camera intrinsics
    print("Getting camera intrinsics...")
    camera_intrinsics = get_camera_intrinsics(os.path.join(args.raw_data_path, 'camera_intrinsics.csv'))

    # Create the csv writer
    print("Creating CSV writer...")
    csvwriter, csvfile = create_csv_writer()

    # Create the output image directory
    if args.output_img_path is not None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", args.output_img_path)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    # Trials to remove
    removed_trials = []
    removed_trials.append((1, "broccoli", 2)) # First image was already skewering
    removed_trials.append((5, "riceandbeans", 1)) # Moved plate at beginning

    # Iterate through and process the trials
    trial_i = -1
    for subject_num in sorted(trials.keys()):
        food_trials = trials[subject_num]
        for food_name in sorted(food_trials.keys()):
            trial_nums = food_trials[food_name]
            for trial_num in sorted(trial_nums.keys()):
                trial_data_paths = trial_nums[trial_num]
                trial_i += 1
                if args.start_trial_i is not None and trial_i < int(args.start_trial_i):
                    continue
                if args.end_trial_i is not None and trial_i > int(args.end_trial_i):
                    continue
                trial_is_removed = False
                for removed_trial in removed_trials:
                    if removed_trial[0] == subject_num and removed_trial[1] == food_name and removed_trial[2] == trial_num:
                        print(f"Skipping trial {trial_i} of {total_trials} because it was removed...")
                        trial_is_removed = True
                        break
                if trial_is_removed:
                    continue
                if args.verbose:
                    print(f"Processing trial {trial_i} of {total_trials}: {trial_data_paths}...")
                else:
                    print(f"Processing trial {trial_i} of {total_trials}...")
                removed, action_schema_point = process_trial(trial_data_paths, subject_num, food_name, trial_num, camera_intrinsics, args.verbose, out_dir)
                if removed:
                    removed_trials.append((subject_num, food_name, trial_num))
                else:
                    write_action_schema_point_to_csv(csvwriter, action_schema_point, subject_num, food_name, trial_num)

    # Print the removed trials
    print(f"Removed {len(removed_trials)} trials:")
    for removed_trial in removed_trials:
        print(f"    - {removed_trial}")

    # Save the CSV
    csvfile.flush()
    os.fsync(csvfile.fileno())
    csvfile.close()