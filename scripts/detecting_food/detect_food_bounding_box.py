import csv
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform, TwistStamped, Twist, Vector3
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import rosbag
import rospy
import time
import tf.transformations
import tf2_py
from tf2_msgs.msg import TFMessage
import traceback

def transform_to_matrix(transform_msg):
    m = tf.transformations.quaternion_matrix([
        transform_msg.rotation.x,
        transform_msg.rotation.y,
        transform_msg.rotation.z,
        transform_msg.rotation.w,
    ])
    m[0][3] = transform_msg.translation.x
    m[1][3] = transform_msg.translation.y
    m[2][3] = transform_msg.translation.z
    return m

def matrix_to_transform(m):
    transform_msg = Transform()

    q = tf.transformations.quaternion_from_matrix(m)
    transform_msg.translation.x = m[0][3]
    transform_msg.translation.y = m[1][3]
    transform_msg.translation.z = m[2][3]
    transform_msg.rotation.x = q[0]
    transform_msg.rotation.y = q[1]
    transform_msg.rotation.z = q[2]
    transform_msg.rotation.w = q[3]

    return transform_msg

def pose_to_matrix(pose_msg):
    m = tf.transformations.quaternion_matrix([
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
        pose_msg.orientation.w,
    ])
    m[0][3] = pose_msg.position.x
    m[1][3] = pose_msg.position.y
    m[2][3] = pose_msg.position.z
    return m

def matrix_to_pose(m):
    pose_msg = Pose()

    q = tf.transformations.quaternion_from_matrix(m)
    pose_msg.position.x = m[0][3]
    pose_msg.position.y = m[1][3]
    pose_msg.position.z = m[2][3]
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]

    return pose_msg

def matrix_to_twist(m, duration):
    twist_msg = Twist()

    rx, ry, rz = tf.transformations.euler_from_matrix(m, 'rxyz')
    twist_msg.linear.x = m[0][3]/duration
    twist_msg.linear.y = m[1][3]/duration
    twist_msg.linear.z = m[2][3]/duration
    twist_msg.angular.x = rx/duration
    twist_msg.angular.y = ry/duration
    twist_msg.angular.z = rz/duration

    return twist_msg

def quaternion_msg_to_euler(quaternion_msg):
    q = [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]
    x, y, z = tf.transformations.euler_from_quaternion(q, 'rxyz')
    return [x, y, z]

def get_blue_range(colors, blue_ratio, min_blue):
    """
    Given a list of RGB colors, gives the range of colors that are "blues",
    where "blue" is defined as a color where the B element is more than blue_ratio
    times the R and G components
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
    image_name=None, save_process=False):
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
    ret, label, center = cv2.kmeans(np.float32(image_no_glare.reshape((-1, 3))), k, criteria, attempts, cv2.KMEANS_PP_CENTERS)
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

        # plt.imshow(np.concatenate((
        #     hull_mask,
        # ), axis=1))
        # plt.show()

        # Get the proportion of pixels that are blue within the hull
        hull_size = np.count_nonzero(hull_mask == 255)
        hull_blue_size = np.count_nonzero(np.logical_and(mask == 255, hull_mask == 255))
        proportion = float(hull_blue_size) / hull_size
        # print(proportion)

        # Store the largest one that is mostly blue
        if (proportion < min_blue_proportion and (best_hull_proportion is None or proportion > best_hull_proportion)) or proportion >= min_blue_proportion:
            if best_hull_size is None or hull_size > best_hull_size:
                best_hull_size = hull_size
                best_hull = hull
                best_hull_proportion = proportion

    # Save the process if specified
    if save_process:
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
            plt.savefig(os.path.join(out_dir, image_name+"_remove_background_process.png"))
            plt.clf()

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

    # plt.imshow(np.concatenate((
    #     retval,
    #     np.repeat(best_hull_outline_mask.reshape((best_hull_outline_mask.shape[0], best_hull_outline_mask.shape[1], 1)), 3, axis=2),
    #     retval_blurred,
    # ), axis=1))
    # plt.show()

    return retval_blurred

def get_food_bounding_box(image,
    glare_lo=np.array([2, 110, 185]),
    glare_hi=np.array([255, 255, 255]),
    k=2,
    blue_ratio=1.75, min_blue=85,
    shrink_border_by=5,
    min_area=500, max_area=35000,
    image_name=None, save_process=False):
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
    ret, label, center = cv2.kmeans(np.float32(image_no_glare.reshape((-1, 3))), k, criteria, attempts, cv2.KMEANS_PP_CENTERS)
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
    # differently. Hence, we set the contours of the mask to black (thereby
    # making every shape slightly smaller)
    contours = cv2.findContours(255-mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else countous[1]
    for i, c in enumerate(contours):
        cv2.drawContours(mask, contours, i, color=0, thickness=shrink_border_by)

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

    if save_process:
        result = image.copy()
        for i in minEllipse:
            color = (255, 0, 0)
            cv2.drawContours(result, contours, i, color)
            # ellipse
            if c.shape[0] > 5:
                cv2.ellipse(result, minEllipse[i], color, 2)
            # # rotated rectangle
            # box = cv2.boxPoints(minRect[i])
            # box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
            # cv2.drawContours(result, [box], 0, color)
        plt.imshow(np.concatenate((
            image,
            image_no_glare,
            simplified_image,
            np.repeat(mask.reshape((mask.shape[0], mask.shape[1], 1)), 3, axis=2),
            result,
        ), axis=1))
        if image_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_dir, image_name+"_bounding_box_process.png"))
            plt.clf()

    return list(minEllipse.values())

def get_food_bounding_box_edge_detection(image,
    glare_lo=np.array([2, 110, 185]),
    glare_hi=np.array([255, 255, 255]),
    k=2,
    blue_ratio=1.75, min_blue=85,
    shrink_border_by=5,
    min_area=500, max_area=35000,
    image_name=None, save_process=False):
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

    edges = cv2.Canny(image_no_glare,100,200)

    plt.imshow(np.concatenate((
        image,
        image_no_glare,
        np.repeat(edges.reshape((edges.shape[0], edges.shape[1], 1)), 3, axis=2),
    ), axis=1))
    if image_name is None:
        plt.show()
    else:
        plt.savefig(os.path.join(out_dir, image_name+"_bounding_box_process.png"))
        plt.clf()

def fit_table(gray, depth, y_min=60,
    _hough_accum=1.5, _hough_min_dist=100, _hough_param1=100, _hough_param2=70,
    _hough_min=110, _hough_max=140, _table_buffer=50, image_name=None,
    save_process=False):
    """
        Find table plane.
        gray(image matrix): grayscale image of plate
        depth(image matrix): depth image of plate

        Returns:
        plate_uv: (u, v) of plate center
        plate_r: radius of plate in px
        height: new image matrix, height of pixel above table
    """
    gray = gray[y_min:, :]
    depth = depth[y_min:, :]

    # Detect Largest Circle (Plate)
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, _hough_accum, _hough_min_dist,
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
    u = np.linspace(0, depth.shape[1], depth.shape[1], False)
    v = np.linspace(0, depth.shape[0], depth.shape[0], False)
    U, V = np.meshgrid(u, v)
    table = a*U + b*V + c
    table = table.astype("uint16")

    # New Height
    height = table - np.clip(depth, None, table)
    height[np.where(depth == 0)] = 0

    # Noise removal
    kernel = np.ones((2,2), np.uint8)
    height = cv2.morphologyEx(height, cv2.MORPH_OPEN, kernel, iterations = 2)
    height = cv2.morphologyEx(height, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Debugging
    if save_process:
        plt.imshow(np.concatenate((
            gray.astype('float32')/255.0*depth.max(),
            plate_mask*depth.max(),
            depth,
            depth_mask.astype("uint16"),
            table,
            height,
        ), axis=1), cmap='gray')
        if image_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_dir, image_name+"_table_height_process.png"))
            plt.clf()

    return (plate_uv[0], plate_uv[1]+y_min), plate_r, table

def getInitialStaticTransforms(start_time, fork_tip_frame_id):
    """
    Return the static transforms from ForqueBody to fork_tip and from CameraBody to
    camera_link
    """
    start_time = rospy.Time(start_time)

    # retval = TFMessage()

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

    # retval.transforms = [fork_transform, camera_transform]

    return [fork_transform, camera_transform]#retval

def get_food_origin_frames(bounding_ellipses, table, first_camera_image_header,
    depth_image, camera_info, tf_buffer, parent_frame):
    """
    Converts detected bounding ellipses of food items in (u,v) space to a 6D
    transform relative to a parent_frame
    """
    food_frame_transforms = []
    ellipsoid_radii = []

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    # Add time to the lookup time so there is time for the first transforms
    # for parent_frame to be published
    try:
        # lookup_future_duration = 2.5
        parent_to_camera_msg = tf_buffer.lookup_transform_core(parent_frame,
            first_camera_image_header.frame_id,
            # first_camera_image_header.stamp+rospy.Duration(lookup_future_duration),
            rospy.Time(0),
        )
        parent_to_camera_matrix = transform_to_matrix(parent_to_camera_msg.transform)
    except Exception:
        print(traceback.format_exc())
        print("Setting parent frame to camera optical frame")
        parent_to_camera_matrix = np.identity(4)
        parent_frame = first_camera_image_header.frame_id
    # print("Parent to Camera", parent_to_camera_msg)

    # Cameras are z-forward, x-right
    for i in range(len(bounding_ellipses)):
        ellipse = bounding_ellipses[i]
        center, size, angle = ellipse# (u, v) dimensions
        angle = angle*np.pi/180.0 # convert to radians
        z = table[center[1], center[0]] / 1000.0 # height is in mm
        food_depth = depth_image[center[1], center[0]] / 1000.0 # height is in mm

        # Get the origin and size of the ellipse
        ellipse_cx = (center[0] - cx) * z / fx
        ellipse_cy = (center[1] - cy) * z / fy
        # NOTE: I think the computation of is not necessarily correct because
        # fx and fy are difference and the ellipse is rotated. But anyway this
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
        Rx = tf.transformations.rotation_matrix(np.pi, (1,0,0))
        Rz = tf.transformations.rotation_matrix(-rotation, (0,0,1))
        camera_to_food_matrix = tf.transformations.concatenate_matrices(Rx, Rz)
        camera_to_food_matrix[0][3] = ellipse_cx
        camera_to_food_matrix[1][3] = ellipse_cy
        camera_to_food_matrix[2][3] = z

        # Chain the transforms together and put it all together
        final_transform = tf.transformations.concatenate_matrices(parent_to_camera_matrix, camera_to_food_matrix)
        final = TransformStamped()
        final.header = first_camera_image_header
        final.header.frame_id = parent_frame
        final.child_frame_id = "detected_food_%d" % i
        final.transform = matrix_to_transform(final_transform)

        food_frame_transforms.append(final)
        ellipsoid_radii.append((ellipsoid_radius_x, ellipsoid_radius_y, ellipsoid_radius_z))

    return food_frame_transforms, ellipsoid_radii

def get_deliminating_timestamps(force_data, forque_transform_data, forque_distance_to_mouth,
    stationary_duration=0.5, height_threshold=0.05,
    distance_to_mouth_threshold=0.4, distance_threshold=0.05,
    force_proportion=1.0/3,
    distance_from_min_height=0.05, time_from_liftoff=0.5,
    image_name=None, save_process=False):
    """
    force_data is a list of geometry_msgs/WrenchStamped messages with the F/T data
    forque_transform_data is a list of geometry_msgs/TransformStamped messages with the forktip pose

    Define start as the end of the latest stationary_duration sec long interval
    where the forque position is within a ball of radii distance_threshold,
    the y is height_threshold above the table, and the forque is distance_to_mouth_threshold
    from the mouth.

    Define contact time to be the first time the exerted force was
    >= force_proportion of the max force

    Define extraction time to be the time the fork was in its minimum point,
    within time_from_liftoff sec of the last time the fork was distance_from_min_height
    above its min height.

    Define end as the last time the fork was distance_from_min_height
    above its min height
    """
    bag_start_time = min(force_data[0].header.stamp.to_sec(), forque_transform_data[0].header.stamp.to_sec())

    # Get the min y of the forque
    forque_y_data = []
    forque_time_data = []
    for msg in forque_transform_data:
        forque_y_data.append(msg.transform.translation.y)
        forque_time_data.append(msg.header.stamp.to_sec() - bag_start_time)
    forque_y_data = np.array(forque_y_data)
    forque_time_data = np.array(forque_time_data)
    min_height = forque_y_data.min()
    points_close_to_table = forque_y_data <= min_height + distance_from_min_height
    latest_point_close_to_table_i = len(points_close_to_table) - 1 - np.argmax(points_close_to_table[::-1])
    latest_timestamp_close_to_table = forque_time_data[latest_point_close_to_table_i]

    # Get the max force time
    force_magnitude = []
    force_time_data = []
    for msg in force_data:
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        force_magnitude.append((fx**2.0 + fy**2.0 + fz**2.0)**0.5)
        force_time_data.append(msg.header.stamp.to_sec() - bag_start_time)
    force_magnitude = np.array(force_magnitude)
    force_time_data = np.array(force_time_data)
    max_force = np.max(force_magnitude)
    max_force_time = force_time_data[np.argmax(force_magnitude)]

    # Get start time
    max_distances = []
    max_distances_times = []
    action_start_time = 0.0
    min_movement = None
    for i in range(len(forque_transform_data)):
        i_time = forque_transform_data[i].header.stamp.to_sec() - bag_start_time
        if i_time >= latest_timestamp_close_to_table: break
        # Check the height condition
        if forque_transform_data[i].transform.translation.y < min_height + height_threshold:
            continue
        # Check the distance to mouth condition
        if forque_distance_to_mouth[i] <= distance_to_mouth_threshold:
            continue
        # Get a range of length stationary_duration
        success = False
        for j in range(i+1, len(forque_transform_data)):
            j_time = forque_transform_data[j].header.stamp.to_sec() - bag_start_time
            if j_time >= latest_timestamp_close_to_table: break
            # if (j_time) > 4:
            #     print("Distance to Mouth", forque_distance_to_mouth[j])
            # Check the height condition
            if forque_transform_data[j].transform.translation.y < min_height + height_threshold:
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
                    forque_transform_data[k].transform.translation.x,
                    forque_transform_data[k].transform.translation.y,
                    forque_transform_data[k].transform.translation.z,
                ])
            points = np.array(points)
            # print("Got matching range from %f to %f" % (i_time - bag_start_time, j_time - bag_start_time), points)
            distances = np.linalg.norm(points - points.mean(axis=0), axis=1)
            # print("Max Distance: ", distances.max())
            max_distance = distances.max()
            max_distances.append(max_distance)
            max_distances_times.append(j_time)
            if (max_distance > distance_threshold and (min_movement is None or max_distance < min_movement)) or max_distance <= distance_threshold:
                action_start_time = j_time
                min_movement = max_distance
    # print(max_distances, max_distances_times)

    # forque_distance = []
    # forque_distance_time = []
    # prev_x = None
    # for i in range(1,len(forque_transform_data)):
    #     prev_msg = forque_transform_data[i-1]
    #     msg = forque_transform_data[i]
    #     curr_x = msg.transform.translation.x
    #     curr_y = msg.transform.translation.y
    #     curr_z = msg.transform.translation.z
    #     curr_time = msg.header.stamp.to_sec()
    #     prev_x = prev_msg.transform.translation.x
    #     prev_y = prev_msg.transform.translation.y
    #     prev_z = prev_msg.transform.translation.z
    #     prev_time = prev_msg.header.stamp.to_sec()
    #     if curr_time - prev_time > 0:
    #         dist = ((curr_x-prev_x)**2.0+(curr_y-prev_y)**2.0+(curr_z-prev_z)**2.0)**0.5
    #         # velocity_magnitude = dist / (curr_time - prev_time)
    #         forque_distance.append(dist)
    #         forque_distance_time.append(curr_time - bag_start_time)
    # action_start_time = 0.0

    # Get the contact time
    contact_i = np.argmax(np.logical_and(force_magnitude >= force_proportion*max_force, force_time_data > action_start_time))
    contact_time = force_time_data[contact_i]

    # Get the extraction time
    points_close_to_table = np.logical_and(forque_y_data <= min_height + distance_from_min_height, forque_time_data > contact_time)
    latest_point_close_to_table_i = len(points_close_to_table) - 1 - np.argmax(points_close_to_table[::-1])
    action_end_time = forque_time_data[latest_point_close_to_table_i]
    points_close_to_liftoff = np.logical_and(
        np.logical_and(
            forque_time_data >= action_end_time - time_from_liftoff,
            points_close_to_table,
        ),
        forque_time_data > contact_time,
    )
    masked_y_data = np.where(points_close_to_liftoff, forque_y_data, np.Inf)
    extraction_i = np.argmin(masked_y_data)
    extraction_time = forque_time_data[extraction_i]

    # Debugging
    if save_process:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharex=True)
        axes[0].plot([msg.header.stamp.to_sec() - bag_start_time for msg in forque_transform_data], forque_distance_to_mouth)
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
        axes[3].axhline(min_height + distance_from_min_height, linestyle='--', c='k')
        for i in range(len(axes)):
            axes[i].axvline(action_start_time, linestyle='--', c='r')
            axes[i].axvline(contact_time, linestyle='--', c='g')
            axes[i].axvline(extraction_time, linestyle='--', c='b')
            axes[i].axvline(action_end_time, linestyle='--', c='k')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if image_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_dir, image_name+"_get_time_process.png"))
            plt.clf()

    return action_start_time, contact_time, extraction_time, action_end_time

def get_distance_to_mouth_threshold(depth, image_header, camera_info, plate_uv, plate_r, tf_buffer):
    """
    Get the distance from the left-most point of the plate to the mouth.
    """
    # Get the camera intrinsics
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    # Get the x, y, z of the left-most plate point in the camera frame
    u = plate_uv[0] - plate_r
    v = plate_uv[1]
    z = depth[v, u] / 1000.0 # depthmap is in mm
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Get the pose of the MouthBody in camera frame
    camera_to_mouth_transform = tf_buffer.lookup_transform_core(image_header.frame_id,
        "MouthBody", rospy.Time(0))

    # Compute the distance between the plate and the mouth
    distance = (
        (camera_to_mouth_transform.transform.translation.x - x)**2.0 +
        (camera_to_mouth_transform.transform.translation.y - y)**2.0 +
        (camera_to_mouth_transform.transform.translation.z - z)**2.0
    )**0.5

    return distance

def get_action_schema_elements(
    food_origin_frames, tf_buffer, action_start_time, contact_time,
    extraction_time, end_time, force_data, forque_transform_data,
    desired_parent_frame, fork_tip_frame_id,
    grasp_ft_proportion=0.5,
):
    """
    Returns:
        - food_reference_frame (geometry_msgs/TransformStamped)
            - Origin (x,y) is the center of the food's bounding ellipse, z is
              level with the table. For rotation, +Z is out of the table and +X
              is along the major axis, facing the user.
        - pre_grasp_target_offset (geometry_msgs/Vector3)
            - Computed as the difference between the forktip (x,y,z) at
              contact_time and the food_reference_frame's origin.
        - pre_grasp_initial_utensil_transform (geometry_msgs/PoseStamped)
            - The utensil's 6D pose at action_start_time in the food_reference_frame
        - pre_grasp_force_threshold (float, newtons)
            - The force reading at the contact time. (This is based on the
              fact that the contact time is currently defined as the first time
              when the force is a proportion (33%) of the max force.)
        - grasp_in_food_twist (geometry_msgs/TwistStamped)
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
        - extraction_out_of_food_twist (geometry_msgs/TwistStamped)
            - Take the utensil's 6D pose at extraction_time, subtract if from the
              fork's  6D pose at end_time, and divide it by the duration.
        - extraction_duration (float, secs)
            - end_time - extraction_time
    """
    # First, find food_reference_frame and pre_grasp_target_offset by
    # determining which of the food_origin_frames the fork is closest to at contact.
    forque_transform_at_start = None
    forque_transform_at_contact = None
    forque_transform_at_extraction = None
    forque_transform_at_end = None
    for forque_transform in forque_transform_data:
        # start
        if forque_transform.header.stamp.to_sec() > action_start_time and forque_transform_at_start is None:
            forque_transform_at_start = forque_transform
        # contact
        if forque_transform.header.stamp.to_sec() > contact_time and forque_transform_at_contact is None:
            forque_transform_at_contact = forque_transform
            min_dist = None
            min_dist_i = None
            for i in range(len(food_origin_frames)):
                food_origin_frame = food_origin_frames[i]
                if food_origin_frame.header.frame_id != forque_transform.header.frame_id:
                    raise Exception("Forque pose and food origin are in different frames!")
                dist = (
                    (forque_transform.transform.translation.x - food_origin_frame.transform.translation.x)**2.0 +
                    (forque_transform.transform.translation.y - food_origin_frame.transform.translation.y)**2.0 +
                    (forque_transform.transform.translation.z - food_origin_frame.transform.translation.z)**2.0)**0.5
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_dist_i = i
                    food_reference_frame = food_origin_frame
                    pre_grasp_target_offset = Vector3()
                    pre_grasp_target_offset.x = forque_transform.transform.translation.x - food_origin_frame.transform.translation.x
                    pre_grasp_target_offset.y = forque_transform.transform.translation.y - food_origin_frame.transform.translation.y
                    pre_grasp_target_offset.z = forque_transform.transform.translation.z - food_origin_frame.transform.translation.z
            # print("Min distance food reference frame is %d" % min_dist_i)
            # print(food_reference_frame)
        # extraction
        if forque_transform.header.stamp.to_sec() > extraction_time and forque_transform_at_extraction is None:
            forque_transform_at_extraction = forque_transform
        # extraction
        if forque_transform.header.stamp.to_sec() > end_time and forque_transform_at_end is None:
            forque_transform_at_end = forque_transform
            break
    if forque_transform_at_end is None: # If the end time is the bag end time
        forque_transform_at_end = forque_transform_data[-1]

    # Compute pre_grasp_initial_utensil_transform
    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
    parent_to_fork_start_matrix = transform_to_matrix(forque_transform_at_start.transform)
    pre_grasp_initial_utensil_transform_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_food_matrix), parent_to_fork_start_matrix)
    pre_grasp_initial_utensil_transform = PoseStamped()
    pre_grasp_initial_utensil_transform.header = forque_transform_at_start.header
    pre_grasp_initial_utensil_transform.header.frame_id = food_reference_frame.child_frame_id
    pre_grasp_initial_utensil_transform.pose = matrix_to_pose(pre_grasp_initial_utensil_transform_matrix)

    # Get pre_grasp_force_threshold
    pre_grasp_force_threshold = None
    for msg in force_data:
        if msg.header.stamp.to_sec() > contact_time and pre_grasp_force_threshold is None:
            fx = msg.wrench.force.x
            fy = msg.wrench.force.y
            fz = msg.wrench.force.z
            pre_grasp_force_threshold = (fx**2.0 + fy**2.0 + fz**2.0)**0.5

    # Get grasp_in_food_twist and grasp_duration
    grasp_duration = extraction_time - contact_time
    # Get the fork pose at extraction_time in the frame of fork pose at contact time
    parent_to_fork_extraction_matrix = transform_to_matrix(forque_transform_at_extraction.transform)
    parent_to_fork_contact_matrix = transform_to_matrix(forque_transform_at_contact.transform)
    fork_contact_to_extraction_transform_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_fork_contact_matrix), parent_to_fork_extraction_matrix)
    grasp_in_food_twist = TwistStamped()
    grasp_in_food_twist.header = forque_transform_at_contact.header
    grasp_in_food_twist.header.frame_id = fork_tip_frame_id
    grasp_in_food_twist.twist = matrix_to_twist(fork_contact_to_extraction_transform_matrix, grasp_duration)

    # Get grasp_force_threshold, grasp_torque_threshold
    max_force, max_torque = None, None
    for msg in force_data:
        if msg.header.stamp.to_sec() >= contact_time and msg.header.stamp.to_sec() <= extraction_time:
            fx = msg.wrench.force.x
            fy = msg.wrench.force.y
            fz = msg.wrench.force.z
            force_magnitude = (fx**2.0 + fy**2.0 + fz**2.0)**0.5
            if max_force is None or force_magnitude > max_force:
                max_force = force_magnitude
            tx = msg.wrench.torque.x
            ty = msg.wrench.torque.y
            tz = msg.wrench.torque.z
            torque_magnitude = (tx**2.0 + ty**2.0 + tz**2.0)**0.5
            if max_torque is None or torque_magnitude > max_torque:
                max_torque = torque_magnitude
    grasp_force_threshold = grasp_ft_proportion*max_force
    grasp_torque_threshold = grasp_ft_proportion*max_torque

    # Get extraction_out_of_food_twist and extraction_duration
    extraction_duration = end_time - extraction_time
    # Get the fork pose at end_time in the frame of fork pose at extraction_time
    parent_to_fork_end_matrix = transform_to_matrix(forque_transform_at_end.transform)
    parent_to_fork_extraction_matrix = transform_to_matrix(forque_transform_at_extraction.transform)
    fork_extraction_to_end_transform_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_fork_extraction_matrix), parent_to_fork_end_matrix)
    extraction_out_of_food_twist = TwistStamped()
    extraction_out_of_food_twist.header = forque_transform_at_extraction.header
    extraction_out_of_food_twist.header.frame_id = fork_tip_frame_id
    extraction_out_of_food_twist.twist = matrix_to_twist(fork_extraction_to_end_transform_matrix, extraction_duration)

    return (
        food_reference_frame, # geometry_msgs/TransformStamped
        pre_grasp_target_offset, # geometry_msgs/Vector3
        pre_grasp_initial_utensil_transform, # geometry_msgs/PoseStamped
        pre_grasp_force_threshold, # float, newtons
        grasp_in_food_twist, # geometry_msgs/TwistStamped
        grasp_force_threshold, # float, newtons
        grasp_torque_threshold, # float, newston-meters
        grasp_duration, # float, secs
        extraction_out_of_food_twist, # geometry_msgs/TwistStamped
        extraction_duration, # float, secs
    )

def process_rosbag(rosbag_path,
    camera_image_topic="/camera/color/image_raw/compressed",
    camera_info_topic="/camera/color/camera_info",
    camera_depth_topic="/camera/aligned_depth_to_color/image_raw",
    tf_static_topic = "/tf_static",
    tf_topic="tf",
    forque_topic="/forque/forqueSensor",
    forque_body_topic="/vrpn_client_node/ForqueBody/pose",
    desired_parent_frame="TableBody",
    csvwriter=None,
    fork_tip_frame_id="fork_tip"
):
    """
    Processes a rosbag, extracts the necessary information

    Note that in desired_parent_frame, y has to be up. This is true for at least
    world and TableBody
    """
    # Get the topics
    topics = [
        camera_image_topic,
        camera_info_topic,
        camera_depth_topic,
        tf_static_topic,
        tf_topic,
        forque_topic,
        forque_body_topic,
    ]
    image_name = os.path.basename(rosbag_path).split(".")[0]

    # Open the bag
    bag = rosbag.Bag(rosbag_path)
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()

    # Open a tf buffer, and populate it with static transforms
    tf_buffer = tf2_py.BufferCore(rospy.Duration((end_time-start_time)*2.0)) # times 2 is to give it enough space in case some messages were backdated
    initial_static_transforms = getInitialStaticTransforms(start_time, fork_tip_frame_id)
    for transform in initial_static_transforms:
        tf_buffer.set_transform_static(transform, "default_authority")

    # Get the first camera and depth images
    first_camera_image = None
    first_camera_image_header = None
    first_camera_info = None
    first_depth_image = None
    # depth_image_base_frame = None
    force_data = []
    forque_transform_data = []
    forque_distance_to_mouth = []
    cv_bridge = CvBridge()
    for topic, msg, timestamp in bag.read_messages(topics=topics):
        # print(timestamp, end_time)
        if topic == camera_image_topic:
            if first_camera_image is None:
                first_camera_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                first_camera_image = cv2.cvtColor(first_camera_image, cv2.COLOR_BGR2RGB)
                first_camera_image_header = msg.header
        elif topic == camera_info_topic:
            if first_camera_info is None:
                first_camera_info = msg
        elif topic == camera_depth_topic:
            if first_depth_image is None:
                first_depth_image = np.array(cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'), dtype=np.float32)
                # depth_image_base_frame = msg.header.frame_id
                # plt.imshow(np.concatenate((
                #     first_depth_image,
                # ), axis=1), cmap='gray')
                # plt.show()
        elif topic == tf_static_topic:
            # print("Before setting static transform at time %f", timestamp)
            for transform in msg.transforms:
                # print(transform)
                # print(type(transform))
                # print(type(transform.transform.translation))
                # print(type(transform.transform.rotation))
                tf_buffer.set_transform_static(transform, "default_authority")
        elif topic == tf_topic:
            # print("Before setting transform at time %f", timestamp)
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "default_authority")
        elif topic == forque_topic:
            force_data.append(msg)
        elif topic == forque_body_topic:
            try:
                forktip_transform = tf_buffer.lookup_transform_core(desired_parent_frame,
                    fork_tip_frame_id, rospy.Time(0))
                mouth_to_forktip_transform = tf_buffer.lookup_transform_core("MouthBody",
                    fork_tip_frame_id, rospy.Time(0))
                distance_to_mouth = (
                    mouth_to_forktip_transform.transform.translation.x**2.0 +
                    mouth_to_forktip_transform.transform.translation.y**2.0 +
                    mouth_to_forktip_transform.transform.translation.z**2.0)**0.5
                forque_transform_data.append(forktip_transform)
                forque_distance_to_mouth.append(distance_to_mouth)
            except Exception:
                elapsed_time = timestamp.to_sec()-start_time
                if elapsed_time > 1:
                    print("ERROR? Couldn't read forktip pose %f secs after start" % (timestamp.to_sec()-start_time))
                pass
    print("Finished Reading rosbag")

    # Remove the background
    image_without_background = remove_background(first_camera_image, image_name=image_name, save_process=True)
    plt.imshow(image_without_background)
    plt.savefig(os.path.join(out_dir, image_name+"_removed_background.png"))
    print("Finished Removing background")

    # Get the food bounding boxes. Broccoli needs one more cluster because
    # its second color makes it harder to distinguish
    bounding_ellipses = get_food_bounding_box(
        image_without_background,
        k=4 if "broc" in image_name or "jello" in image_name else 3,
        image_name=image_name, save_process=True)
    # print("Ellipses: ", bounding_ellipses)
    image_with_bounding_ellipses = first_camera_image.copy()
    color=(255, 0, 0)
    for ellipse in bounding_ellipses:
        cv2.ellipse(image_with_bounding_ellipses, ellipse, color, 2)
    plt.imshow(image_with_bounding_ellipses)
    plt.savefig(os.path.join(out_dir, image_name+"_bounding_box.png"))
    print("finished finding bounding boxes")

    # Fit the table
    plate_uv, plate_r, table = fit_table(
        cv2.cvtColor(first_camera_image, cv2.COLOR_RGB2GRAY),
        first_depth_image, image_name=image_name, save_process=True)
    # print(plate_uv, plate_r, height)
    image_with_plate = first_camera_image.copy()
    color=(255, 0, 0)
    # print(image_with_plate, plate_uv, plate_r, color, 2)
    cv2.circle(image_with_plate, plate_uv, plate_r, color, 2)
    plt.imshow(image_with_plate)
    plt.savefig(os.path.join(out_dir, image_name+"_with_plate.png"))
    print("finished fitting the table")

    # # Get the distance from the mouth to the left-most point of the plate
    # distance_to_mouth_threshold = get_distance_to_mouth_threshold(first_depth_image,
    # first_camera_image_header, first_camera_info, plate_uv, plate_r, tf_buffer)
    # # print("distance_to_mouth_threshold", distance_to_mouth_threshold)

    # Get the food origin frame for each food item
    # desired_parent_frame = "TableBody"#first_camera_image_header.frame_id#
    food_origin_frames, ellipsoid_radii = get_food_origin_frames(bounding_ellipses, table,
        first_camera_image_header, first_depth_image, first_camera_info, tf_buffer, desired_parent_frame)
    # print("Food Origin Frames: ", food_origin_frames)
    print("finished getting food origin frame")

    # Get the contact time
    action_start_time, contact_time, extraction_time, action_end_time = get_deliminating_timestamps(
        force_data, forque_transform_data, forque_distance_to_mouth,
        # distance_to_mouth_threshold=distance_to_mouth_threshold,
        image_name=image_name, save_process=True)

    # Extract Action Schema Components
    (
        food_reference_frame, # geometry_msgs/TransformStamped
        pre_grasp_target_offset, # geometry_msgs/Vector3
        pre_grasp_initial_utensil_transform, # geometry_msgs/PoseStamped
        pre_grasp_force_threshold, # float, newtons
        grasp_in_food_twist, # geometry_msgs/TwistStamped
        grasp_force_threshold, # float, newtons
        grasp_torque_threshold, # float, newston-meters
        grasp_duration, # float, secs
        extraction_out_of_food_twist, # geometry_msgs/TwistStamped
        extraction_duration, # float, secs
    ) = get_action_schema_elements(
        food_origin_frames, tf_buffer, start_time+action_start_time, start_time+contact_time,
        start_time+extraction_time, start_time+action_end_time, force_data, forque_transform_data,
        desired_parent_frame=desired_parent_frame, fork_tip_frame_id=fork_tip_frame_id,
    )

    # Save the bounding box origin as a CSV
    if csvwriter is not None:
        csvwriter.writerow([
            time.time(),
            image_name,
            action_start_time,
            contact_time,
            extraction_time,
            action_end_time,
            end_time - start_time,
            food_reference_frame.transform.translation.x,
            food_reference_frame.transform.translation.y,
            food_reference_frame.transform.translation.z,
        ] + quaternion_msg_to_euler(food_reference_frame.transform.rotation) + [
            pre_grasp_target_offset.x,
            pre_grasp_target_offset.y,
            pre_grasp_target_offset.z,
            pre_grasp_initial_utensil_transform.pose.position.x,
            pre_grasp_initial_utensil_transform.pose.position.y,
            pre_grasp_initial_utensil_transform.pose.position.z,
        ] + quaternion_msg_to_euler(pre_grasp_initial_utensil_transform.pose.orientation) + [
            pre_grasp_force_threshold,
            grasp_in_food_twist.twist.linear.x,
            grasp_in_food_twist.twist.linear.y,
            grasp_in_food_twist.twist.linear.z,
            grasp_in_food_twist.twist.angular.x,
            grasp_in_food_twist.twist.angular.y,
            grasp_in_food_twist.twist.angular.z,
            grasp_force_threshold,
            grasp_torque_threshold,
            grasp_duration,
            extraction_out_of_food_twist.twist.linear.x,
            extraction_out_of_food_twist.twist.linear.y,
            extraction_out_of_food_twist.twist.linear.z,
            extraction_out_of_food_twist.twist.angular.x,
            extraction_out_of_food_twist.twist.angular.y,
            extraction_out_of_food_twist.twist.angular.z,
            extraction_duration,
        ])


if __name__ == "__main__":
    all_images = True

    if not all_images:
        # Running on a few rosbags
        continue_from = 0
        base_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study"
        out_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/src/feeding_study_cleanup/scripts/detecting_food/data"
        rosbag_names = [
            "1-sandwich-1_2021-11-23-14-00-49.bag",
            "1-spinach-4_2021-11-23-13-42-15.bag",
            "2-pizza-4_2021-11-23-15-19-49.bag",
            "2-sandwich-5_2021-11-23-15-08-52.bag",
            "3-lettuce-4_2021-11-23-18-04-06.bag",
            "3-riceandbeans-5_2021-11-23-17-43-50.bag",
            "4-jello-1_2021-11-23-16-12-06.bag",
            "4-mashedpotato-4_2021-11-23-16-10-04.bag",
            "5-broccoli-4_2021-11-24-09-58-50.bag",
            "6-doughnutholes-5_2021-11-24-12-42-28.bag",
            "6-jello-4_2021-11-24-12-46-39.bag",
            "6-noodles-4_2021-11-24-12-39-57.bag",
            "7-bagels-1_2021-11-24-14-51-50.bag",
            "7-fries-5_2021-11-24-15-10-04.bag",
            "8-broccoli-5_2021-11-24-15-45-19.bag",
            "9-chicken-4_2021-11-24-16-40-03.bag",
        ]
        rosbag_paths = [os.path.join(base_dir, rosbag_name) for rosbag_name in rosbag_names]
    else:
        # Running on all rosbags
        base_dir = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study"
        out_dir = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/extracted_data"
        continue_from = 0#511#
        rosbag_paths = []
        for root, subfolders, files in os.walk(base_dir):
            for filename in files:
                # No depth images in pilot data?
                if filename.lower().endswith(".bag") and "pilot" not in root:
                    rosbag_paths.append(os.path.join(root, filename))
        files_to_remove = [
            "1-broccoli-2_2021-11-23-13-38-30", # First image was already skewering
            "1-fries-1_2021-11-23-13-55-22", # Did not detect a TableBody
            "3-spinach-1_2021-11-23-18-06-44", # Did not detect a TableBody
            "4-noodles-1_2021-11-23-16-02-06", # Did not detect a TableBody
            "4-noodles-2_2021-11-23-16-02-37", # Did not detect a TableBody
            "5-jello-1_2021-11-24-10-12-17", # Did not detect a TableBody
            "9-jello-1_2021-11-24-16-37-33", # Did not detect a TableBody
            "9-lettuce-1_2021-11-24-16-42-16", # Did not detect a TableBody
            "9-lettuce-3_2021-11-24-16-42-45", # Did not detect a TableBody
            "5-riceandbeans-1_2021-11-24-10-02-34", # Moved plate at beginning
            "8-riceandbeans-1_2021-11-24-16-03-41", # HoughTransform could not detect a plate. Can remove that step since I do separate contouring of mat anyway.
        ]
        for file_to_remove in files_to_remove:
            for i in range(len(rosbag_paths)):
                if file_to_remove in rosbag_paths[i]:
                    rosbag_paths.pop(i)
                    break

        # continue_from = 0
        # rosbag_paths = ["/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/participant5/5-jello-2_2021-11-24-10-12-38.bag"]

    rosbag_paths = sorted(rosbag_paths)
    # print(rosbag_paths)
    # raise Exception(len(rosbag_paths))

    # Process each rosbag, save as a CSV!
    csv_filename = "action_schema_data.csv"
    csv_header = [
        "Save Timestamp",
        "Bag File Name",
        "Action Start Time",
        "Action Contact Time",
        "Action Extraction Time",
        "Action End Time",
        "Bag Duration",
        "Food Reference Frame Translation X",
        "Food Reference Frame Translation Y",
        "Food Reference Frame Translation Z",
        "Food Reference Frame Rotation X",
        "Food Reference Frame Rotation Y",
        "Food Reference Frame Rotation Z",
        "Pre-Grasp Target Offset X",
        "Pre-Grasp Target Offset Y",
        "Pre-Grasp Target Offset Z",
        "Pre-Grasp Initial Utensil Transform Translation X",
        "Pre-Grasp Initial Utensil Transform Translation Y",
        "Pre-Grasp Initial Utensil Transform Translation Z",
        "Pre-Grasp Initial Utensil Transform Rotation X",
        "Pre-Grasp Initial Utensil Transform Rotation Y",
        "Pre-Grasp Initial Utensil Transform Rotation Z",
        "Pre-Grasp Force Threshold",
        "Grasp In-Food Twist Linear X",
        "Grasp In-Food Twist Linear Y",
        "Grasp In-Food Twist Linear Z",
        "Grasp In-Food Twist Angular X",
        "Grasp In-Food Twist Angular Y",
        "Grasp In-Food Twist Angular Z",
        "Grasp Force Threshold",
        "Grasp Torque Threshold",
        "Grasp Duration",
        "Extraction Out-Of-Food Twist Linear X",
        "Extraction Out-Of-Food Twist Linear Y",
        "Extraction Out-Of-Food Twist Linear Z",
        "Extraction Out-Of-Food Twist Angular X",
        "Extraction Out-Of-Food Twist Angular Y",
        "Extraction Out-Of-Food Twist Angular Z",
        "Extraction Duration",
    ]
    # Open the CSV
    if continue_from == 0:
        mode = "w"
    else:
        mode = "a"
    csvfile = open(os.path.join(out_dir, csv_filename), mode)
    csvwriter = csv.writer(csvfile)
    if mode == "w":
        csvwriter.writerow(csv_header)
    # Process the Rosbag
    for i in range(continue_from, len(rosbag_paths)):
        rosbag_path = rosbag_paths[i]
        print(i, rosbag_path)
        process_rosbag(rosbag_path, csvwriter=csvwriter)
    # Save the CSV
    csvfile.flush()
    os.fsync(csvfile.fileno())
    csvfile.close()



    # base_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study"
    # out_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/src/feeding_study_cleanup/scripts/detecting_food/data"
    # image_names = [
    #     # "1-chicken-1_2021-11-23-13-43-33_1637703813982",
    #     # "1-sandwich-1_2021-11-23-14-00-49_1637704849761",
    #     # "1-spinach-4_2021-11-23-13-42-15_1637703735874",
    #     # "2-donutholes-1_2021-11-23-15-09-33_1637708974083",
    #     # "2-pizza-4_2021-11-23-15-19-49_1637709589793",
    #     # "2-sandwich-5_2021-11-23-15-08-52_1637708933050",
    #     # "3-lettuce-4_2021-11-23-18-04-06_1637719446754",
    #     # "3-riceandbeans-5_2021-11-23-17-43-50_1637718230462",
    #     # "4-jello-1_2021-11-23-16-12-06_1637712726983",
    #     # "4-mashedpotato-4_2021-11-23-16-10-04_1637712604342",
    #     # "5-broccoli-4_2021-11-24-09-58-50_1637776731343",
    #     # "6-doughnutholes-5_2021-11-24-12-42-28_1637786548742",
    #     # "6-jello-4_2021-11-24-12-46-39_1637786800493",
    #     # "6-noodles-4_2021-11-24-12-39-57_1637786397483",
    #     # "7-bagels-1_2021-11-24-14-51-50_1637794310763",
    #     "7-fries-5_2021-11-24-15-10-04_1637795404741",
    #     # "8-broccoli-5_2021-11-24-15-45-19_1637797519602",
    #     # "9-chicken-4_2021-11-24-16-40-03_1637800803873",
    # ]
    #
    # for image_name in image_names:
    #     print(image_name)
    #
    #     # Open the image
    #     image = cv2.imread(os.path.join(base_dir, image_name+".jpg"))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # print(image.shape)
    #     # plt.imshow(image)
    #     # plt.show()
    #
    #     # Remove the background
    #     image_without_background = remove_background(image, image_name=image_name, save_process=True)
    #     plt.imshow(image_without_background)
    #     plt.savefig(os.path.join(out_dir, image_name+"_removed_background.png"))
    #
    #     # # Get the food bounding boxes. Broccoli needs one fewer cluster because
    #     # # its second color makes it harder to distinguish
    #     # bounding_ellipses = get_food_bounding_box(image_without_background, k=2 if "broc" in image_name else 3, image_name=image_name, save_process=True)
    #     # image_with_bounding_ellipses = image.copy()
    #     # color=(255, 0, 0)
    #     # for ellipse in bounding_ellipses:
    #     #     cv2.ellipse(image_with_bounding_ellipses, ellipse, color, 2)
    #     # plt.imshow(image_with_bounding_ellipses)
    #     # plt.savefig(os.path.join(out_dir, image_name+"_bounding_box.png"))
    #
    #     bounding_ellipses = get_food_bounding_box_edge_detection(image_without_background, k=2 if "broc" in image_name else 3, image_name=None)
