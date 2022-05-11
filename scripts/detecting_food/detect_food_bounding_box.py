import csv
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Transform
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
    best_hull, best_hull_size = None, None
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
        if proportion > min_blue_proportion:
            if best_hull_size is None or hull_size > best_hull_size:
                best_hull_size = hull_size
                best_hull = hull

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
    minRect = {}
    minEllipse = {}
    for i, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        center, size, angle = rect
        area = size[0]*size[1]
        # print(area)
        if area >= min_area and area <= max_area:
            minRect[i] = rect
            if c.shape[0] > 5:
                minEllipse[i] = cv2.fitEllipse(c)

    if save_process:
        result = image.copy()
        for i in minRect:
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

def getInitialStaticTransforms(start_time):
    """
    Return the static transforms from ForqueBody to fork_tip and from CameraBody to
    camera_link
    """
    start_time = rospy.Time(start_time)

    # retval = TFMessage()

    fork_transform = TransformStamped()
    fork_transform.header.stamp = start_time
    fork_transform.header.frame_id = "ForqueBody"
    fork_transform.child_frame_id = "fork_tip"
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
    lookup_future_duration = 1.0
    parent_to_camera_msg = tf_buffer.lookup_transform_core(parent_frame,
        first_camera_image_header.frame_id, first_camera_image_header.stamp+rospy.Duration(lookup_future_duration))
    parent_to_camera_matrix = tf.transformations.quaternion_matrix([
        parent_to_camera_msg.transform.rotation.x,
        parent_to_camera_msg.transform.rotation.y,
        parent_to_camera_msg.transform.rotation.z,
        parent_to_camera_msg.transform.rotation.w,
    ])
    parent_to_camera_matrix[0][3] = parent_to_camera_msg.transform.translation.x
    parent_to_camera_matrix[1][3] = parent_to_camera_msg.transform.translation.y
    parent_to_camera_matrix[2][3] = parent_to_camera_msg.transform.translation.z
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

        # Chain the transforms together
        final_transform = tf.transformations.concatenate_matrices(parent_to_camera_matrix, camera_to_food_matrix)
        q = tf.transformations.quaternion_from_matrix(final_transform)

        # Put this all together into a Transform
        final = TransformStamped()
        final.header = parent_frame
        final.child_frame_id = "detected_food_%d" % i
        final.transform.translation.x = final_transform[0][3]
        final.transform.translation.y = final_transform[1][3]
        final.transform.translation.z = final_transform[2][3]
        final.transform.rotation.x = q[0]
        final.transform.rotation.y = q[1]
        final.transform.rotation.z = q[2]
        final.transform.rotation.w = q[3]

        food_frame_transforms.append(final)
        ellipsoid_radii.append((ellipsoid_radius_x, ellipsoid_radius_y, ellipsoid_radius_z))

    return food_frame_transforms, ellipsoid_radii

def process_rosbag(rosbag_path,
    camera_image_topic="/camera/color/image_raw/compressed",
    camera_info_topic="/camera/color/camera_info",
    camera_depth_topic="/camera/aligned_depth_to_color/image_raw",
    tf_static_topic = "/tf_static",
    tf_topic="tf",
    csvwriter=None,
):
    """
    Processes a rosbag, extracts the necessary information
    """
    # Get the topics
    topics = [
        camera_image_topic,
        camera_info_topic,
        camera_depth_topic,
        tf_static_topic,
        tf_topic,
    ]
    image_name = os.path.basename(rosbag_path).split(".")[0]

    # Open the bag
    bag = rosbag.Bag(rosbag_path)
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()

    # Open a tf buffer, and populate it with static transforms
    tf_buffer = tf2_py.BufferCore(rospy.Duration((end_time-start_time)*2.0)) # times 2 is to give it enough space in case some messages were backdated
    initial_static_transforms = getInitialStaticTransforms(start_time)
    for transform in initial_static_transforms:
        tf_buffer.set_transform_static(transform, "default_authority")

    # Get the first camera and depth images
    first_camera_image = None
    first_camera_image_header = None
    first_camera_info = None
    first_depth_image = None
    # depth_image_base_frame = None
    cv_bridge = CvBridge()
    for topic, msg, timestamp in bag.read_messages(topics=topics):
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
    # print(tf_buffer.all_frames_as_string())

    # Remove the background
    image_without_background = remove_background(first_camera_image, image_name=image_name, save_process=True)
    plt.imshow(image_without_background)
    plt.savefig(os.path.join(out_dir, image_name+"_removed_background.png"))

    # Get the food bounding boxes. Broccoli needs one fewer cluster because
    # its second color makes it harder to distinguish
    bounding_ellipses = get_food_bounding_box(image_without_background, k=2 if "broc" in image_name else 3, image_name=image_name, save_process=True)
    # print("Ellipses: ", bounding_ellipses)
    image_with_bounding_ellipses = first_camera_image.copy()
    color=(255, 0, 0)
    for ellipse in bounding_ellipses:
        cv2.ellipse(image_with_bounding_ellipses, ellipse, color, 2)
    plt.imshow(image_with_bounding_ellipses)
    plt.savefig(os.path.join(out_dir, image_name+"_bounding_box.png"))

    # Get the wall info
    plate_uv, plate_r, table = fit_table(cv2.cvtColor(first_camera_image, cv2.COLOR_RGB2GRAY), first_depth_image, image_name=image_name, save_process=True)
    # print(plate_uv, plate_r, height)
    image_with_plate = first_camera_image.copy()
    color=(255, 0, 0)
    cv2.circle(image_with_plate, plate_uv, plate_r, color, 2)
    plt.imshow(image_with_plate)
    plt.savefig(os.path.join(out_dir, image_name+"_with_plate.png"))

    # Get the food origin frame for each food item
    # TODO: implement a custom desired frame!!!
    desired_parent_frame = "TableBody"#first_camera_image_header.frame_id#
    food_origin_frames, ellipsoid_radii = get_food_origin_frames(bounding_ellipses, table,
        first_camera_image_header, first_depth_image, first_camera_info, tf_buffer, desired_parent_frame)
    # print("Food Origin Frames: ", food_origin_frames)

    # Save the bounding box origin as a CSV
    if csvwriter is not None:
        for i in range(len(bounding_ellipses)):

            ellipse = bounding_ellipses[i]
            center, size, angle = ellipse# (u, v) dimensions
            angle = angle*np.pi/180.0 # convert to radians

            csvwriter.writerow([
                time.time(),
                image_name,
                i,
                center[0],
                center[1],
                size[0],
                size[1],
                angle,
                desired_parent_frame,
                food_origin_frames[i].transform.translation.x,
                food_origin_frames[i].transform.translation.y,
                food_origin_frames[i].transform.translation.z,
                food_origin_frames[i].transform.rotation.x,
                food_origin_frames[i].transform.rotation.y,
                food_origin_frames[i].transform.rotation.z,
                food_origin_frames[i].transform.rotation.w,
                ellipsoid_radii[i][0],
                ellipsoid_radii[i][1],
                ellipsoid_radii[i][2],
            ])


if __name__ == "__main__":
    # # Running on a few rosbags
    # base_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study"
    # out_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/src/feeding_study_cleanup/scripts/detecting_food/data"
    # rosbag_names = [
    #     "1-sandwich-1_2021-11-23-14-00-49.bag",
    #     "1-spinach-4_2021-11-23-13-42-15.bag",
    #     "2-pizza-4_2021-11-23-15-19-49.bag",
    #     "2-sandwich-5_2021-11-23-15-08-52.bag",
    #     "3-lettuce-4_2021-11-23-18-04-06.bag",
    #     "3-riceandbeans-5_2021-11-23-17-43-50.bag",
    #     "4-jello-1_2021-11-23-16-12-06.bag",
    #     "4-mashedpotato-4_2021-11-23-16-10-04.bag",
    #     "5-broccoli-4_2021-11-24-09-58-50.bag",
    #     "6-doughnutholes-5_2021-11-24-12-42-28.bag",
    #     "6-jello-4_2021-11-24-12-46-39.bag",
    #     "6-noodles-4_2021-11-24-12-39-57.bag",
    #     "7-bagels-1_2021-11-24-14-51-50.bag",
    #     "7-fries-5_2021-11-24-15-10-04.bag",
    #     "8-broccoli-5_2021-11-24-15-45-19.bag",
    #     "9-chicken-4_2021-11-24-16-40-03.bag",
    # ]
    # rosbag_paths = [os.path.join(base_dir, rosbag_name) for rosbag_name in rosbag_names]

    # Running on all rosbags
    base_dir = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study"
    out_dir = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/extracted_data"
    rosbag_paths = []
    for root, subfolders, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith(".bag"):
                rosbag_paths.append(os.path.join(root, filename))
    rosbag_paths.sort()
    # print(rosbag_paths)
    # raise Exception(len(rosbag_paths))

    # Process each rosbag, save as a CSV!
    csv_filename = "action_schema_data.csv"
    csv_header = [
        "Save Timestamp",
        "Bag File Name",
        "Bounding Ellipse Index",
        "Bounding Ellipse Center U",
        "Bounding Ellipse Center V",
        "Bounding Ellipse Size W",
        "Bounding Ellipse Size H",
        "Bounding Ellipse Angle",
        "Food Origin Parent Frame",
        "Food Origin Translation X",
        "Food Origin Translation Y",
        "Food Origin Translation Z",
        "Food Origin Rotation X",
        "Food Origin Rotation Y",
        "Food Origin Rotation Z",
        "Food Origin Rotation W",
        "Food Radius X",
        "Food Radius Y",
        "Food Radius Z",
    ]
    # Open the CSV
    csvfile = open(os.path.join(out_dir, csv_filename), 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csv_header)
    # Process the Rosbag
    for i in range(len(rosbag_paths)):
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
