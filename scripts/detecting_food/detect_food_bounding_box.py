import copy
import csv
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform, TwistStamped, Twist, Vector3, Quaternion
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import random
import rosbag
import rospy
from scipy.ndimage import median_filter
from std_msgs.msg import Header
import time
import tf.transformations
import tf2_py
from tf2_msgs.msg import TFMessage
import traceback

EULER_ORDER = 'rxyz'

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

    rx, ry, rz = tf.transformations.euler_from_matrix(m, EULER_ORDER)
    # print(m, ry, duration)
    twist_msg.linear.x = m[0][3]/duration
    twist_msg.linear.y = m[1][3]/duration
    twist_msg.linear.z = m[2][3]/duration
    twist_msg.angular.x = rx/duration
    twist_msg.angular.y = ry/duration
    twist_msg.angular.z = rz/duration

    return twist_msg

def quaternion_msg_to_euler(quaternion_msg, prev_euler=None):
    q = [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]
    x, y, z = tf.transformations.euler_from_quaternion(q, EULER_ORDER)
    # Fix wraparound for continuity. Only for x and z since in the xyz order y is [-pi/2, pi/2]
    if prev_euler is not None:
        threshold = 0.75
        if (prev_euler[0] > 0) and (x < 0) and ((prev_euler[0] - x) > 2*np.pi*threshold):
            # print("A", prev_euler, x, y, z)
            x += np.pi*2
        elif (prev_euler[0] < 0) and (x > 0) and ((x - prev_euler[0]) > 2*np.pi*threshold):
            # print("B", prev_euler, x, y, z)
            x -= np.pi*2
        if (prev_euler[2] > 0) and (z < 0) and ((prev_euler[2] - z) > 2*np.pi*threshold):
            # print("C", prev_euler, x, y, z)
            z += np.pi*2
        elif (prev_euler[2] < 0) and (z > 0) and ((z - prev_euler[2]) > 2*np.pi*threshold):
            # print("D", prev_euler, x, y, z)
            z -= np.pi*2
    return [x, y, z]

def euler_to_quaternion_msg(euler):
    # Revert wraparound to previous standard
    euler = copy.copy(euler)
    if euler[0] > np.pi:
        euler[0] -= np.pi*2
    elif euler[0] < np.pi:
        euler[0] += np.pi*2
    if euler[2] > np.pi:
        euler[2] -= np.pi*2
    elif euler[2] < np.pi:
        euler[2] += np.pi*2
    qx, qy, qz, qw = tf.transformations.quaternion_from_euler(*euler, axes=EULER_ORDER)
    msg = Quaternion()
    msg.x = qx
    msg.y = qy
    msg.z = qz
    msg.w = qw
    return msg

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
    min_area=500, max_area=50000, #max_area=35000,#
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

    # # If two discrete food items are slightly touching, we want to treat them
    # # differently. Hence, we set the contours of the mask to black (thereby
    # # making every shape slightly smaller)
    # contours = cv2.findContours(255-mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0] if len(contours) == 2 else countous[1]
    # for i, c in enumerate(contours):
    #     cv2.drawContours(mask, contours, i, color=0, thickness=shrink_border_by)

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
            np.repeat(edges.reshape((edges.shape[0], edges.shape[1], 1)), 3, axis=2),
            result,
        ), axis=1))
        if image_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_dir, image_name+"_bounding_box_process.png"))
            plt.clf()
        del result

    del image_only_food

    return list(minEllipse.values())

# def get_food_bounding_box_edge_detection(image,
#     glare_lo=np.array([2, 110, 185]),
#     glare_hi=np.array([255, 255, 255]),
#     k=2,
#     blue_ratio=1.75, min_blue=85, max_proportion_blue=0.35,
#     shrink_border_by=5,
#     min_area=500, max_area=50000, #max_area=35000,#
#     image_name=None, save_process=False):
#     """
#     The function has several steps, that were fine-tuned for the type of images
#     captured in the Nov 21 Bite Acquisition Study.
#         1) Inpaint white / light-blue portions of the image to remove glare on
#            the plate.
#         4) Run k-means with 2-3 centers (depending on food item) to simplify
#            the colors in the image.
#         5) Mask out the blue colors in the simplified image (where "blue colors"
#            are any colors where the B value is more than blue_ratio times the R
#            and G values and greater than min_blue).
#         6) Narrow the mask slightly to separate adjacent food items.
#         7) Get the contours, fit rotated rectangles to them.
#         8) Treat every rectangle with area between min_area and max_area as a
#            food item.
#
#     Note that the default parameters were selected by manually analyzing several
#     of the images.
#     """
#
#     # Remove the glare
#     image_no_glare = cv2.inpaint(image, cv2.inRange(image, glare_lo, glare_hi), 3, cv2.INPAINT_TELEA)
#
#     # Run K-means
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     attempts = 10
#     ret, label, center = cv2.kmeans(np.float32(image_no_glare.reshape((-1, 3))), k, criteria, attempts, cv2.KMEANS_PP_CENTERS)
#     center = np.uint8(center)
#     # print(center)
#     res = center[label.flatten()]
#     simplified_image = res.reshape((image_no_glare.shape))
#
#     # Get blue_lo and blue_hi
#     blue_lo, blue_hi = get_blue_range(center, blue_ratio, min_blue)
#
#     # Mask out the blue in the image
#     mask = cv2.inRange(simplified_image, blue_lo, blue_hi)
#     image_only_food = image_no_glare.copy()
#     image_only_food[mask == 255] = (255, 255, 255)
#
#     edges = cv2.Canny(cv2.cvtColor(image_only_food, cv2.COLOR_RGB2GRAY),0,50)
#     # plt.imshow(edges)
#     # plt.show()
#
#     contours = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#     contours = contours[0] if len(contours) == 2 else countous[1]
#
#     minEllipse = {}
#     for i, c in enumerate(contours):
#         rect = cv2.minAreaRect(c)
#         center, size, angle = rect
#         area = size[0]*size[1]
#         # print(area)
#         if area >= min_area and area <= max_area:
#             # minRect[i] = rect
#             if c.shape[0] > 5:
#                 ellipse = cv2.fitEllipse(c)
#                 # Only add the ellipse if its center is in-bounds
#                 if ellipse[0][0] >= 0 and ellipse[0][0] < image.shape[1] and ellipse[0][1] >= 0 and ellipse[0][1] < image.shape[0]:
#                     # Only consider contours whose propotion blue is <= max_proportion_blue
#                     contour_mask = np.zeros(edges.shape, np.uint8)
#                     # cv2.drawContours(contour_mask, contours, i, (255), 1)
#                     cv2.ellipse(contour_mask, ellipse, (255), -1)
#                     countour_size = np.count_nonzero(contour_mask == 255)
#                     contour_blue_size = np.count_nonzero(np.logical_and(mask == 255, contour_mask == 255))
#                     proportion = float(contour_blue_size) / countour_size
#                     print(proportion)
#                     if proportion <= max_proportion_blue:
#                         minEllipse[i] = ellipse
#
#     result = image.copy()
#     for i in minEllipse:
#         color = list(np.random.random(size=2) * 256) + [0]
#         cv2.drawContours(result, contours, i, color, 2)
#     # plt.imshow(result)
#     # plt.show()
#
#     if save_process:
#         result = image.copy()
#         for i in minEllipse:
#             color = (255, 0, 0)
#             cv2.drawContours(result, contours, i, color)
#             # ellipse
#             if i in minEllipse:
#                 cv2.ellipse(result, minEllipse[i], color, 2)
#                 pass
#             # # rotated rectangle
#             # box = cv2.boxPoints(minRect[i])
#             # box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
#             # cv2.drawContours(result, [box], 0, color)
#         plt.imshow(np.concatenate((
#             image,
#             image_no_glare,
#             simplified_image,
#             image_only_food,
#             np.repeat(edges.reshape((edges.shape[0], edges.shape[1], 1)), 3, axis=2),
#             result,
#         ), axis=1))
#         if image_name is None:
#             plt.show()
#         else:
#             plt.savefig(os.path.join(out_dir, image_name+"_bounding_box_process.png"))
#             plt.clf()
#
#     return list(minEllipse.values())

def fit_table(gray_raw, depth_raw, y_min=60,
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
        table: depth image of just the table
        height: new depth image, filling in pixels below the table with the table
    """
    gray = gray_raw[y_min:, :]
    depth = depth_raw[y_min:, :]

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
    if save_process:
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
            plt.savefig(os.path.join(out_dir, image_name+"_table_height_process.png"))
            plt.clf()

    return (plate_uv[0], plate_uv[1]+y_min), plate_r, table, height

def get_initial_static_transforms(start_time, fork_tip_frame_id):
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
        Rx = tf.transformations.rotation_matrix(np.pi, (1,0,0))
        Rz = tf.transformations.rotation_matrix(-rotation, (0,0,1))
        camera_to_food_matrix = tf.transformations.concatenate_matrices(Rx, Rz)
        camera_to_food_matrix[0][3] = ellipse_cx
        camera_to_food_matrix[1][3] = ellipse_cy
        camera_to_food_matrix[2][3] = z

        # Chain the transforms together and put it all together
        final_transform = tf.transformations.concatenate_matrices(parent_to_camera_matrix, camera_to_food_matrix)
        final = TransformStamped()
        final.header = copy.deepcopy(first_camera_image_header)
        final.header.frame_id = parent_frame
        final.child_frame_id = "detected_food_%d" % i
        final.transform = matrix_to_transform(final_transform)

        food_frame_transforms.append(final)
        ellipsoid_radii.append((ellipsoid_radius_x, ellipsoid_radius_y, ellipsoid_radius_z))

    return food_frame_transforms, ellipsoid_radii

def get_forktip_to_table_distances(
    forque_transform_data, tf_buffer, first_camera_image_header, depth_image, camera_info, plate_uv, plate_r, expected_forque_hz):

    forktip_to_table_distances, forktip_to_table_timestamps = [], []

    camera_projection_matrix = np.array(camera_info.K).reshape((3, 3))

    plate_u = plate_uv[0]
    plate_v = plate_uv[1]

    for forque_transform in forque_transform_data:
        # print(first_camera_image_header, forque_transform)
        camera_to_parent_msg = tf_buffer.lookup_transform_core(first_camera_image_header.frame_id,
            forque_transform.header.frame_id,
            # first_camera_image_header.stamp+rospy.Duration(lookup_future_duration),
            rospy.Time(0),
        )
        camera_to_parent_matrix = transform_to_matrix(camera_to_parent_msg.transform)
        parent_to_forque_matrix = transform_to_matrix(forque_transform.transform)
        camera_to_forque_matrix = tf.transformations.concatenate_matrices(camera_to_parent_matrix, parent_to_forque_matrix)


        # Get the forktip location in pixels
        [[forque_u], [forque_v], [forque_z]] = np.dot(camera_projection_matrix, camera_to_forque_matrix[0:3,3:])
        forque_u /= forque_z
        forque_v /= forque_z
        # forque_v = camera_info.height - forque_v  # OpenCv has +y going down
        # if abs(forque_transform.header.stamp.to_sec() - 1637797521.815702) < 0.1:
        #     print("Time", forque_transform.header.stamp.to_sec(), 1637797521.815702)
        #     print("camera_to_parent_matrix", camera_to_parent_matrix)
        #     print("parent_to_forque_matrix", parent_to_forque_matrix)
        #     print("camera_to_forque_matrix", camera_to_forque_matrix)
        #     print("camera_projection_matrix", camera_projection_matrix)
        #     print("forque_u", forque_u, "forque_v", forque_v, "plate_u", plate_u, "plate_v", plate_v, "plate_r", plate_r)
        #     print("Actual radii", ((forque_u - plate_u)**2 + (forque_v - plate_v)**2)**0.5, "plate_r", plate_r)

        # Check if the forktip is out of bounds of the camera
        if ((forque_u < 0) or (forque_u > camera_info.width) or (forque_v < 0) or (forque_v > camera_info.height)):
            continue

        # Check if the forktip is above the plate
        if ((forque_u - plate_u)**2 + (forque_v - plate_v)**2)**0.5 <= plate_r:
            plate_depth = depth_image[int(forque_v), int(forque_u)] / 1000.0 # convert from mm
            if plate_depth > 0: # Deal with unperceived parts of the image
                forktip_to_table_distances.append(plate_depth - camera_to_forque_matrix[2,3])
                forktip_to_table_timestamps.append(forque_transform.header.stamp.to_sec())

    # print("forktip_to_table_distances", forktip_to_table_distances, len(forktip_to_table_distances))
    # print("forktip_to_table_timestamps", forktip_to_table_timestamps, len(forktip_to_table_timestamps))
    forktip_to_table_distances = median_filter(np.array(forktip_to_table_distances), size=int(expected_forque_hz/3.0))
    forktip_to_table_timestamps = np.array(forktip_to_table_timestamps)

    return forktip_to_table_distances, forktip_to_table_timestamps

def get_deliminating_timestamps(force_data, forque_transform_data, forque_distance_to_mouth,
    forktip_to_table_distances, forktip_to_table_timestamps,
    stationary_duration=0.5, height_threshold=0.05,
    distance_to_mouth_threshold=0.35, distance_threshold=0.05,
    force_proportion=1.0/3,
    distance_from_min_height=0.07, time_from_liftoff=2.0, extraction_epsilon=0.01,
    image_name=None, save_process=False):
    """
    force_data is a list of geometry_msgs/WrenchStamped messages with the F/T data
    forque_transform_data is a list of geometry_msgs/TransformStamped messages with the forktip pose

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
    bag_start_time = min(force_data[0].header.stamp.to_sec(), forque_transform_data[0].header.stamp.to_sec())

    # Crop the data at the fork's min distance to the mouth
    forque_closest_to_mouth_i = np.argmin(forque_distance_to_mouth)
    forque_transform_data = forque_transform_data[:forque_closest_to_mouth_i+1]
    forque_distance_to_mouth = forque_distance_to_mouth[:forque_closest_to_mouth_i+1]
    forque_closest_to_mouth_time = forque_transform_data[forque_closest_to_mouth_i].header.stamp.to_sec()
    force_data_i = np.argmax((np.array([msg.header.stamp.to_sec() for msg in force_data]) > forque_closest_to_mouth_time))
    if force_data_i == 0: force_data_i = len(force_data) # in case the closest to mouth is the last time
    force_data = force_data[:force_data_i]

    # Get the min y of the forque
    forque_y_data = []
    forque_time_data = []
    for msg in forque_transform_data:
        forque_y_data.append(msg.transform.translation.y)
        forque_time_data.append(msg.header.stamp.to_sec() - bag_start_time)
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
        i_time = forque_transform_data[i].header.stamp.to_sec() - bag_start_time
        if i_time >= contact_time: #latest_timestamp_close_to_table:
            break
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
            if j_time >= contact_time: #latest_timestamp_close_to_table:
                break
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
    if save_process:
        fig, axes = plt.subplots(5, 1, figsize=(6, 25), sharex=True)
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

# def get_ols(forque_transform_data, first_timestamp, ols_start_time, ols_end_time,
#     ols_intercept_time=None, translation_only=False,
#     return_slopes=False, return_transform_at_start=False,
#     return_transform_at_end=False):
#     """
#     Treats each dimension of the fork's 6D pose as independent, and computes
#     6 linear models of time versus the respective factor.
#
#     If ols_intercept_time is set, fits a model with no intercept where the
#     msg closest to that time is the intercept. Else, includes an intercept.
#
#     if return_slopes, return the 6 slopes of the model
#     """
#     # Get the pose at ols_start_time
#     if ols_intercept_time is not None:
#         transform_at_ols_intercept_time = None
#         for msg in forque_transform_data:
#             if msg.header.stamp.to_sec() > ols_intercept_time:
#                 transform_at_ols_intercept_time = msg
#                 break
#         if transform_at_ols_intercept_time is None:
#             transform_at_ols_intercept_time = forque_transform_data[-1]
#         if not translation_only:
#             roll_at_ols_start_time, pitch_at_ols_start_time, yaw_at_ols_start_time = quaternion_msg_to_euler(transform_at_ols_intercept_time.transform.rotation)
#
#     # Break the poses up into their consistuent components
#     ts, xs, ys, zs = [], [], [], []
#     if not translation_only:
#         rolls, pitches, yaws = [], [], []
#     for msg in forque_transform_data:
#         if msg.header.stamp.to_sec() < ols_start_time or msg.header.stamp.to_sec() > ols_end_time:
#             continue
#         if ols_intercept_time is None:
#             ts.append([msg.header.stamp.to_sec() - first_timestamp, 1])
#         else:
#             ts.append([msg.header.stamp.to_sec() - ols_intercept_time])
#         if ols_intercept_time is None:
#             xs.append(msg.transform.translation.x)
#         else:
#             xs.append(msg.transform.translation.x - transform_at_ols_intercept_time.transform.translation.x)
#         if ols_intercept_time is None:
#             ys.append(msg.transform.translation.y)
#         else:
#             zs.append(msg.transform.translation.z - transform_at_ols_intercept_time.transform.translation.z)
#         if ols_intercept_time is None:
#             zs.append(msg.transform.translation.z)
#         else:
#             zs.append(msg.transform.translation.z - transform_at_ols_intercept_time.transform.translation.z)
#
#         if not translation_only:
#             roll, pitch, yaw = quaternion_msg_to_euler(msg.transform.rotation)
#             if ols_intercept_time is None:
#                 rolls.append(roll)
#             else:
#                 rolls.append(roll - roll_at_ols_start_time)
#             if ols_intercept_time is None:
#                 pitches.append(pitch)
#             else:
#                 pitches.append(pitch - pitch_at_ols_start_time)
#             if ols_intercept_time is None:
#                 yaws.append(yaw)
#             else:
#                 yaws.append(yaw - yaw_at_ols_start_time)
#     ts = np.array(ts)
#     xs = np.array(xs).reshape((-1,1))
#     ys = np.array(ys).reshape((-1,1))
#     zs = np.array(zs).reshape((-1,1))
#     if not translation_only:
#         rolls = np.array(rolls).reshape((-1,1))
#         pitches = np.array(pitches).reshape((-1,1))
#         yaws = np.array(yaws).reshape((-1,1))
#
#     model_X = ts
#     model_Y = np.concatenate([xs, ys, zs] + ([] if translation_only else [rolls, pitches, yaws]), axis=1)
#
#     # Fit the model
#     coeffs = np.linalg.lstsq(model_X, model_Y)
#
#     # Write a de-transformed prediction function
#     def predict(desired_time):
#         timestamp_transformed = np.array([[desired_time - first_timestamp, 1]]) if ols_intercept_time is None else np.array([desired_time - ols_intercept_time])
#         pred_pose = np.dot(timestamp_transformed, coeffs).reshape((-1,))
#         pred_x = pred_pose[0]
#         pred_y = pred_pose[1]
#         pred_z = pred_pose[2]
#         if not translation_only:
#             pred_roll = pred_pose[3]
#             pred_pitch = pred_pose[4]
#             pred_yaw = pred_pose[5]
#         if ols_intercept_time is not None:
#             pred_x += transform_at_ols_intercept_time.transform.translation.x
#             pred_y += transform_at_ols_intercept_time.transform.translation.y
#             pred_z += transform_at_ols_intercept_time.transform.translation.z
#             if not translation_only:
#                 pred_roll += roll_at_ols_start_time
#                 pred_pitch += pitch_at_ols_start_time
#                 pred_yaw += yaw_at_ols_start_time
#         pred_transform = TransformStamped()
#         pred_transform.header = forque_transform_data[0].header
#         pred_transform.header.stamp = ols_start_time
#         pred_transform.transform.translation.x = pred_x
#         pred_transform.transform.translation.y = pred_y
#         pred_transform.transform.translation.z = pred_z
#         pred_transform.transform.rotation = euler_to_quaternion_msg([pred_roll, pred_pitch, pred_yaw])
#
#         return pred_transform
#
#     # Extract the return values
#     retval = []
#     if return_slopes:
#         retval.append(coeffs[0,:])
#     if return_transform_at_start:
#         retval.append(predict(ols_start_time))
#     if return_transform_at_end:
#         retval.append(predict(ols_end_time))
#
#     return retval

def get_action_schema_elements(
    food_origin_frames, tf_buffer, action_start_time, contact_time,
    extraction_time, end_time, force_data, forque_transform_data,
    desired_parent_frame, fork_tip_frame_id,
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
            - Defined as pre_grasp_ft_proportion of the max torque between start_time
              and contact_time.
        - approach_frame (geometry_msgs/TransformStamped)
            - Transform from the food frame to the approach frame, which has the same
              origin and is oriented where +x points away from the fork at the
              pre-grasp initial transform.
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
    forque_transform_at_beginning_of_contact_window = None
    forque_transform_at_contact = None
    forque_transform_at_extraction = None
    forque_transform_at_end = None
    for forque_transform in forque_transform_data:
        # print(forque_transform.header.stamp.to_sec(), action_start_time, contact_time, extraction_time, end_time )
        # start
        if forque_transform.header.stamp.to_sec() > action_start_time and forque_transform_at_start is None:
            forque_transform_at_start = forque_transform
        # contact
        if forque_transform.header.stamp.to_sec() > contact_time - pre_grasp_initial_transform_linear_velocity_window and forque_transform_at_beginning_of_contact_window is None:
            forque_transform_at_beginning_of_contact_window = forque_transform
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
                # print(food_origin_frame, dist)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_dist_i = i
                    food_reference_frame = food_origin_frame
                    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
                    parent_to_forque_matrix = transform_to_matrix(forque_transform.transform)
                    food_offset_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_food_matrix), parent_to_forque_matrix)
                    pre_grasp_target_offset = Vector3()
                    pre_grasp_target_offset.x = food_offset_matrix[0][3]
                    pre_grasp_target_offset.y = food_offset_matrix[1][3]
                    pre_grasp_target_offset.z = food_offset_matrix[2][3]
                    # NOTE: The error with the below code is that its in the TableBody frame!
                    # pre_grasp_target_offset = Vector3()
                    # pre_grasp_target_offset.x = forque_transform.transform.translation.x - food_origin_frame.transform.translation.x
                    # pre_grasp_target_offset.y = forque_transform.transform.translation.y - food_origin_frame.transform.translation.y
                    # pre_grasp_target_offset.z = forque_transform.transform.translation.z - food_origin_frame.transform.translation.z
            # print("Min distance food reference frame is %d" % min_dist_i)
            # print(food_reference_frame)
            # print("parent_to_food_matrix", parent_to_food_matrix)
            # print("pre_grasp_target_offset", pre_grasp_target_offset)
            # print("parent_to_forque_matrix", parent_to_forque_matrix)
        # extraction
        if forque_transform.header.stamp.to_sec() > extraction_time and forque_transform_at_extraction is None:
            forque_transform_at_extraction = forque_transform
        # end
        if forque_transform.header.stamp.to_sec() > end_time and forque_transform_at_end is None:
            forque_transform_at_end = forque_transform
            break
    if forque_transform_at_end is None: # If the end time is the bag end time
        forque_transform_at_end = forque_transform_data[-1]

    # Compute pre_grasp_initial_utensil_transform, by getting the movement near contact and extrapolating that a fixed distance from the contact position
    d_position_near_contact = np.array([
        forque_transform_at_beginning_of_contact_window.transform.translation.x - forque_transform_at_contact.transform.translation.x,
        forque_transform_at_beginning_of_contact_window.transform.translation.y - forque_transform_at_contact.transform.translation.y,
        forque_transform_at_beginning_of_contact_window.transform.translation.z - forque_transform_at_contact.transform.translation.z,
    ])
    d_position_near_contact /= np.linalg.norm(d_position_near_contact)
    fork_start_transform = Transform(
        Vector3(
            d_position_near_contact[0] * pre_grasp_initial_transform_distance + forque_transform_at_contact.transform.translation.x,
            d_position_near_contact[1] * pre_grasp_initial_transform_distance + forque_transform_at_contact.transform.translation.y,
            d_position_near_contact[2] * pre_grasp_initial_transform_distance + forque_transform_at_contact.transform.translation.z,
        ),
        forque_transform_at_contact.transform.rotation,
    )
    # # Compute pre_grasp_initial_utensil_transform, by getting the transform from the initial fork position to the fork position at contact
    # fork_start_transform = copy.deepcopy(forque_transform_at_start.transform)
    # fork_start_transform.rotation = forque_transform_at_contact.rotation

    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
    parent_to_fork_start_matrix = transform_to_matrix(fork_start_transform)
    pre_grasp_initial_utensil_transform_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_food_matrix), parent_to_fork_start_matrix)
    pre_grasp_initial_utensil_transform = PoseStamped()
    pre_grasp_initial_utensil_transform.header = forque_transform_at_start.header
    pre_grasp_initial_utensil_transform.header.frame_id = food_reference_frame.child_frame_id
    pre_grasp_initial_utensil_transform.pose = matrix_to_pose(pre_grasp_initial_utensil_transform_matrix)

    # # Get pre_grasp_force_threshold
    # pre_grasp_force_threshold = None
    # for msg in force_data:
    #     if msg.header.stamp.to_sec() > contact_time and pre_grasp_force_threshold is None:
    #         fx = msg.wrench.force.x
    #         fy = msg.wrench.force.y
    #         fz = msg.wrench.force.z
    #         pre_grasp_force_threshold = (fx**2.0 + fy**2.0 + fz**2.0)**0.5
    max_force = None
    for msg in force_data:
        if msg.header.stamp.to_sec() >= action_start_time and msg.header.stamp.to_sec() <= contact_time:
            fx = msg.wrench.force.x
            fy = msg.wrench.force.y
            fz = msg.wrench.force.z
            force_magnitude = (fx**2.0 + fy**2.0 + fz**2.0)**0.5
            if max_force is None or force_magnitude > max_force:
                max_force = force_magnitude
    pre_grasp_force_threshold = pre_grasp_ft_proportion*max_force

    # Get Approach Frame
    fork_to_food_vector = pre_grasp_initial_utensil_transform.pose.position # since food is at the origin
    angle_to_rotate = np.arctan2(-fork_to_food_vector.y, -fork_to_food_vector.x) # we want to point away from the fork
    food_frame_to_approach_frame_matrix = tf.transformations.rotation_matrix(angle_to_rotate, [0,0,1])
    approach_frame = TransformStamped(
        Header(0, food_reference_frame.header.stamp, food_reference_frame.child_frame_id),
        approach_frame_id,
        matrix_to_transform(food_frame_to_approach_frame_matrix),
    )
    approach_to_parent_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(food_frame_to_approach_frame_matrix), tf.transformations.inverse_matrix(parent_to_food_matrix))

    # Get grasp_in_food_twist and grasp_duration
    grasp_duration = extraction_time - contact_time
    # Get the fork pose at extraction_time in the frame of fork pose at contact time
    parent_to_fork_contact_matrix = transform_to_matrix(forque_transform_at_contact.transform)
    approach_to_fork_contact_matrix = tf.transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_contact_matrix)
    parent_to_fork_extraction_matrix = transform_to_matrix(forque_transform_at_extraction.transform)
    # approach_to_fork_extraction_matrix = tf.transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_extraction_matrix)
    # print(approach_to_fork_contact_matrix, approach_to_fork_extraction_matrix)
    fork_contact_to_extraction_transform_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_fork_contact_matrix), parent_to_fork_extraction_matrix)
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
    grasp_in_food_twist = TwistStamped()
    grasp_in_food_twist.header = forque_transform_at_contact.header
    grasp_in_food_twist.header.frame_id = "linear_%s_angular_%s" % (approach_frame_id, fork_tip_frame_id)
    # print("GRASP")
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
    parent_to_fork_extraction_matrix = transform_to_matrix(forque_transform_at_extraction.transform)
    approach_to_fork_extraction_matrix = tf.transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_extraction_matrix)
    parent_to_fork_end_matrix = transform_to_matrix(forque_transform_at_end.transform)
    # approach_to_fork_end_matrix = tf.transformations.concatenate_matrices(approach_to_parent_matrix, parent_to_fork_end_matrix)
    fork_extraction_to_end_transform_matrix = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_fork_extraction_matrix), parent_to_fork_end_matrix)
    # Translation in appraoch frame
    fork_extraction_to_end_transform_matrix[0:3,3:] = np.dot(approach_to_fork_extraction_matrix[0:3,0:3], fork_extraction_to_end_transform_matrix[0:3,3:])
    # fork_extraction_to_end_transform_matrix[0,3] = approach_to_fork_end_matrix[0,3] - approach_to_fork_extraction_matrix[0,3]
    # fork_extraction_to_end_transform_matrix[1,3] = approach_to_fork_end_matrix[1,3] - approach_to_fork_extraction_matrix[1,3]
    # fork_extraction_to_end_transform_matrix[2,3] = approach_to_fork_end_matrix[2,3] - approach_to_fork_extraction_matrix[2,3]
    extraction_out_of_food_twist = TwistStamped()
    extraction_out_of_food_twist.header = forque_transform_at_extraction.header
    extraction_out_of_food_twist.header.frame_id = "linear_%s_angular_%s" % (approach_frame_id, fork_tip_frame_id)
    extraction_out_of_food_twist.twist = matrix_to_twist(fork_extraction_to_end_transform_matrix, extraction_duration)

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
    if t0.header.frame_id != t1.header.frame_id:
        print("ERROR? linear_average_transform have different frame_ids %s and %s" % (t0.header.frame_id, t1.header.frame_id))
    if t0.child_frame_id != t1.child_frame_id:
        print("ERROR? linear_average_transform have different child_frame_ids %s and %s" % (t0.child_frame_id, t1.child_frame_id))
    retval = TransformStamped()
    retval.header.frame_id = t0.header.frame_id
    retval.header.stamp = rospy.Time((1-alpha)*t0.header.stamp.to_sec() + alpha*t1.header.stamp.to_sec())
    retval.child_frame_id = t0.child_frame_id
    retval.transform.translation.x = (1-alpha)*t0.transform.translation.x + alpha*t1.transform.translation.x
    retval.transform.translation.y = (1-alpha)*t0.transform.translation.y + alpha*t1.transform.translation.y
    retval.transform.translation.z = (1-alpha)*t0.transform.translation.z + alpha*t1.transform.translation.z
    t0_euler = quaternion_msg_to_euler(t0.transform.rotation)
    t1_euler = quaternion_msg_to_euler(t1.transform.rotation)
    retval_euler = [
        (1-alpha)*t0_euler[0] + alpha*t1_euler[0],
        (1-alpha)*t0_euler[1] + alpha*t1_euler[1],
        (1-alpha)*t0_euler[2] + alpha*t1_euler[2],
    ]
    retval.transform.rotation = euler_to_quaternion_msg(retval_euler)

    return retval

def apply_twist(start_transform, approach_frame, food_reference_frame, twist, duration, granularity):
    """
    Let F be the frame that start_transform is in. Then, approach_frame must have
    as its parent_frame F. And then the angular velocity of the twist will be
    interpreted in the start_transform frame, but the linear velocity will be
    interpreted in the approach_frame.
    """
    retval = []
    parent_to_start_transform_matrix = transform_to_matrix(start_transform.transform)
    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
    food_to_approach_matrix = transform_to_matrix(approach_frame.transform)
    start_transform_to_approach_matrix = tf.transformations.concatenate_matrices(tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(parent_to_start_transform_matrix), parent_to_food_matrix), food_to_approach_matrix)

    for i in range(1, granularity+1):
        elapsed_time = duration*float(i)/granularity
        transform = copy.deepcopy(start_transform)
        transform.header.stamp += rospy.Duration(elapsed_time)
        # Apply the angular velocity in the start_transform frame
        elapsed_transform = tf.transformations.euler_matrix(
            twist.twist.angular.x*elapsed_time,
            twist.twist.angular.y*elapsed_time,
            twist.twist.angular.z*elapsed_time,
            EULER_ORDER)
        # # Apply the linear velocity in the start_transform frame
        # elapsed_transform[0][3] = twist.twist.linear.x*elapsed_time
        # elapsed_transform[1][3] = twist.twist.linear.y*elapsed_time
        # elapsed_transform[2][3] = twist.twist.linear.z*elapsed_time
        # Apply the linear velocity in the approach frame
        displacement = np.array([[twist.twist.linear.x], [twist.twist.linear.y], [twist.twist.linear.z]])*elapsed_time
        elapsed_transform[0:3, 3:] = np.dot(start_transform_to_approach_matrix[0:3,0:3], displacement)
        # if i == granularity:
        #     print("B: displacement in approach frame", displacement)
        #     print("B: displacement in start transform frame", elapsed_transform[0:3, 3:])
        #     print("B: transform inv", tf.transformations.inverse_matrix(start_transform_to_approach_matrix))
        #     print("B: start_transform", start_transform)
        #     print("B: approach_frame", approach_frame)
        #     print("B: food_to_approach_matrix", food_to_approach_matrix)

        final_matrix = tf.transformations.concatenate_matrices(parent_to_start_transform_matrix, elapsed_transform)
        transform.transform = matrix_to_transform(final_matrix)
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
    parent_to_food_matrix = transform_to_matrix(food_reference_frame.transform)
    food_to_forque = pose_to_matrix(pre_grasp_initial_utensil_transform.pose)
    parent_to_forque_start_matrix = tf.transformations.concatenate_matrices(parent_to_food_matrix, food_to_forque)
    start_forque_transform = TransformStamped()
    start_forque_transform.header.frame_id = food_reference_frame.header.frame_id
    start_forque_transform.header.stamp = rospy.Time(action_start_time)
    start_forque_transform.child_frame_id = fork_tip_frame_id
    start_forque_transform.transform = matrix_to_transform(parent_to_forque_start_matrix)

    food_to_food_offset = Transform()
    food_to_food_offset.translation = pre_grasp_target_offset
    food_to_food_offset.rotation.w = 1
    food_to_food_offset_matrix = transform_to_matrix(food_to_food_offset)
    # print("pre_grasp_target_offset", pre_grasp_target_offset)
    # print("food_to_food_offset_matrix", food_to_food_offset_matrix)
    # print("parent_to_food_matrix", parent_to_food_matrix)
    parent_to_food_offset_matrix = tf.transformations.concatenate_matrices(parent_to_food_matrix, food_to_food_offset_matrix)
    # print("parent_to_food_offset_matrix", parent_to_food_offset_matrix)
    parent_to_forque_end_matrix = np.copy(parent_to_forque_start_matrix)
    parent_to_forque_end_matrix[0][3] = parent_to_food_offset_matrix[0][3]
    parent_to_forque_end_matrix[1][3] = parent_to_food_offset_matrix[1][3]
    parent_to_forque_end_matrix[2][3] = parent_to_food_offset_matrix[2][3]
    end_forque_transform = TransformStamped()
    end_forque_transform.header.frame_id = food_reference_frame.header.frame_id
    end_forque_transform.header.stamp = rospy.Time(contact_time)
    end_forque_transform.child_frame_id = fork_tip_frame_id
    end_forque_transform.transform = matrix_to_transform(parent_to_forque_end_matrix)

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
    image_name=None):

    # Get actual data
    ts, xs, ys, zs, rolls, pitches, yaws = [], [], [], [], [], [], []
    prev_euler = None
    for msg in forque_transform_data:
        ts.append(msg.header.stamp.to_sec() - first_timestamp)
        xs.append(msg.transform.translation.x)
        ys.append(msg.transform.translation.y)
        zs.append(msg.transform.translation.z)
        if len(rolls) > 0:
            prev_euler = [roll, pitch, yaw]
        roll, pitch, yaw = quaternion_msg_to_euler(msg.transform.rotation, prev_euler)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

    # Get raw data
    ts_raw, xs_raw, ys_raw, zs_raw, rolls_raw, pitches_raw, yaws_raw = [], [], [], [], [], [], []
    # prev_euler = None
    for msg in forque_transform_data_raw:
        ts_raw.append(msg.header.stamp.to_sec() - first_timestamp)
        xs_raw.append(msg.transform.translation.x)
        ys_raw.append(msg.transform.translation.y)
        zs_raw.append(msg.transform.translation.z)
        # if len(rolls_raw) > 0:
        #     prev_euler = [roll_raw, pitch_raw, yaw_raw]
        roll_raw, pitch_raw, yaw_raw = quaternion_msg_to_euler(msg.transform.rotation)#, prev_euler)
        rolls_raw.append(roll_raw)
        pitches_raw.append(pitch_raw)
        yaws_raw.append(yaw_raw)

    # Get the predicted data
    pred_ts, pred_xs, pred_ys, pred_zs, pred_rolls, pred_pitches, pred_yaws = [], [], [], [], [], [], []
    prev_euler = None
    for msg in predicted_forque_transform_data:
        pred_ts.append(msg.header.stamp.to_sec() - first_timestamp)
        pred_xs.append(msg.transform.translation.x)
        pred_ys.append(msg.transform.translation.y)
        pred_zs.append(msg.transform.translation.z)
        if len(rolls) > 0:
            prev_euler = [roll, pitch, yaw]
        roll, pitch, yaw = quaternion_msg_to_euler(msg.transform.rotation, prev_euler)
        pred_rolls.append(roll)
        pred_pitches.append(pitch)
        pred_yaws.append(yaw)

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

def smoothen_transforms(transform_data_raw, expected_forque_hz):
    smoothened_transform_data = []

    # Get actual data
    xs, ys, zs, rolls, pitches, yaws = [], [], [], [], [], []
    prev_euler = None
    for msg in transform_data_raw:
        xs.append(msg.transform.translation.x)
        ys.append(msg.transform.translation.y)
        zs.append(msg.transform.translation.z)
        if len(rolls) > 0:
            prev_euler = [roll, pitch, yaw]
        roll, pitch, yaw = quaternion_msg_to_euler(msg.transform.rotation, prev_euler)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

    xs_filtered = median_filter(np.array(xs), size=int(expected_forque_hz/3.0))
    ys_filtered = median_filter(np.array(ys), size=int(expected_forque_hz/3.0))
    zs_filtered = median_filter(np.array(zs), size=int(expected_forque_hz/3.0))
    rolls_filtered = median_filter(np.array(rolls), size=int(expected_forque_hz/3.0))
    pitches_filtered = median_filter(np.array(pitches), size=int(expected_forque_hz/3.0))
    yaws_filtered = median_filter(np.array(yaws), size=int(expected_forque_hz/3.0))

    for i in range(len(transform_data_raw)):
        transform = transform_data_raw[i]
        smoothened_transform = copy.deepcopy(transform)
        smoothened_transform.transform.translation.x = xs_filtered[i]
        smoothened_transform.transform.translation.y = ys_filtered[i]
        smoothened_transform.transform.translation.z = zs_filtered[i]
        smoothened_transform.transform.rotation = euler_to_quaternion_msg([rolls_filtered[i], pitches_filtered[i], yaws_filtered[i]])
        smoothened_transform_data.append(smoothened_transform)

    return smoothened_transform_data

def rotate_ft_data(force_data_raw, forque_transform_data, first_timestamp, fork_tip_frame_id, image_name=None, save_process=False):
    forque_ts, forque_xs, forque_ys, forque_zs = [], [], [], []
    force_ts, force_xs, force_ys, force_zs = [], [], [], []

    for forque_msg in forque_transform_data:
        forque_ts.append(forque_msg.header.stamp.to_sec() - first_timestamp)
        forque_xs.append(forque_msg.transform.translation.x)
        forque_ys.append(forque_msg.transform.translation.y)
        forque_zs.append(forque_msg.transform.translation.z)

    for force_msg in force_data_raw:
        force_ts.append(force_msg.header.stamp.to_sec() - first_timestamp)
        force_xs.append(force_msg.wrench.force.x)
        force_ys.append(force_msg.wrench.force.y)
        force_zs.append(force_msg.wrench.force.z)

    rotated_force_data = {} # angle --> [xs, ys]
    angles_to_check = [2*np.pi/3, -2*np.pi/3]
    for angle in angles_to_check:
        rotated_xs = [np.cos(angle)*force_xs[i] - np.sin(angle)*force_ys[i] for i in range(len(force_xs))]
        rotated_ys = [np.sin(angle)*force_xs[i] + np.cos(angle)*force_ys[i] for i in range(len(force_xs))]
        rotated_force_data[angle] = [rotated_xs, rotated_ys]

    # Based on the debugging, 2*np.pi/3 appears to be the correct rotation.
    correct_angle = angles_to_check[0]
    force_data = []
    for i in range(len(force_data_raw)):
        force_msg_raw = force_data_raw[i]
        force_msg = copy.deepcopy(force_msg_raw)
        # force_msg.header.frame_id = fork_tip_frame_id # Removed because it is not actually in forktip frame, it is a translation of forktip frame.
        force_msg.wrench.force.x = rotated_force_data[correct_angle][0][i]
        force_msg.wrench.force.y = rotated_force_data[correct_angle][1][i]
        force_msg.wrench.torque.x = np.cos(correct_angle)*force_msg_raw.wrench.torque.x - np.sin(correct_angle)*force_msg_raw.wrench.torque.y
        force_msg.wrench.torque.y = np.sin(correct_angle)*force_msg_raw.wrench.torque.x + np.cos(correct_angle)*force_msg_raw.wrench.torque.y
        force_data.append(force_msg)

    # # # Divide by the acceleration
    # # first_diff_ts = np.diff(forque_ts)
    # # second_diff_ts = np.diff([(forque_ts[i-1] + forque_ts[i])/2 for i in range(1, len(forque_ts))])
    # # vel_xs = np.divide(np.diff(forque_xs), first_diff_ts)
    # # acc_xs = np.divide(np.diff(vel_xs), second_diff_ts)
    # # vel_ys = np.divide(np.diff(forque_ys), first_diff_ts)
    # # acc_ys = np.divide(np.diff(vel_ys), second_diff_ts)
    # # vel_zs = np.divide(np.diff(forque_zs), first_diff_ts)
    # # acc_zs = np.divide(np.diff(vel_zs), second_diff_ts)
    # acc_xs = np.gradient(np.gradient(forque_xs, forque_ts), forque_ts)
    # acc_ys = np.gradient(np.gradient(forque_ys, forque_ts), forque_ts)
    # acc_zs = np.gradient(np.gradient(forque_zs, forque_ts), forque_ts)
    # # Aligns the torque and force timestamps to divide them
    # force_div_acc_xs, force_div_acc_ys, force_div_acc_zs = [], [], []
    # forque_i = 1
    # for force_i in range(len(force_ts)):
    #     while forque_i < len(forque_ts) - 2 and forque_ts[forque_i+1] < force_ts[force_i]:
    #         forque_i += 1
    #     if forque_i == len(forque_ts) - 2 or (forque_ts[forque_i+1] == forque_ts[forque_i]):
    #         acc_x_parent_frame = acc_xs[forque_i]
    #         acc_y_parent_frame = acc_ys[forque_i]
    #         acc_z_parent_frame = acc_zs[forque_i]
    #     else:
    #         p = (force_ts[force_i] - forque_ts[forque_i]) / (forque_ts[forque_i+1] - forque_ts[forque_i])
    #         acc_x_parent_frame = acc_xs[forque_i]*(1-p) + acc_xs[forque_i+1]*p
    #         acc_y_parent_frame = acc_ys[forque_i]*(1-p) + acc_ys[forque_i+1]*p
    #         acc_z_parent_frame = acc_zs[forque_i]*(1-p) + acc_zs[forque_i+1]*p
    #     forktip_to_parent = tf.transformations.inverse_matrix(transform_to_matrix(forque_transform_data[forque_i].transform))
    #     [[acc_x, acc_y, acc_z]] = np.dot([[acc_x_parent_frame, acc_y_parent_frame, acc_z_parent_frame]], forktip_to_parent[0:3,0:3].T)
    #     print(acc_x_parent_frame, acc_x, acc_y_parent_frame, acc_y, acc_z_parent_frame, acc_z)
    #     force_div_acc_xs.append(force_xs[force_i] / acc_x)
    #     force_div_acc_ys.append(force_ys[force_i] / acc_y)
    #     force_div_acc_zs.append(force_zs[force_i] / acc_z)

    # Convert the force data to world frame
    force_xs_world, force_ys_world, force_zs_world = [], [], []
    forque_i = 0
    for force_i in range(len(force_ts)):
        while forque_i < len(forque_ts) - 2 and forque_ts[forque_i+1] < force_ts[force_i]:
            forque_i += 1
        parent_to_forktip_matrix = transform_to_matrix(forque_transform_data[forque_i].transform)
        [[force_x_world, force_y_world, force_z_world]] = np.dot([[force_xs[force_i], force_ys[force_i], force_zs[force_i]]], parent_to_forktip_matrix[0:3,0:3].T)
        force_xs_world.append(force_x_world)
        force_ys_world.append(force_y_world)
        force_zs_world.append(force_z_world)

    # Debugging
    if save_process:

        graph_data = [
            [
                [forque_ts, forque_xs, "Forktip X (m)"],
                [forque_ts, forque_ys, "Forktip Y (m)"],
                [forque_ts, forque_zs, "Forktip Z (m)"],
            ],
            [
                [force_ts, force_xs, "Force X (N), rot 0"],
                [force_ts, force_ys, "Force Y (N), rot 0"],
                [force_ts, force_zs, "Force Z (N), rot 0"],
            ],
        ]
        for angle in angles_to_check:
            graph_data.append([
                [force_ts, rotated_force_data[angle][0], "Force X (N), rot %f" % angle],
                [force_ts, rotated_force_data[angle][1], "Force Y (N), rot %f" % angle],
                [force_ts, force_zs, "Force Z (N), rot %f" % angle],
            ])

        # graph_data.append([
        #     [forque_ts, acc_xs, "Forque Acc X (m/s^2)"],#[1:len(forque_ts)-1]
        #     [forque_ts, acc_ys, "Forque Acc Y (m/s^2)"],
        #     [forque_ts, acc_zs, "Forque Acc Z (m/s^2)"],
        # ])
        #
        # graph_data.append([
        #     [force_ts, force_div_acc_xs, "Force X / acc, rot %f" % correct_angle],
        #     [force_ts, force_div_acc_ys, "Force Y / acc, rot %f" % correct_angle],
        #     [force_ts, force_div_acc_zs, "Force Z / acc, rot %f" % correct_angle],
        # ])

        graph_data.append([
            [force_ts, force_xs_world, "Force X World, rot %f" % correct_angle],
            [force_ts, force_ys_world, "Force Y World, rot %f" % correct_angle],
            [force_ts, force_zs_world, "Force Z World, rot %f" % correct_angle],
        ])

        fig, axes = plt.subplots(len(graph_data), len(graph_data[0]), figsize=(20, 20), sharex=True)
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                axes[i][j].xaxis.set_minor_locator(AutoMinorLocator())
                axes[i][j].yaxis.set_minor_locator(AutoMinorLocator())
                axes[i][j].plot(graph_data[i][j][0], graph_data[i][j][1], linestyle='-', c='k', marker='o', markersize=4)
                axes[i][j].set_xlabel("Elapsed Time (sec)")
                axes[i][j].set_ylabel(graph_data[i][j][2])
                # axes[i][j].set_ylim(-5, 5)
                axes[i][j].grid(visible=True, which='both')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if image_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_dir, image_name+"_rotate_force_torque.png"))
            plt.clf()

    return force_data

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
    fork_tip_frame_id="fork_tip",
    expected_forque_hz = 50.0,
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
    initial_static_transforms = get_initial_static_transforms(start_time, fork_tip_frame_id)
    for transform in initial_static_transforms:
        tf_buffer.set_transform_static(transform, "default_authority")

    # Get the first camera and depth images
    first_camera_image = None
    first_camera_image_header = None
    first_camera_info = None
    first_depth_image = None
    # depth_image_base_frame = None
    force_data_raw = []
    forque_transform_data_raw = []
    forque_timstamp_data = []
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
            force_data_raw.append(msg)
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
                forque_transform_data_raw.append(forktip_transform)
                forque_timstamp_data.append(forktip_transform.header.stamp.to_sec())
                forque_distance_to_mouth.append(distance_to_mouth)
            except Exception:
                elapsed_time = timestamp.to_sec()-start_time
                if elapsed_time > 1:
                    print("ERROR? Couldn't read forktip pose %f secs after start" % (timestamp.to_sec()-start_time))
                pass
    # print("force_data_raw", force_data_raw)
    # print("Finished Reading rosbag")

    # Smoothen the detected forque transforms
    forque_transform_data = smoothen_transforms(forque_transform_data_raw, expected_forque_hz)

    # Rotate the F/T readings
    force_data = rotate_ft_data(force_data_raw, forque_transform_data, start_time, fork_tip_frame_id, image_name=image_name, save_process=True)

    # Remove the background
    image_without_background = remove_background(first_camera_image, image_name=image_name, save_process=True)
    plt.imshow(image_without_background)
    plt.savefig(os.path.join(out_dir, image_name+"_removed_background.png"))
    plt.clf()
    # print("Finished Removing background")

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
    plt.clf()
    # print("finished finding bounding boxes")

    # Fit the table
    plate_uv, plate_r, table, depth_image_clipped = fit_table(
        cv2.cvtColor(first_camera_image, cv2.COLOR_RGB2GRAY),
        first_depth_image, image_name=image_name, save_process=True)
    if len(bounding_ellipses) == 0:
        print("ERROR: No bounding ellipses! Resorting to the table")
        bounding_ellipses = [[plate_uv, (plate_r, plate_r), 0]]
    # print(plate_uv, plate_r, height)
    image_with_plate = first_camera_image.copy()
    color=(255, 0, 0)
    # print(image_with_plate, plate_uv, plate_r, color, 2)
    cv2.circle(image_with_plate, plate_uv, plate_r, color, 2)
    plt.imshow(image_with_plate)
    plt.savefig(os.path.join(out_dir, image_name+"_with_plate.png"))
    plt.clf()
    # print("finished fitting the table")

    # # Get the distance from the mouth to the left-most point of the plate
    # distance_to_mouth_threshold = get_distance_to_mouth_threshold(first_depth_image,
    # first_camera_image_header, first_camera_info, plate_uv, plate_r, tf_buffer)
    # # print("distance_to_mouth_threshold", distance_to_mouth_threshold)

    # Get the food origin frame for each food item
    # desired_parent_frame = "TableBody"#first_camera_image_header.frame_id#
    food_origin_frames, ellipsoid_radii = get_food_origin_frames(bounding_ellipses, table,
        first_camera_image_header, first_depth_image, first_camera_info, tf_buffer, desired_parent_frame)
    # print("Food Origin Frames: ", food_origin_frames)
    # print("finished getting food origin frame")

    # Get the distance from the forktip to the table
    forktip_to_table_distances, forktip_to_table_timestamps = get_forktip_to_table_distances(
        forque_transform_data, tf_buffer, first_camera_image_header, depth_image_clipped,
        first_camera_info, plate_uv, plate_r, expected_forque_hz)

    was_error = False
    if len(forktip_to_table_distances) == 0:
        print("ERROR, the user never picked up the fork!")
        was_error = True
    else:

        # Get the contact time
        action_start_time, contact_time, extraction_time, action_end_time = get_deliminating_timestamps(
            force_data, forque_transform_data, forque_distance_to_mouth, forktip_to_table_distances, forktip_to_table_timestamps,
            # distance_to_mouth_threshold=distance_to_mouth_threshold,
            image_name=image_name, save_process=True)

        # Determine whether too much time was lost tracking to make this trial invalid
        forque_timstamp_data = np.array(forque_timstamp_data)
        msgs_during_action = np.logical_and((forque_timstamp_data >= start_time+action_start_time), (forque_timstamp_data <= start_time+action_end_time))
        num_msgs_during_action = np.sum(msgs_during_action)
        avg_hz = num_msgs_during_action/(action_end_time-action_start_time)
        msg_diffs = np.diff(forque_timstamp_data)[msgs_during_action[1:]]
        np.insert(msg_diffs, 0, start_time+action_start_time)
        np.insert(msg_diffs, msg_diffs.shape[0], start_time+action_end_time)
        max_interval_between_messages = np.max(msg_diffs)
        print("avg_hz", avg_hz, "max_interval_between_messages", max_interval_between_messages)
        if (avg_hz <= 0.5*expected_forque_hz) or (max_interval_between_messages >= 25/expected_forque_hz):
            print("TOO MUCH TIME LOST TRACKING, avg_hz=%f,max_interval_between_messages=%f,  SKIPPING" % (avg_hz, max_interval_between_messages))
            was_error = True
        else:


            # Extract Action Schema Components
            (
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
            ) = get_action_schema_elements(
                food_origin_frames, tf_buffer, start_time+action_start_time, start_time+contact_time,
                start_time+extraction_time, start_time+action_end_time, force_data, forque_transform_data,
                desired_parent_frame=desired_parent_frame, fork_tip_frame_id=fork_tip_frame_id,
            )

            # Ge tthe predicted fork pose over time based on the action schema
            predicted_forque_transform_data = get_predicted_forque_transform_data(
                start_time+action_start_time,
                start_time+contact_time,
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
            )

            # Graph The Fork Pose
            graph_forktip_motion(
                forque_transform_data, forque_transform_data_raw, predicted_forque_transform_data, start_time, action_start_time, contact_time,
                extraction_time, action_end_time, image_name=image_name)

            # Save the bounding box origin as a CSV
            if csvwriter is not None and not was_error:
                participant, food, trial = image_name.split("-")[:3]
                csvwriter.writerow([
                    time.time(),
                    participant,
                    food,
                    trial,
                    image_name,
                    start_time,
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
                ] + quaternion_msg_to_euler(approach_frame.transform.rotation) + [
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

    del first_camera_image
    del first_depth_image
    del image_without_background
    del image_with_bounding_ellipses
    del image_with_plate

    return was_error


if __name__ == "__main__":
    all_images = True

    if not all_images:
        # Running on a few rosbags
        continue_from = 0
        base_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study"
        out_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/src/feeding_study_cleanup/scripts/detecting_food/data"
        rosbag_names = [
            # "1-sandwich-1_2021-11-23-14-00-49.bag",
            # "1-spinach-4_2021-11-23-13-42-15.bag",
            # "2-pizza-4_2021-11-23-15-19-49.bag",
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
            "2-chickentenders-3_2021-11-23-14-58-24", # couldn't detect fork tip pose
            # "2-jello-1_2021-11-23-14-47-35", # seems to have lost tracking, don't have good extraction time, think I fixed
            # "2-fries-1_2021-11-23-15-00-36", # seems to have lost tracking, don't have good extraction time, think I fixed
            # "3-bagel-1_2021-11-23-17-46-13", # force reading has noise.
            "9-lettuce-2_2021-11-24-16-42-29", # lost like 5 sec of fork tracking, no TableBody?
            "1-mashedpotato-1_2021-11-23-13-58-07", # maybe should be removed, lost tracking for a while but might be before start time?
            "1-noodles-2_2021-11-23-14-03-19", # only 1.5 secs
            "2-2-chickentenders_2021-11-23-14-56-02", # mistrail, didn't pick up fork
        ]
        # Files with too much time lost tracking
        files_to_remove += ['1-bagel-2_2021-11-23-14-05-32', '1-bagel-3_2021-11-23-14-05-57', '1-bagel-4_2021-11-23-14-06-23', '1-broccoli-4_2021-11-23-13-39-21', '1-chicken-1_2021-11-23-13-43-33', '1-doughnuthole-2_2021-11-23-13-46-36', '1-doughnuthole-4_2021-11-23-13-47-35', '1-jello-1_2021-11-23-13-50-34', '1-jello-3_2021-11-23-13-51-42', '1-jello-4_2021-11-23-13-52-15', '1-riceandbeans-2_2021-11-23-13-53-37', '1-sandwich-1_2021-11-23-14-00-49', '1-sandwich-2_2021-11-23-14-01-15', '1-sandwich-4_2021-11-23-14-02-04', '1-spinach-3_2021-11-23-13-41-49', '1-spinach-4_2021-11-23-13-42-15', '2-bagel-2_2021-11-23-15-15-04', '2-bagel-3_2021-11-23-15-15-28', '2-bagel-4_2021-11-23-15-16-04', '2-broccoli-3_2021-11-23-15-21-02', '2-broccoli-4_2021-11-23-15-21-25', '2-broccoli-5_2021-11-23-15-21-50', '2-chickentenders-1_2021-11-23-14-55-46', '2-chickentenders-1_2021-11-23-14-56-27', '2-chickentenders-2_2021-11-23-14-57-02', '2-chickentenders-4_2021-11-23-14-59-04', '2-fries-2_2021-11-23-15-01-07', '2-fries-3_2021-11-23-15-01-33', '2-fries-4_2021-11-23-15-02-06', '2-fries-5_2021-11-23-15-02-40', '2-jello-2_2021-11-23-14-48-56', '2-lettuce-1_2021-11-23-15-16-43', '2-lettuce-2_2021-11-23-15-17-10', '2-lettuce-4_2021-11-23-15-18-04', '2-noodles-1_2021-11-23-15-12-52', '2-pizza-3_2021-11-23-15-19-25', '2-pizza-4_2021-11-23-15-19-49', '2-sandwich-1_2021-11-23-15-03-41', '2-sandwich-2_2021-11-23-15-07-27', '2-sandwich-3_2021-11-23-15-07-58', '2-spinach-3_2021-11-23-15-11-47', '3-broccoli-3_2021-11-23-17-53-48', '3-broccoli-4_2021-11-23-17-54-11', '3-pizza-1_2021-11-23-17-44-24', '3-pizza-3_2021-11-23-17-45-04', '3-riceandbeans-1_2021-11-23-17-42-16', '4-chicken-2_2021-11-23-16-15-52', '4-fries-2_2021-11-23-16-05-09', '4-jello-1_2021-11-23-16-12-06', '4-mashedpotato-1_2021-11-23-16-08-39', '4-riceandbeans-1_2021-11-23-15-46-27', '5-bagel-4_2021-11-24-10-22-50', '5-chicken-3_2021-11-24-10-05-40', '5-fries-5_2021-11-24-10-11-09', '5-lettuce-2_2021-11-24-10-15-32', '5-noodles-1_2021-11-24-09-49-23', '5-noodles-2_2021-11-24-09-50-17', '5-spinach-2_2021-11-24-09-53-30', '5-spinach-3_2021-11-24-09-54-12', '5-spinach-5_2021-11-24-09-55-30', '6-bagel-3_2021-11-24-12-52-35', '6-chicken-3_2021-11-24-13-01-45', '6-chicken-5_2021-11-24-13-02-21', '6-jello-1_2021-11-24-12-45-15', '6-jello-2_2021-11-24-12-45-42', '6-spinach-1_2021-11-24-13-02-50', '7-bagels-2_2021-11-24-14-52-13', '7-bagels-3_2021-11-24-14-52-36', '7-chicken-2_2021-11-24-14-59-09', '7-doughnutholes-2_2021-11-24-14-50-33', '7-fries-4_2021-11-24-15-09-40', '7-jello-2_2021-11-24-15-02-20', '7-jello-3_2021-11-24-15-02-48', '7-jello-5_2021-11-24-15-03-50', '7-sandwich-4_2021-11-24-14-42-51', '7-spinach-4_2021-11-24-15-05-50', '8-broccoli-5_2021-11-24-15-45-19', '8-donutholes-1_2021-11-24-15-49-56', '8-donutholes-4_2021-11-24-15-50-59', '8-fries-1_2021-11-24-15-45-51', '8-fries-2_2021-11-24-15-46-15', '8-fries-3_2021-11-24-15-46-33', '8-mashedpotatoes-1_2021-11-24-15-39-21', '8-pizza-1_2021-11-24-15-41-56', '9-broccoli-1_2021-11-24-16-44-35', '9-chicken-4_2021-11-24-16-40-03', '9-riceandbeans-4_2021-11-24-16-49-42']
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
        "Participant",
        "Food",
        "Trial",
        "Bag File Name",
        "Bag Start Time",
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
        "Approach Frame Rotation X",
        "Approach Frame Rotation Y",
        "Approach Frame Rotation Z",
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
    removed_rosbags = []
    for i in range(continue_from, len(rosbag_paths)):
        rosbag_path = rosbag_paths[i]
        print(i, rosbag_path)
        removed = process_rosbag(rosbag_path, csvwriter=csvwriter)
        plt.close('all')
        if removed:
            removed_rosbags.append(rosbag_path)
    # Save the CSV
    csvfile.flush()
    os.fsync(csvfile.fileno())
    csvfile.close()

    print(removed_rosbags)
