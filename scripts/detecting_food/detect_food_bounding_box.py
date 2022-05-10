import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

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
    ret, label, center = cv2.kmeans(np.float32(image_no_glare.reshape((-1, 3))), k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    print(center)
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
        print(area)
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

if __name__ == "__main__":
    base_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/expanded_action_space_study"
    out_dir = "/home/amalnanavati/workspaces/amal_noetic_ws/src/feeding_study_cleanup/scripts/detecting_food/data"
    image_names = [
        "1-chicken-1_2021-11-23-13-43-33_1637703813982",
        "1-sandwich-1_2021-11-23-14-00-49_1637704849761",
        "1-spinach-4_2021-11-23-13-42-15_1637703735874",
        "2-donutholes-1_2021-11-23-15-09-33_1637708974083",
        "2-pizza-4_2021-11-23-15-19-49_1637709589793",
        "2-sandwich-5_2021-11-23-15-08-52_1637708933050",
        "3-lettuce-4_2021-11-23-18-04-06_1637719446754",
        "3-riceandbeans-5_2021-11-23-17-43-50_1637718230462",
        "4-jello-1_2021-11-23-16-12-06_1637712726983",
        "4-mashedpotato-4_2021-11-23-16-10-04_1637712604342",
        "5-broccoli-4_2021-11-24-09-58-50_1637776731343",
        "6-doughnutholes-5_2021-11-24-12-42-28_1637786548742",
        "6-jello-4_2021-11-24-12-46-39_1637786800493",
        "6-noodles-4_2021-11-24-12-39-57_1637786397483",
        "7-bagels-1_2021-11-24-14-51-50_1637794310763",
        "7-fries-5_2021-11-24-15-10-04_1637795404741",
        "8-broccoli-5_2021-11-24-15-45-19_1637797519602",
        "9-chicken-4_2021-11-24-16-40-03_1637800803873",
    ]

    for image_name in image_names:
        print(image_name)

        # Open the image
        image = cv2.imread(os.path.join(base_dir, image_name+".jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)
        # plt.imshow(image)
        # plt.show()

        # Remove the background
        image_without_background = remove_background(image, image_name=image_name, save_process=True)
        plt.imshow(image_without_background)
        plt.savefig(os.path.join(out_dir, image_name+"_removed_background.png"))

        # Get the food bounding boxes. Broccoli needs one fewer cluster because
        # its second color makes it harder to distinguish
        bounding_ellipses = get_food_bounding_box(image_without_background, k=2 if "broc" in image_name else 3, image_name=image_name, save_process=True)
        image_with_bounding_ellipses = image.copy()
        color=(255, 0, 0)
        for ellipse in bounding_ellipses:
            cv2.ellipse(image_with_bounding_ellipses, ellipse, color, 2)
        plt.imshow(image_with_bounding_ellipses)
        plt.savefig(os.path.join(out_dir, image_name+"_bounding_box.png"))
