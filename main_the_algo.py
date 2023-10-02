import cv2
import matplotlib.pyplot as plt
import numpy as np
import grayscale_image_processing as imp
import random as rand
import time as tm

if __name__ == '__main__':
    timer_start = tm.time()
    rand.seed(0)

    # The image
    im = imp.load_image("a1.jpg")

    # Parameters
    threshold = 50  # the minimal pixel length of objects that can be detected as circles
    d = 3   # (2*d) is the minimal distance (as objects in a list) between points when trying to fit circles

    low_sensitivity = "1"  # low_sensitivity == "1" -> more chance to "miss" an arc, but will be less "false positives"

    omit_extreme_circles = "1"    # this option deletes small circles (this is relevant only if low_sensitivity == "1")
    small_circle_param = 20     # denominator for the omitting option (see below)
    big_circle_param = 4        # denominator for the omitting option (see below)
    statistical_samples = 30    # the number of samples for the approximation of the radius
    error_bound_radius = 3   # maximal relative error in the radius that is allowed for low sensitivity method
    error_bound_center = 4   # similar to the one for the radius, but with the x and y coordinates of the center

    # For the plotting
    marker_thickness = 10

    # The algorithm
    kernel = imp.create_circle_kernel(10)
    original_im = im
    im = imp.detect_edges(im)
    edges_im = im
    im = imp.max_contrast(im, 200)
    small_circle_threshold = min(im.shape) / small_circle_param
    big_circle_threshold = min(im.shape) / big_circle_param
    objects = imp.find_objects(im, threshold=threshold)
    if len(objects) < 50:
        im = edges_im
        im = imp.max_contrast(im, 80)
        objects = imp.find_objects(im, threshold=threshold)
    circles = np.zeros(shape=im.shape)
    im_right_before_stat_algo = im
    for index in range(len(objects)):
        a = np.zeros(shape=im.shape)
        for item in objects[index]:
            a[item[0], item[1]] = 255
        a = imp.apply_filter(a, kernel)
        if np.sum(a) > (255 * np.sum(kernel)):
            if low_sensitivity == "1":
                radii = []
                centers = []
                for i in range(statistical_samples):
                    dots = [rand.randint(0, int(threshold / 3) - d),
                            rand.randint(int(threshold / 3) + d, 2 * int(threshold / 3) - d),
                            rand.randint(2 * int(threshold / 3) + d, threshold)]
                    params = imp.find_circle(objects[index][dots[0]][0], objects[index][dots[0]][1],
                                objects[index][dots[1]][0], objects[index][dots[1]][1],
                                objects[index][dots[2]][0], objects[index][dots[2]][1])
                    radii += [params[1]]
                    centers += [params[0]]
                avg_radius = sum(radii) / len(radii)
                if (omit_extreme_circles == "1") and (avg_radius < small_circle_threshold):
                    continue
                if (omit_extreme_circles == "1") and (big_circle_threshold < avg_radius):
                    continue
                radius_error = abs(max(radii) - min(radii))
                if (radius_error / avg_radius) > error_bound_radius:
                    continue
                # due to the way find circles is calculated, 0- axis is x, 1- axis is y, origin is top left corner
                x_error = abs(max([item[0] for item in centers]) - min([item[0] for item in centers]))
                y_error = abs(max([item[1] for item in centers]) - min([item[1] for item in centers]))
                x_avg = sum([item[0] for item in centers]) / len(centers)
                y_avg = sum([item[1] for item in centers]) / len(centers)
                if (x_error / avg_radius) > error_bound_center:
                    continue
                if (y_error / avg_radius) > error_bound_center:
                    continue
                for t in range(marker_thickness):
                    if ((x_avg - avg_radius > 0) and (x_avg + avg_radius < im.shape[0] - 1)
                            and (y_avg - avg_radius > 0) and (y_avg + avg_radius < im.shape[1] - 1)):
                        circles = imp.add_circle_on_arr(circles, (int(x_avg), int(y_avg)), avg_radius - t)
            elif low_sensitivity != "1":
                circles += a
    rgb_im = imp.create_rgb_array([circles, original_im, circles])
    timer_end = tm.time()
    calc_time = timer_end - timer_start
    print("Calculation time: " + str(calc_time))
    imp.show_some_grayscale_images([original_im, rgb_im])
    imp.show_some_grayscale_images([im_right_before_stat_algo, circles])

