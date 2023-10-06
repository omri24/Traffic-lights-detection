import numpy as np
import cv2
from PIL import Image as image
from PIL import ImageOps as ops
import matplotlib.pyplot as plt
from math import sqrt


def load_image(file_name):
    """
    :param file_name: file name of an image (non grayscale are converted to greyscale)
    :return: the image as np array
    """
    im = image.open(file_name)
    im = ops.grayscale(im)
    im = np.asarray(im)
    return im

def show_grayscale_image(arr, title="Image"):
    """
    :param arr: np array that represents an grayscale image to plot
    :param title: the title of the image
    :return: None- this func just shows the image
    """
    plt.imshow(arr, cmap=plt.get_cmap("gray"))
    plt.title(title)
    plt.show()
    return None

def gray_to_bin(arr):
    """
    :param arr: np array- an image in greyscale that will be tuned to a binary matrix
    :return: the binary matrix
    """
    threshold = np.mean(arr)
    return (arr > threshold) * 255

def detect_edges(arr):
    """
    IMPORTANT! this filter will work better with lower quality images (but not too low)
    if the output is not satisfying send the image to yourself through whatsapp- it may improve the output image
    :param arr: the np array to apply the filter to
    :return: another array, after the
    """
    kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    a = cv2.filter2D(src=arr, ddepth=-1, kernel=kernel)
    return a

def apply_filter(arr, kernel):
    """
    When the kernel is partially 'outside' the image, the borders are reflected outside, without the lsat index.
    for example, for row index 0 in an image, and a 5 * 5 kernel, rows with indexes 1, 2 will be mirrored outside
    before applying the kernel
    :param arr: the im as np array, to apply the filter to
    :param kernel: the filter kernel
    :return: the image after applying the filter
    """
    a = cv2.filter2D(src=arr, ddepth=-1, kernel=kernel)
    return a

def create_circle_kernel(radius, samples=4000, balance=0):
    """
    :param radius: the radius of the circle for the filter
    :param samples: the number of samples in the circle filter
    :param balance: if balance is set to 1, the middle of the kernel will be -1 * the sum of the kernel
    :return: the kernel as np array
    """
    theta = np.linspace(0, 2 * np.pi, samples)
    x = (radius * np.cos(theta)).astype(int)
    y = (radius * np.sin(theta)).astype(int)
    a = np.zeros(shape=(2 * radius + 1, 2 * radius + 1))
    for i in range(x.shape[0]):
        a[x[i] + radius, y[i] + radius] = 1
    if balance == 1:
        sum_kernel = np.sum(a)
        a[radius, radius] = (-1) * sum_kernel
    return a

def max_contrast(arr, threshold=200):
    """
    :param arr: the input image array
    :param threshold: pixels with values below that will turn to 0, and above to 1
    :return: image array with dimensions like the input, and values 0 or 255 for each pixel
    """
    return 255 * (arr > threshold)

def show_some_grayscale_images(arr_lst):
    """
    :param arr_lst: list of image arrays to show
    :return: None
    """
    fig = plt.figure(figsize=(10, 7))
    for i in range(len(arr_lst)):
        fig.add_subplot(1, len(arr_lst), i + 1)
        plt.imshow(arr_lst[i], cmap=plt.get_cmap('gray'))
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def find_objects(arr, threshold=30):
    """
    :param arr: the image array
    :param threshold: the minimal number of pixels for a relevant object
    :return: list of lists, each list contains the points that are part of a single shape
    """
    visited = np.zeros(shape=arr.shape)
    lst_o_lst = []
    small_range = [-1, 0, 1]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if visited[i, j] == 0:
                visited[i, j] = 1
                if arr[i, j] != 0:
                    if (i > 0) and (j > 0) and (i < (arr.shape[0] - 1)) and (j < (arr.shape[1] - 1)):
                        q = [(i, j)]
                        lst = []
                        while len(q) > 0:
                            curr = q.pop(0)
                            a = curr[0]
                            b = curr[1]
                            lst += [curr]
                            for m in small_range:
                                for n in small_range:
                                    if ((a + m > 0) and (a + m < (arr.shape[0] - 1)) and
                                            (b + n > 0) and (b + n < (arr.shape[1] - 1))):
                                        if visited[a + m, b + n] == 0:
                                            visited[a + m, b + n] = 1
                                            if arr[a + m, b + n] != 0:
                                                q += [(a + m, b + n)]
                        if len(lst) > threshold:
                            lst_o_lst += [lst]
    return lst_o_lst


def find_circle(x1, y1, x2, y2, x3, y3):
    """
    source: https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
    :param x1: x of point 1
    :param y1: y of point 1
    :param x2: x of point 2
    :param y2: y of point 2
    :param x3: x of point 3
    :param y3: y of point 3
    :return: a list with the center of the circle and its radius
    """
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    sx13 = pow(x1, 2) - pow(x3, 2)
    sy13 = pow(y1, 2) - pow(y3, 2)
    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    if (2 *((y31) * (x12) - (y21) * (x13))) == 0:   # this part is to prevent division by 0
        y31 += 0.000001
        x12 += 0.00001
        y21 += 0.0001
        x13 += 0.001

    if (2 * ((x31) * (y12) - (x21) * (y13))) == 0:   # this part is to prevent division by 0
        x31 += 0.000001
        y12 += 0.00001
        x21 += 0.0001
        y13 += 0.001

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)

    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    r = round(sqrt(sqr_of_r), 5)
    return [(h, k), r]

def add_circle_on_arr(arr, center, radius, samples=1000):
    """
    :param arr: the input image array
    :param center: the center of the circle
    :param radius: the radius of the circle
    :param samples: the number of points on the circle
    :return: the original image, with the circle on it
    """
    theta = np.linspace(0, 2 * np.pi, samples)
    x = (radius * np.cos(theta)).astype(int)
    y = (radius * np.sin(theta)).astype(int)
    mat = np.zeros(shape=arr.shape)
    for i in range(x.shape[0]):
        if mat[center[0] + x[i], center[1] + y[i]] == 0:
            mat[center[0] + x[i], center[1] + y[i]] = 255
    arr_w_circle = arr + mat
    return arr_w_circle

def create_rgb_array(arr_lst):
    """
    :param arr_lst: list of 3 arrays, that together will make an RGB image
    :return: 3D array that can be interpreted as an image in RGB format
    """
    rgb_im = np.zeros(shape=(arr_lst[0].shape[0], arr_lst[0].shape[1], 3), dtype=int)
    rgb_im[:, :, 0:1] += np.reshape(arr_lst[0], (arr_lst[0].shape[0], arr_lst[0].shape[1], 1)).astype(int)
    rgb_im[:, :, 1:2] += np.reshape(arr_lst[1], (arr_lst[1].shape[0], arr_lst[1].shape[1], 1)).astype(int)
    rgb_im[:, :, 2:3] += np.reshape(arr_lst[2], (arr_lst[2].shape[0], arr_lst[2].shape[1], 1)).astype(int)
    mask1 = (rgb_im > 255) * 255             # the masks here are in order to fix values above 255
    mask2 = (rgb_im <= 255) * rgb_im         # the masks here are in order to fix values above 255
    rgb_im = mask1 + mask2
    return rgb_im