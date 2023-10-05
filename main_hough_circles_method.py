import numpy as np
import time as tm
import cv2

if __name__ == '__main__':
    timer_start = tm.time()
    im = cv2.imread("h1.jpg", cv2.IMREAD_GRAYSCALE)
    im = cv2.medianBlur(im, 5)       # with or without this filter, the output of this algorithm is far from satisfying
    # the built in HoughCircles transformation of OpenCV automatically applies the necessary edge detection
    cim = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT,
                                 dp=2, minDist=200, param1=0.1, param2=0.1, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(cim, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cim, (i[0], i[1]), 2, (0, 0, 255), 3)

    timer_end = tm.time()
    calc_time = timer_end - timer_start
    print(f"Calculation time: {calc_time}")
    cv2.imshow("Circles", cim)
    cv2.waitKey(0)







