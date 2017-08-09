#  https://stackoverflow.com/questions/19532473/floor-plan-edge-detection-image-processing

import numpy as np
import cv2

img = cv2.imread("/home/samuel/PycharmProjects/TCC/floor.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

img_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour_area = 0
for cnt in contours:
    if cv2.contourArea(cnt) > largest_contour_area:
        largest_contour_area = cv2.contourArea(cnt)
        largest_contour = cnt

epsilon = 0.001 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

final = cv2.drawContours(img, [approx], 0, [0, 255, 0])

cv2.imwrite('image-saida.png', final)
