import cv2
import numpy as np
import imutils
from skimage.measure import compare_ssim

image1 = cv2.imread('Screenshot (20).png')
image2 = cv2.imread('Screenshot (22).png')


grayimage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
(score, diff) = compare_ssim(grayimage1, grayimage2, full=True)

diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 1)

# show the output images
cv2.imshow("Original", image1)
cv2.imshow("Modified", image2)
# cv2.imshow("Diff", diff)
# cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
