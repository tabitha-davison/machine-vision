import cv2
import numpy as np

img = cv2.imread('vision_test_noise.jpg')

original_height, original_width = img.shape[:2]

new_width = 800
aspect_ratio = new_width / original_width
new_height = int(original_height * aspect_ratio)

img = cv2.resize(img, (new_width, new_height))

# grayscale image ------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscaled Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blur image ------------------------------
blur = cv2.bilateralFilter(gray, 9, 75, 75)  # or GaussianBlur

cv2.imshow('Blurred Image', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# detect skin and remove it from edge search ------------------------------
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
skin_mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
blur[skin_mask > 0] = 0

cv2.imshow('Skin mask on phone', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# detect edge ------------------------------------------------------------
edges = cv2.Canny(gray, 100, 200)

cv2.imshow('Edges on Phone', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphological closing to connect phone edges --------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Morphological Closed Edge on Phone', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contour ------------------------------------------------------------
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contour_copy = img.copy()

cv2.drawContours(img_contour_copy, contours, -1, (0,255,0), 3) # draw green contours

print(f'Number of contours detected: {len(contours)}')

cv2.imshow('Detected Contours', img_contour_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# filter out small sizes in contours ------------------------------------------------------------
# can change to optimize in one loop later

# filter out small buttons and noisy contours
filter_contours = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 5000: # arbitrary pixel area amount
        filter_contours.append(cnt)

print(f'Number of filtered contours: {len(filter_contours)}')

img_filter_contours = img.copy()
cv2.drawContours(img_filter_contours, filter_contours, -1, (0,255,0), 3)
cv2.imshow('Filtered Contours by Size', img_filter_contours)
cv2.waitKey(0)

# loop through contours to find the phone screen ------------------------------------------------------------
img_approx_copy = img.copy()
img_detection_copy = img.copy()
for cnt in filter_contours:
    
    print(f'Contour Perimeter: {cv2.arcLength(cnt, True) }')
    epsilon = 0.02 * cv2.arcLength(cnt, True) # get contour perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True) # get clean shape from approx

    cv2.drawContours(img_approx_copy, [approx], -1, (0,255,0), 3)
    
    for point in approx: # draw vertice points
        x, y = point[0]
        cv2.circle(img_approx_copy, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow('Approximated Contour Shape', img_approx_copy)
    cv2.waitKey(0)

    # if len(approx) > 3 and cv2.isContourConvex(approx): 
    if len(approx) > 3: # check convex polygons with at least 4 pts
        x, y, w, h = cv2.boundingRect(approx)
        # aspect = w / float(h)
        # print(f"aspect ratio: {aspect}")
        # if 0 < aspect < 1:  # check for phone aspect ratio
        # cv2.drawContours(img_detection_copy, [approx], -1, (0,255,0), 3) # draw green contours
        cv2.rectangle(img_detection_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Phone Detection', img_detection_copy)
cv2.waitKey(0)

