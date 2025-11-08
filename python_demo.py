import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("./vision_test_intermediate.jpg")
# show image
# plt.imshow(img)
# plt.show()

# show image in cv version
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.imwrite('image_thres1.jpg', img)
# cv.destroyAllWindows()

# resize 
img_resized = cv.resize(img, (800, int(img.shape[0]*800/img.shape[1])))

# convert to grayscale
img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# gaussian blur to remove noise
blur = cv.blur(img,(5,5))
blur_gaussian = cv.GaussianBlur(img,(51,51),0)
# blur_bialteral = cv.bilateralFilter(img, 31,200,21) 
 
plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(blur_gaussian),plt.title('Blurred Gaussian')
plt.xticks([]), plt.yticks([])
plt.show()

# === 2. Skin detection (HSV mask) ===
hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)

# broad skin tone range, adjustable for lighting
lower = np.array([0, 30, 60], dtype=np.uint8)
upper = np.array([50, 180, 255], dtype=np.uint8)
skin_mask = cv.inRange(hsv, lower, upper)

# clean mask with morphological ops
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel, iterations=2)
skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_OPEN, kernel, iterations=2)

cv.imshow("2. Skin Mask", skin_mask)
cv.waitKey(0)

# === 3. Edge detection with skin suppression ===
edges = cv.Canny(img_gray, 50, 150)
edges[skin_mask > 0] = 0  # remove edges inside skin regions

cv.imshow("3. Canny Edges (Skin Suppressed)", edges)
cv.waitKey(0)

# Canny edge detection
edges = cv.Canny(blur, threshold1=50, threshold2=150)
edges_test = cv.Canny(blur_gaussian, threshold1=50, threshold2=150)
edges_bil = cv.Canny(blur, threshold1=50, threshold2=150)

f = plt.figure(figsize=(20,6))
ax = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)
ax.imshow(blur, cmap="gray")
ax2.imshow(edges, cmap="gray")
ax3.imshow(edges_test, cmap="gray")
plt.show()

# find contours
contours, hierarchy = cv.findContours(image=edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

# draw contours on og image
image_copy = img.copy()
cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
plt.imshow(image_copy)
plt.show()

# find contour corners
candidates = []

for cnt in contours:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4 and cv.isContourConvex(approx):
        area = cv.contourArea(approx)
        if area > 2000:
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 1.3 < aspect_ratio < 2.5:
                candidates.append(approx)

overlay = img.copy()
cv.drawContours(overlay, candidates, -1, (0,255,0), 2)
plt.imshow(overlay)
plt.show()

