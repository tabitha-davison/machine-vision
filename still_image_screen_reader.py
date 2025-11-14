from screen_detection import screen_detection
from ocr import easy_ocr
import cv2

image = cv2.imread("phone_image_example.jpg")
screen_detection(image)
screen_image = cv2.imread("saved_images/detected_screen.jpg")
text = easy_ocr(screen_image)
with open("output.txt", "w") as f:
    for line in text:
        f.write(line + "\n")
