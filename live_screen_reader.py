from screen_detection import screen_detection
from glare_detection import detect_glare
from ocr import easy_ocr
import cv2
import time

# in loop, get screen then read
def live_loop():
    """
    A function that runs the live feed pipeline when this file is run as a script.
    Using OpenCV, the webcam is captured for the first person camera view, and
    every 1 second, the screen detection, glare detection, and OCR features are run if
    a phone screen is detected.

    Args:
        None:

    Returns:
        None
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Running. Press 'q' in the window to quit (or Ctrl+C if headless).")
    last = 0.0
    use_window = True  # try GUI once

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame.")
                break

            # screen and glare detection every 1 sec
            if time.time() - last >= 1.0:
                last = time.time()
                if screen_detection(frame):
                    res = detect_glare(frame)
                    print(f"Has glare: {res['has_glare']} | coverage: {res['score']:.4f}")

                    # screen image 
                    screen_image = cv2.imread("./saved_images/detected_screen.jpg")
                    text = easy_ocr(screen_image)
                    with open("output.txt", "w") as f:
                        for line in text:
                            f.write(line + "\n")

            # showing webcam feed
            if use_window:
                try:
                    cv2.imshow("Webcam (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    print("No GUI backend available; switching to headless mode.")
                    use_window = False

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        cap.release()
        if use_window:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    live_loop()
