import cv2
import time
from glare_detection import detect_glare
from screen_detection import screen_detection

def camera_move():
    
    """
    Continuously captures webcam frames, checks once per second whether a phone
    screen is detected, and if so, runs glare detection on the frame. Prints
    whether glare is present and the glare coverage score. If a GUI backend is
    available, displays the live webcam feed and allows quitting with 'q'.

    Handles both GUI and headless environments, and stops cleanly on Ctrl+C.
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


# camera_move()
