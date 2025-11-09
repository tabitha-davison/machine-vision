import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def screen_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(gray, cmap="gray")
    # plt.axis("off")
    # plt.show()

    blur = cv2.bilateralFilter(cv2.equalizeHist(gray), 9, 75, 75)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(blur, cmap="gray")
    # plt.axis("off")
    # plt.show()

    mask = cv2.inRange(blur, 0, 50)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(mask, cmap="gray")
    # plt.axis("off")
    # plt.show()

    edges = cv2.Canny(mask, 50, 150)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(edges, cmap="gray")
    # plt.axis("off")
    # plt.show()

    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # plt.figure(figsize=(10, 8))
    # plt.imshow(edges_closed, cmap="gray")
    # plt.axis("off")
    # plt.show()

    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_vis = image.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(contour_vis, cmap="gray")
    # plt.axis("off")
    # plt.show()

    target_ar = 1.78
    max_tilt = 30
    min_area = 3000

    best_screen = None
    min_ar_diff = float("inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rot_rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rot_rect)
        box = np.intp(box)
        w, h = rot_rect[1]
        angle = rot_rect[2]

        if h >= w:
            portrait_angle = angle
            portrait_h, portrait_w = h, w
        else:
            portrait_angle = angle + 90
            portrait_h, portrait_w = w, h

        tilt_from_vertical = portrait_angle
        if tilt_from_vertical > 90:
            tilt_from_vertical -= 180
        elif tilt_from_vertical < -90:
            tilt_from_vertical += 180

        ar = portrait_h / portrait_w

        print(
            f"Candidate area: {area:.0f}, AR: {ar:.2f}, Tilt from vertical: {tilt_from_vertical:.1f}°"
        )

        if 1.3 < ar < 2.5 and abs(tilt_from_vertical) <= max_tilt:
            ar_diff = abs(ar - target_ar)
            if ar_diff < min_ar_diff:
                min_ar_diff = ar_diff
                best_screen = box

        debug = image.copy()
        cv2.drawContours(debug, [box], 0, (0, 255, 255), 3)
        cv2.putText(
            debug,
            f"AR: {ar:.2f}, Tilt: {tilt_from_vertical:.1f}",
            tuple(box[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        # plt.figure(figsize=(10, 8))
        # plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
        # plt.title(f"Candidate rectangle, AR={ar:.2f}, Tilt={tilt_from_vertical:.1f}")
        # plt.axis("off")
        # plt.show()

    detected = image.copy()
    if best_screen is not None:
        cv2.drawContours(detected, [best_screen], 0, (0, 0, 255), 10)
        print("Detected screen rectangle (rotated):")
        print(best_screen)

        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        rect = order_points(best_screen)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        save_dir = "saved_images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "detected_screen.jpg")
        cv2.imwrite(save_path, warped)
        print(f"Saved cropped & aligned screen as '{save_path}'")
        return True
    else:
        print("No screen detected.")
        return False
