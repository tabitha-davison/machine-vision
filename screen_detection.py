import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def screen_detection(image):
    """
    Extracts the rectangular region around a phone screen in an image. If found,
    it crops out that region, corrects the perspective on it, and saves it as a
    jpg.

    Overview:
    1. Convert image to grayscale
    2. Apply histogram equalization and bilateral filter
    3. Create mask of image, making pixels below a certain threshold black and
        above that threshold white
    4. Run Canny edge detection on the mask
    5. Run morphological closing to close edges
    6. Find contours in image
    7. Evaluate each contour based on area, aspect ratio, and tilt from vertical
    8. Apply a perspective transform using the detected rectangle corners to
        extract and deskew the screen region in the image
    9. Save the result to saved_images/detected_screen.jpg

    Args:
        image (np.ndarray): BGR image loaded through OpenCV

    Returns:
        bool: True if a screen was detected and saved. False otherwise
    """
    # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(grayscale, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Apply histogram equalization to enhance contrast and bilateral filter to reduce noise
    bilateral_filter = cv2.bilateralFilter(cv2.equalizeHist(grayscale), 9, 75, 75)
    # plt.imshow(bilateral_filter, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Apply threshold to isolate dark regions that could be the region around the phone screen
    mask = cv2.inRange(bilateral_filter, 0, 50)
    # plt.imshow(mask, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Detect edges
    edges = cv2.Canny(mask, 50, 150)
    # plt.imshow(edges, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Close gaps in edges
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # plt.imshow(edges_closed, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # Find contours
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # contour_visualization = image.copy()
    # cv2.drawContours(contour_visualization, contours, -1, (0, 255, 0), 2)
    # plt.imshow(contour_visualization, cmap="gray")
    # plt.axis("off")
    # plt.show()

    target_aspect_ratio = 1.78
    max_tilt = 20
    min_area = 3000

    best_screen = None
    min_aspect_ratio_diff = float("inf")

    # Evaluate contours to find best screen candidate
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Get smallest rectangle around a contour
        rotated_rect = cv2.minAreaRect(contour)
        potential_screen_corners = cv2.boxPoints(rotated_rect)
        potential_screen_corners = np.intp(potential_screen_corners)
        width, height = rotated_rect[1]
        angle = rotated_rect[2]

        # Orient in potrait mode as rotated rect might have oriented region wrong
        # (we assume that the phone will be held in potrait mode)
        if height >= width:
            portrait_angle = angle
            portrait_height, portrait_width = height, width
        else:
            portrait_angle = angle + 90
            portrait_height, portrait_width = width, height

        # Calculate tilt from vertical (we care about this because we assume the phone is being held vertical)
        tilt_from_vertical = portrait_angle
        if tilt_from_vertical > 90:
            tilt_from_vertical -= 180
        elif tilt_from_vertical < -90:
            tilt_from_vertical += 180

        aspect_ratio = portrait_height / portrait_width

        print(
            f"Candidate area: {area:.0f}, AR: {aspect_ratio:.2f}, Tilt from vertical: {tilt_from_vertical:.1f}°"
        )
        # Save contour closest to aspect ratio and with angle less than max tilt
        if 1.3 < aspect_ratio < 2.5 and abs(tilt_from_vertical) <= max_tilt:
            aspect_ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if aspect_ratio_diff < min_aspect_ratio_diff:
                min_aspect_ratio_diff = aspect_ratio_diff
                best_screen = potential_screen_corners

        # debug = image.copy()
        # cv2.drawContours(debug, [potential_screen_corners], 0, (0, 255, 255), 3)
        # cv2.putText(
        #     debug,
        #     f"AR: {aspect_ratio:.2f}, Tilt: {tilt_from_vertical:.1f}",
        #     tuple(potential_screen_corners[0]),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 0, 0),
        #     2,
        # )
        # plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()

    if best_screen is not None:
        screen_corners = best_screen.astype("float32")

        # Order the rectangle corners
        x_sorted = screen_corners[np.argsort(screen_corners[:, 0])]
        left_pts, right_pts = x_sorted[:2], x_sorted[2:]

        # Determine corners
        left_pts = left_pts[np.argsort(left_pts[:, 1])]
        top_left, bottom_left = left_pts[0], left_pts[1]
        right_pts = right_pts[np.argsort(right_pts[:, 1])]
        top_right, bottom_right = right_pts[0], right_pts[1]

        screen_corners_ordered = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype="float32"
        )

        # Compute width and height of aligned screen region
        width = int(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
        height = int(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )

        # Coordinates of the aligned screen region in the final image
        aligned_screen_coords = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        # Apply perspective transform to get aligned screen image
        perspective_transform_matrix = cv2.getPerspectiveTransform(
            screen_corners_ordered, aligned_screen_coords
        )
        screen_region_image = cv2.warpPerspective(
            image, perspective_transform_matrix, (width, height)
        )

        # Save image
        save_dir = "saved_images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "detected_screen.jpg")
        cv2.imwrite(save_path, screen_region_image)

        print(f"Saved cropped & aligned screen as '{save_path}'")
        return True
    else:
        print("No screen detected.")
        return False
