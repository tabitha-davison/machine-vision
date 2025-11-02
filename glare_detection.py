import cv2
import numpy as np

# Read the image and raise error if problem
bgr = cv2.imread("easy_ocr_result.png")
if bgr is None:
    raise FileNotFoundError("Couldn't load easy_OCR.jpeg.")

def detect_glare(
    bgr,
    win=15,
    weights=(0.45, 0.35, 0.20),  # (w_intensity, w_lowSat, w_lowLocalContrast)
    score_thresh=0.65,           # pixel-level threshold for glare-like
    coverage_thresh=0.02,        # fraction of image area to call “glare”
):
    """
    Classical glare detection via photometric maps - Intensity Map, Saturation Map and Low-Contrast Mapping

    Inputs
    ------
    bgr : np.ndarray (H,W,3)  uint8   OpenCV image (BGR)
    win : int                 local window (odd) for contrast (std dev) map
    weights : tuple[float]    weights for (Intensity, LowSaturation, LowLocalContrast)
    score_thresh : float      per-pixel glare score cutoff in [0,1]
    coverage_thresh : float   fraction of pixels above score_thresh to label image as “glare”

    Returns
    -------
    result : dict with keys:
        'score'   : float                # percentage of glare pixels, 0..1
        'has_glare': bool
        'score_map': np.ndarray float32  # HxW glare score in [0,1]
        'mask'     : np.ndarray uint8    # HxW {0,255} glare mask after cleanup
    """
    # Run Ashley's function to extract the image from the bounding box and use that output as the input for the rest of this
    
    # Etxract image height and width
    H, W = bgr.shape[:2]

    # Convert photo to HSV (hue, saturation, value)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    V = V.astype(np.float32) / 255.0 # Illuminance
    S = S.astype(np.float32) / 255.0 # Saturation

    # Local contrast mapping on illuminance (V)
    k = (win, win)
    mean_V  = cv2.boxFilter(V, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_V2 = cv2.boxFilter(V*V, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    std_V   = np.sqrt(np.maximum(mean_V2 - mean_V*mean_V, 0.0))

    # Robust normalize
    def robust_norm(x, p_lo=2.0, p_hi=98.0, eps=1e-6):
        lo = np.percentile(x, p_lo); hi = np.percentile(x, p_hi)
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

    I_norm = robust_norm(V)
    S_norm = robust_norm(S)
    C_norm = robust_norm(std_V)

    intensity_like      = I_norm                # bright
    low_saturation_like = 1.0 - S_norm          # low S
    low_contrast_like   = 1.0 - C_norm          # low local contrast

    # Combine the scores and weights for all the 3 methods to get overall score
    wI, wS, wC = weights
    score_map = (wI*intensity_like + wS*low_saturation_like + wC*low_contrast_like) / (wI+wS+wC)
    score_map = score_map.astype(np.float32)

    # Pixel-level threshold → mask
    mask = (score_map >= score_thresh).astype(np.uint8) * 255

    coverage = np.count_nonzero(mask) / float(H * W)
    has_glare = coverage >= coverage_thresh

    # Find the center of the glare?
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, _, _, maxLoc = cv2.minMaxLoc(gray)   # (x, y)
    print("Glare center (px):", maxLoc)


    return {
        'score': coverage,
        'has_glare': has_glare,
        'score_map': score_map,
        'mask': mask,
        'X_coord' : maxLoc
    }

res = detect_glare(bgr)

print("Has glare:", res['has_glare'])
print("Glare coverage:", f"{res['score']:.4f}")


# Edit this code to get the center coordinate for the glare blob, so that it can be used in moving the camera
# Also output the image that was extracted to put into the camera move function