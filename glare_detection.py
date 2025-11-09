import cv2
import numpy as np
from move_instructions import get_move_instruction

def detect_glare(
    bgr,
    win=15,
    weights=(0.45, 0.35, 0.20),   # (w_intensity, w_lowSat, w_lowLocalContrast)
    score_thresh=0.65,            # pixel-level threshold for glare-like
    coverage_thresh=0.02          # fraction of image area to call “glare”
):

    if bgr is None or bgr.size == 0:
        return {'score': 0.0, 'has_glare': False, 'mask': np.zeros((1,1), np.uint8)}

    if len(bgr.shape) == 2: # grayscale 
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    H, W = bgr.shape[:2]
    if H == 0 or W == 0:
        return {'score': 0.0, 'has_glare': False, 'mask': np.zeros((1,1), np.uint8)}

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

    # Normalize 
    def robust_norm(x, p_lo=2.0, p_hi=98.0, eps=1e-6):
        lo = np.percentile(x, p_lo); hi = np.percentile(x, p_hi)
        if hi - lo < eps:   # avoid divide-by-zero on flat images
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

    I_norm = robust_norm(V)
    S_norm = robust_norm(S)
    C_norm = robust_norm(std_V)

    intensity_like      = I_norm
    low_saturation_like = 1.0 - S_norm
    low_contrast_like   = 1.0 - C_norm

    # Combine the scores and weights for all the 3 methods to get overall score
    wI, wS, wC = weights
    score_map = (wI*intensity_like + wS*low_saturation_like + wC*low_contrast_like) / (wI+wS+wC)

    # Find the center of the glare?
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, _, _, maxLoc = cv2.minMaxLoc(gray)   # (x, y)
    instr = get_move_instruction(maxLoc, gray.shape)
    print(instr)

    # Threshold to get glare mask
    mask = (score_map >= score_thresh).astype(np.uint8) * 255
    coverage = float(np.count_nonzero(mask)) / float(H * W)
    has_glare = coverage >= coverage_thresh

    return {'score': coverage, 'has_glare': has_glare, 'mask': mask}







