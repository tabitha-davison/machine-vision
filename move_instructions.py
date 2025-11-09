

def get_move_instruction(glare_pos, frame_shape):
    
    # needs to get phone frame shape from the screen grabber and maybe keep that a consistent dimension

    if glare_pos is None:
        return "No glare detected"

    h, w = frame_shape
    x, y = glare_pos
    center_x, center_y = w // 2, h // 2

    # Define a "safe" zone in the middle where glare is acceptable
    zone_x = w * 0.2  # 20% of width
    zone_y = h * 0.2  # 20% of height

    # Check which side the glare is on relative to the center
    if x < center_x - zone_x:
        return "Move phone LEFT"
    elif x > center_x + zone_x:
        return "Move phone RIGHT"
    elif y < center_y - zone_y:
        return "Tilt phone UP"
    elif y > center_y + zone_y:
        return "Tilt phone DOWN"
    else:
        return "Hold still — glare centered"
