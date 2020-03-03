import numpy as np
import cv2

def random_scale_rotate_shift(image, mode={'rotate': 10, 'scale': 0.1, 'shift': 0.1}):
    dangle = 0
    dscale_x, dscale_y = 0, 0
    dshift_x, dshift_y = 0, 0

    for k, v in mode.items():
        if 'rotate' == k:
            dangle = np.random.uniform(-v, v)
        elif 'scale' == k:
            dscale_x, dscale_y = np.random.uniform(-1, 1, 2) * v
        elif 'shift' == k:
            dshift_x, dshift_y = np.random.uniform(-1, 1, 2) * v
        else:
            print('sth is wrong with random_scale_rotate_shift function... \n')
    # ----

    height, width = image.shape[:2]

    cos = np.cos(dangle / 180 * np.pi)
    sin = np.sin(dangle / 180 * np.pi)
    sx, sy = 1 + dscale_x, 1 + dscale_y  # 1,1 #
    tx, ty = dshift_x * width, dshift_y * height

    src = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]],
        np.float32)
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1) + width / 2 + tx
    y = (src * [sin, cos]).sum(1) + height / 2 + ty
    src = np.column_stack([x, y])

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s, d)
    image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))

    return image