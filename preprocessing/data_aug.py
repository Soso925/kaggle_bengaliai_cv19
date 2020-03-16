import numpy as np
import cv2

import numpy as np


def data_generator(x_img, y_gr, y_vd, y_cd, batch_size=128, mode_data_aug=True, mixup_alpha=1, \
                   srs_mode={'rotate': 5, 'scale': 0.1, 'shift': 0.1}, cutmix=True \
                   ):
    while 1:
        res_x = []
        res_y_gr = []
        res_y_vd = []
        res_y_cd = []

        N = len(x_img)
        i_batch = np.random.choice(a=N, size=batch_size)
        tmp_x = x_img[i_batch]
        tmp_y_gr, tmp_y_vd, tmp_y_cd = y_gr[i_batch], y_vd[i_batch], y_cd[i_batch]

        if mode_data_aug == False:
            res_x.append(tmp_x)
            res_y_gr.append(tmp_y_gr)
            res_y_vd.append(tmp_y_vd)
            res_y_cd.append(tmp_y_cd)

        if mode_data_aug == True:

            if mixup_alpha != 0:
                mix_img, mix_y_gr, mix_y_vd, mix_y_cd = mix_up(tmp_x, tmp_y_gr, tmp_y_vd, tmp_y_cd, mixup_alpha)
                res_x.append(mix_img)
                res_y_gr.append(mix_y_gr)
                res_y_vd.append(mix_y_vd)
                res_y_cd.append(mix_y_cd)

            if cutmix != 0:
                res_img, res_y_gr, res_y_vd, res_y_cd = cutmix(tmp_x, tmp_y_gr, tmp_y_vd, tmp_y_cd)
                res_x.append(res_img)
                res_y_gr.append(res_y_gr)
                res_y_vd.append(res_y_vd)
                res_y_cd.append(res_y_cd)

            if ((srs_mode['rotate'] != 0) or (srs_mode['scale'] != 0) or (srs_mode['shift'] != 0)):
                tmp = []
                for index in range(batch_size):
                    tmp.append(random_scale_rotate_shift(tmp_x[index],
                                                         mode={'rotate': srs_mode['rotate'], 'scale': srs_mode['scale'],
                                                               'shift': srs_mode['shift']}))
                tmp = np.array(tmp)
                res_x.append(tmp)
                res_y_gr.append(tmp_y_gr)
                res_y_vd.append(tmp_y_vd)
                res_y_cd.append(tmp_y_cd)

        res_y_gr = np.array(res_y_gr)
        res_y_vd = np.array(res_y_vd)
        res_y_cd = np.array(res_y_cd)

        res_y_gr = res_y_gr.reshape(batch_size, 168)
        res_y_vd = res_y_vd.reshape(batch_size, 11)
        res_y_cd = res_y_cd.reshape(batch_size, 7)

        yield (res_x, {'hgr': res_y_gr, 'hvd': res_y_vd, 'hcd': res_y_cd})


def cutmix(images, y_gr, y_vd, y_cd):  # of a batch

    batch_size, h, w, c = images.shape

    # mix-up
    perm = np.random.permutation(batch_size)
    perm_img = images[perm]
    perm_y_gr = y_gr[perm]
    perm_y_vd = y_vd[perm]
    perm_y_cd = y_cd[perm]

    lbd = np.random.uniform(low=0.0, high=1.0, size=None)
    r_x = np.random.randint(w)
    r_y = np.random.randint(h)
    r_w = np.int(np.sqrt(1 - lbd) * w)
    r_h = np.int(np.sqrt(1 - lbd) * h)
    x1 = np.clip(r_x - r_w // 2, 0, w)
    x2 = np.clip(r_x + r_w // 2, 0, w)
    y1 = np.clip(r_y - r_h // 2, 0, h)
    y2 = np.clip(r_y + r_h // 2, 0, h)
    print('test:', x1, x2, y1, y2)

    res_img = images.copy()
    res_img[:, x1:x2, y1:y2, :] = perm_img[:, x1:x2, y1:y2, :]
    lbd = 1 - (x2 - x1) * (y2 - y1) / (w * h)

    res_y_gr = lbd * y_gr + (1 - lbd) * perm_y_gr
    res_y_vd = lbd * y_vd + (1 - lbd) * perm_y_vd
    res_y_cd = lbd * y_cd + (1 - lbd) * perm_y_cd

    return res_img, res_y_gr, res_y_vd, res_y_cd


def mix_up(images, y_gr, y_vd, y_cd, alpha=1):
    gamma = np.random.beta(0.4, alpha)  # by default, beta = 0.4 (according to original paper)
    gamma = max(1 - gamma, gamma)

    batch_size = len(images)

    # mix-up
    perm = np.random.permutation(batch_size)
    perm_img = images[perm]

    perm_y_gr = y_gr[perm]
    perm_y_vd = y_vd[perm]
    perm_y_cd = y_cd[perm]

    mix_img = gamma * images + (1 - gamma) * perm_img
    mix_y_gr = gamma * y_gr + perm_y_gr * (1 - gamma)
    mix_y_vd = gamma * y_vd + perm_y_vd * (1 - gamma)
    mix_y_cd = gamma * y_cd + perm_y_cd * (1 - gamma)

    # print('mix-up done with alpha : ', alpha, ' with : ', batch_size, 'images.')
    return mix_img, mix_y_gr, mix_y_vd, mix_y_cd


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
    # print(f"random scale: {mode['scale']}, rotate :  {mode['rotate']}, shift : {mode['shift']} on {len(image)} images.")
    if len(image.shape) == 2:
        image = image.reshape(height, width, 1)
    return image
