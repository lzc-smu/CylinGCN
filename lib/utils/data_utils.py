import numpy as np
import cv2
import random
import math

def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug_3d(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    for i in range(image.shape[2]):
        inp = np.concatenate((image[..., [i]], image[..., [i]], image[..., [i]]), axis=-1)
        gs_mean = image[..., i].mean()
        for f in functions:
            f(data_rng, inp, image[..., i], gs_mean, 0.4)
        lighting_(data_rng, inp, 0.1, eig_val, eig_vec)
        image[..., i] = grayscale(inp)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_gt_mask(img, contour):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    for i in range(len(contour)):
        for j in range(len(contour[i])):
            cv2.fillPoly(mask, [np.round(contour[i][j]['poly']).astype(int)], 1)
    return mask


def init_uniform_cirsample(center, inp_out_hw, num, rr):
    """create initial circle contour"""
    h, w = inp_out_hw[2:]
    a, b = center[:]
    r = min((h - a), (w - b), a, b) * rr

    alpha = np.linspace(0, 2*math.pi*(num-1)/num, num)
    x = np.reshape(a + r * np.sin(alpha), (-1, 1))
    y = np.reshape(b + r * np.cos(alpha), (-1, 1))
    contour = np.concatenate((x, y), axis=1)
    return contour


def add_deep_dim(contour, layer):
    d = contour[..., [0]]
    d[...] = layer
    new = np.concatenate((d, contour), axis=-1)
    return new


def init_gt_match(init, gt, center, num):
    alpha = np.linspace(0, 2 * math.pi * (num - 1) / num, num)
    gt_hw = gt - center.reshape(1, 2)
    fp_r = np.sqrt(np.power(gt_hw, 2).sum(axis=1))
    fp_alpha = np.arccos(gt_hw[..., 1] / fp_r)
    for j in range(len(fp_alpha[:, ...])):
        fp_alpha[j] = 2 * math.pi - fp_alpha[j] if gt_hw[j, 0] < 0 else fp_alpha[j]
    fir_idx = np.argmin(np.absolute(fp_alpha - alpha[0]))
    sample_ = gt[[fir_idx]]
    for i in range(init.shape[0] - 1):
        idx = np.argmin(np.absolute(fp_alpha - alpha[i + 1]))
        sample_= np.concatenate((sample_, gt[[idx]]), axis=0)
    return sample_