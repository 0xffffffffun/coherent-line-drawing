import numpy as np
import cv2 as cv
import time
import os


"""
    A simple re-implementation for paper "Coherent Line Drawing"
using numpy and opencv, all operations are vectorized. Performance
is faster than any other re-implementation of Python in github but
slower than re-implementation of C++.

    Author: linxinqi@tju.edu.cn
    Date:   2021-12-18 03:55
"""


def cv_imshow(img, title='[TEST]', wait=0.5, move=None):
    cv.imshow(title, img)
    if move:
        cv.moveWindow(title, *move)
    cv.waitKey(int(wait * 1000))


def find_neighbors(x, ksize, s, out_h, out_w):
    in_c, in_h, in_w = x.shape
    shape = (out_h, out_w, in_c, ksize, ksize)
    itemsize = x.itemsize
    strides = (
        s    * in_w * itemsize,
        s    * itemsize,
        in_w * in_h * itemsize,
        in_w * itemsize,
        itemsize
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape,
        strides=strides)


def refine_flow(flow, mag, ksize):
    _, h, w = flow.shape

    # do padding
    p = ksize // 2
    flow_padded = np.pad(flow, ((0, 0), (p, p), (p, p)))

    # get neighbors of each tangent vector
    flow_neighbors = find_neighbors(flow_padded, ksize,
            s=1, out_h=h, out_w=w)

    # retrive centural tangent vector in each window
    flow_me = flow_neighbors[:, :, :, ksize // 2, ksize // 2]
    flow_me = np.expand_dims(flow_me, axis=(3, 4))

    # compute dot
    dots = np.sum(flow_neighbors * flow_me, axis=2, keepdims=True)

    # compute phi
    phi = np.where(dots > 0, 1, -1)
    
    # compute wd
    # wd = np.abs(dots)
    
    # compute wm
    mag_padded = np.pad(mag, ((0, 0), (p, p), (p, p)))
    mag_neighbors = find_neighbors(mag_padded, ksize,
            s=1, out_h=h, out_w=w)
    mag_me = mag_neighbors[:, :, :, ksize // 2, ksize // 2]
    mag_me = np.expand_dims(mag_me, axis=(3, 4))
    # wm = (1 + np.tanh(mag_neighbors - mag_me)) / 2
    wm = 1 + mag_neighbors - mag_me

    # compute ws
    # ws = np.ones_like(wm)
    # x, y = np.meshgrid(np.arange(ksize), np.arange(ksize))
    # cx, cy = ksize // 2, ksize // 2
    # dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)[None, :, :]
    # ws[:, :, dist >= ksize // 2] = 0

    # update flow
    # flow = np.sum(phi * flow_neighbors * ws * wm * wd, axis=(3, 4))
    flow = np.sum(phi * flow_neighbors * wm, axis=(3, 4))
    flow = np.transpose(flow, axes=(2, 0, 1))

    # normalize flow
    norm = np.sqrt(np.sum(flow ** 2, axis=0))
    none_zero_mask = norm != 0
    flow[:, none_zero_mask] /= norm[none_zero_mask]

    return flow


def create_filters(p, q):
    sigma_c = 1
    sigma_s = 1.6 * sigma_c
    sigma_m = 3
    rho = 0.99

    x = np.arange(-p, p + 1)
    gauss_f = np.exp(-(x ** 2) / (2 * sigma_m ** 2)) / \
        (np.sqrt(2 * np.pi) * sigma_m)

    x = np.arange(-q, q + 1)
    gauss_c = np.exp(-(x ** 2) / (2 * sigma_c ** 2)) / \
        (np.sqrt(2 * np.pi) * sigma_c)
    gauss_s = np.exp(-(x ** 2) / (2 * sigma_s ** 2)) / \
        (np.sqrt(2 * np.pi) * sigma_s)
    log_f = gauss_c - rho * gauss_s

    return gauss_f.astype('float32'), log_f.astype('float32')


def detect_edge(img, flow, p, q):
    h, w = img.shape

    # generate filter
    gauss_f, log_f = create_filters(p, q)
 
    # create start coords
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    sx = np.expand_dims(sx, axis=0)
    sy = np.expand_dims(sy, axis=0)
    start = np.concatenate((sx, sy), axis=0) + q

    steps = np.arange(-q, q + 1).reshape(-1, 1, 1, 1)
    steps = np.repeat(steps, repeats=2, axis=1)

    grad = np.empty_like(flow)
    grad[0, :, :] = flow[1, :, :]
    grad[1, :, :] = -flow[0, :, :]

    xy = start + (steps * grad)
    ixy = np.round(xy).astype('int32')
    ix, iy = np.split(ixy, indices_or_sections=2, axis=1)
    ix = ix.reshape(2 * q + 1, h, w)
    iy = iy.reshape(2 * q + 1, h, w)

    # get neighbers for each pixel based on gradient
    img_padded = np.pad(img, ((q, q), (q, q)))
    neighbors = img_padded[iy, ix]

    # merge neighbors
    log_f = log_f.reshape(2 * q + 1, 1, 1)
    img = np.sum(log_f * neighbors, axis=0)

    img_padded = np.pad(img, ((p, p), (p, p)))
    flow_padded = np.pad(flow, ((0, 0), (p, p), (p, p)))

    H = np.zeros_like(img)

    # smooth neighbors alone tangent
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    sx += p; sy += p

    x = sx.astype('float32')
    y = sy.astype('float32')
    for i in range(p):
        ix, iy = np.round(x).astype('int32'), \
            np.round(y).astype('int32')
        neighbor = img_padded[iy, ix]
        # add weight
        H += (neighbor * gauss_f[p + i])
        # take a step
        x += flow_padded[0, iy, ix]
        y += flow_padded[1, iy, ix]

    x = sx.astype('float32')
    y = sy.astype('float32')
    for i in range(1, p):
        ix, iy = np.round(x).astype('int32'), \
            np.round(y).astype('int32')
        neighbor = img_padded[iy, ix]
        # add weight
        H += (neighbor * gauss_f[p - i])
        # take a step
        x -= flow_padded[0, iy, ix]
        y -= flow_padded[1, iy, ix]

    # binarize
    tau = 0.2
    mask = np.logical_and(H < 0, (1 + np.tanh(H)) < tau)
    edge = np.where(mask, 0, 255).astype('float32')
    return edge


def run(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.bilateralFilter(img, 15, 15, 10)
    img = img.astype('float32')

    grad_x = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # normalize gradient and get tangent
    none_zero_mask = mag != 0
    grad_x[none_zero_mask] /= mag[none_zero_mask]
    grad_y[none_zero_mask] /= mag[none_zero_mask]

    # normalize magnitude
    mag = cv.normalize(mag, dst=None, norm_type=cv.NORM_MINMAX)

    # expand dimension in axis=0 for vectorizing
    flow_x = np.expand_dims(-grad_y, axis=0)
    flow_y = np.expand_dims(grad_x, axis=0)
    flow = np.concatenate((flow_x, flow_y), axis=0)
    mag = np.expand_dims(mag, axis=0)

    for i in range(2):
        start = time.perf_counter()
        flow = refine_flow(flow, mag, ksize=9)
        end = time.perf_counter()
        print(f"smoothing edge tangent flow, iteration {i + 1}, "
                f"time cost = {round(end - start, 6)}s")

    for i in range(3):
        start = time.perf_counter()
        edge = detect_edge(img, flow, p=13, q=5)
        img += edge
        img = np.clip(img, 0, 255)
        end = time.perf_counter()
        print(f"applying fdog, iteration {i + 1}, "
                f"time cost = {round(end - start, 6)}s")

    return detect_edge(img, flow, p=13, q=5)


if __name__ == "__main__":
    tests = [
        'test1.jpg', 'test2.jpg', 'test3.jpg',
        'eagle.jpg', 'butterfly.jpg', 'lighthouse.png',
        'star.jpg', 'girl.jpg'
    ]

    for test in tests:
        print(f"running on test {test}")
        img = cv.imread(os.path.join('images', test))
        # img = cv.resize(img, dsize=(100, 100))
        size = np.array(img.shape[:-1])
        if (size > 500).any():
            img = cv.resize(img,
                tuple((size * 0.5).astype('int').tolist()[::-1]))
        edge = run(img)
        print("{}-{}".format('- ' * 10, ' -' * 9))

        cv_imshow(img, title="input", wait=0.1)
        cv_imshow(edge.astype('uint8'), title="output", wait=0,
            move=(int(img.shape[1] * 1.5), 0))
        cv.imwrite(f"edge_{test}", edge)
        cv.destroyAllWindows()