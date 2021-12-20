import numpy as np
import cv2 as cv
import time
import os


"""
    A simple re-implementation for paper "Coherent Line Drawing" using
numpy and opencv, all operations are vectorized (but not so well =v=).
Performance is better than any other re-implementation of Python in github
but slower than re-implementation of C++.
    The result is bad in comparison with paper's result. But I think most
of my implementation is correct.

    Author: linxinqi@tju.edu.cn
    Date:   2021-12-18 03:55
    << Do whatever you want with this code >>
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

    # neighbors of each tangent vector in each window
    flow_neighbors = find_neighbors(flow_padded, ksize,
            s=1, out_h=h, out_w=w)

    # centural tangent vector in each window
    flow_me = flow_neighbors[:, :, :, ksize // 2, ksize // 2]
    flow_me = np.expand_dims(flow_me, axis=(3, 4))

    # compute dot
    dots = np.sum(flow_neighbors * flow_me,
        axis=2, keepdims=True)

    # compute phi
    phi = np.where(dots > 0, 1, -1)

    # compute wd
    wd = np.abs(dots)
    
    # compute wm
    mag_padded = np.pad(mag, ((0, 0), (p, p), (p, p)))
    mag_neighbors = find_neighbors(mag_padded, ksize,
            s=1, out_h=h, out_w=w)
    mag_me = mag_neighbors[:, :, :, ksize // 2, ksize // 2]
    mag_me = np.expand_dims(mag_me, axis=(3, 4))
    wm = (1 + np.tanh(mag_neighbors - mag_me)) / 2

    # compute ws
    ws = np.ones_like(wm)
    x, y = np.meshgrid(np.arange(ksize), np.arange(ksize))
    cx, cy = ksize // 2, ksize // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)[None, :, :]
    ws[:, :, dist >= ksize // 2] = 0

    # update flow
    flow = np.sum(phi * flow_neighbors * ws * wm * wd, axis=(3, 4))
    flow = np.transpose(flow, axes=(2, 0, 1))

    # normalize flow
    norm = np.sqrt(np.sum(flow ** 2, axis=0))
    none_zero_mask = norm != 0
    flow[:, none_zero_mask] /= norm[none_zero_mask]

    return flow


def create_filters(gauss_size, sigma_m,
    dog_size, sigma_c, rho):
    sigma_s = 1.6 * sigma_c

    x = np.arange(-gauss_size, gauss_size + 1)
    gauss_f = np.exp(-(x ** 2) / (2 * sigma_m ** 2)) / \
        (np.sqrt(2 * np.pi) * sigma_m)

    x = np.arange(-dog_size, dog_size + 1)
    gauss_c = np.exp(-(x ** 2) / (2 * sigma_c ** 2)) / \
        (np.sqrt(2 * np.pi) * sigma_c)
    gauss_s = np.exp(-(x ** 2) / (2 * sigma_s ** 2)) / \
        (np.sqrt(2 * np.pi) * sigma_s)
    dog_f = gauss_c - rho * gauss_s

    return gauss_f.astype('float32'), dog_f.astype('float32')


def detect_edge(img, flow, gauss_size, sigma_m,
    dog_size, sigma_c, rho, tau):
    h, w = img.shape

    # gaussian filter and log filter
    gauss_f, dog_f = create_filters(gauss_size, sigma_m,
                    dog_size, sigma_c, rho)
    
    # do padding
    img_padded = np.pad(img,
        ((dog_size, dog_size), (dog_size, dog_size)))
    
    # start coords of each pixel (shifted by q)
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    sx = np.expand_dims(sx, axis=0)
    sy = np.expand_dims(sy, axis=0)
    start = np.concatenate((sx, sy), axis=0) + dog_size

    steps = np.arange(-dog_size, dog_size + 1).reshape(-1, 1, 1, 1)
    steps = np.repeat(steps, repeats=2, axis=1)

    grad = np.empty_like(flow)
    grad[0, :, :] = flow[1, :, :]
    grad[1, :, :] = -flow[0, :, :]

    # take steps along the gradient
    xy = start + (steps * grad)
    ixy = np.round(xy).astype('int32')
    ix, iy = np.split(ixy, indices_or_sections=2, axis=1)
    ix = ix.reshape(2 * dog_size + 1, h, w)
    iy = iy.reshape(2 * dog_size + 1, h, w)

    # neighbors of each pixel along the gradient
    neighbors = img_padded[iy, ix]

    # apply dog filter in gradient's direction
    dog_f = dog_f.reshape(2 * dog_size + 1, 1, 1)
    img = np.sum(dog_f * neighbors, axis=0)

    img_padded = np.pad(img,
        ((gauss_size, gauss_size), (gauss_size, gauss_size)))
    flow_padded = np.pad(flow, ((0, 0),
        (gauss_size, gauss_size), (gauss_size, gauss_size)))

    H = np.zeros_like(img)

    # smooth neighbors alone tangent
    sx, sy = np.meshgrid(np.arange(w), np.arange(h))
    sx += gauss_size; sy += gauss_size

    x = sx.astype('float32')
    y = sy.astype('float32')
    for i in range(gauss_size):
        ix, iy = np.round(x).astype('int32'), \
            np.round(y).astype('int32')
        neighbor = img_padded[iy, ix]
        # multiply weight
        H += (neighbor * gauss_f[gauss_size + i])
        # take a step
        x += flow_padded[0, iy, ix]
        y += flow_padded[1, iy, ix]

    x = sx.astype('float32')
    y = sy.astype('float32')
    for i in range(1, gauss_size):
        ix, iy = np.round(x).astype('int32'), \
            np.round(y).astype('int32')
        neighbor = img_padded[iy, ix]
        # multiply weight
        H += (neighbor * gauss_f[gauss_size - i])
        # take a step
        x -= flow_padded[0, iy, ix]
        y -= flow_padded[1, iy, ix]

    # binarize
    mask = np.logical_and(H < 0, (1 + np.tanh(H)) < tau)
    edge = np.where(mask, 0, 255).astype('float32')
    return edge


def run(img, sobel_size=3, etf_iter=2, etf_size=9,
    fdog_iter=3, gauss_size=13, sigma_m=3.0,
    dog_size=5, sigma_c=1.0, rho=0.99, tau=0.2):
    """
    Running coherent line drawing on input image.

    Parameters:
    -----------
    - img: ndarray
        input image, with shape (h, w, c)

    - sobel_size: int, default=3
        size of sobel filter

    - etf_iter: int, default=2
        iteration times of refining edge tangent flow
    
    - etf_size: int, default=9
        size of etf filter

    - fdog_iter: int, default=3
        iteration times of applying fdog on input image

    - gauss_size: int, default=13
        size of 1-d gaussian filter along tangent vector,
        if gauss_size = k, the length of gaussian filter 
        will be 2 * k + 1
    
    - sigma_m: float, default=3.0
        standard variance of gaussian filter

    - dog_size: int, default=5
        size of 1-d different of gaussian filter along
        gradient, if dog_size = k, then length of dog filter
        will be 2 * k + 1

    - sigma_c: float, default=1.0
        standard variance of one gaussian filter of dog filter,
        another's standard variance will be set to 1.6 * sigma_c

    - rho: float, default=0.99
        dog = first gaussian - rho * second gaussian

    - tau: float, default=0.2
        threshold of edge map

    Returns:
    --------
    - edge: ndarray
        edge map of input image, data type is float32 and pixel's
        range is clipped to [0, 255]
    """
    # preprocess input image
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.bilateralFilter(img, 15, 15, 10)

    # conver to float32 for convenience
    img = img.astype('float32')

    grad_x = cv.Sobel(img, cv.CV_32F,
            1, 0, ksize=sobel_size)
    grad_y = cv.Sobel(img, cv.CV_32F,
            0, 1, ksize=sobel_size)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # normalize gradient and get tangent vector
    none_zero_mask = mag != 0
    grad_x[none_zero_mask] /= mag[none_zero_mask]
    grad_y[none_zero_mask] /= mag[none_zero_mask]

    # normalize magnitude
    mag = cv.normalize(mag, dst=None,
        norm_type=cv.NORM_MINMAX)

    # rotate gradient and get tangent vector
    flow_x, flow_y = -grad_y, grad_x

    # expand dimension in axis=0 for vectorizing
    flow_x = np.expand_dims(flow_x, axis=0)
    flow_y = np.expand_dims(flow_y, axis=0)
    flow = np.concatenate((flow_x, flow_y), axis=0)
    mag = np.expand_dims(mag, axis=0)

    # refine edge tangent flow
    for i in range(etf_iter):
        start = time.perf_counter()
        flow = refine_flow(flow, mag, ksize=etf_size)
        end = time.perf_counter()
        print(f"smoothing edge tangent flow, iteration {i + 1}, "
                f"time cost = {end - start:<6f}s")

    # do fdog
    for i in range(fdog_iter):
        start = time.perf_counter()
        edge = detect_edge(img, flow,
            gauss_size=gauss_size, sigma_m=sigma_m,
            dog_size=dog_size, sigma_c=sigma_c, rho=rho, tau=tau)
        img += edge
        img = np.clip(img, 0, 255)
        end = time.perf_counter()
        print(f"applying fdog, iteration {i + 1}, "
                f"time cost = {end - start:<6f}s")

    return detect_edge(img, flow,
        gauss_size=gauss_size, sigma_m=sigma_m,
        dog_size=dog_size, sigma_c=sigma_c, rho=rho, tau=tau)


if __name__ == "__main__":
    tests = [
        'test1.jpg', 'test2.jpg', 'test3.jpg',
        'eagle.jpg', 'butterfly.jpg', 'lighthouse.png',
        'star.jpg', 'girl.jpg'
    ]

    for test in tests:
        print(f"running on test {test}")
        # read image by opencv
        img = cv.imread(os.path.join('benchmarks', test))

        # shrink image if its size is considerable (500?)
        shape = img.shape[:-1][::-1]
        if any(map(lambda sz: sz > 500, shape)):
            img = cv.resize(img,
                tuple(map(lambda sz: int(sz * 0.5), shape)))
        print(f"input shape = {shape}")

        # run on this image and return edge map
        edge = run(
            img=img,
            sobel_size=3,
            etf_iter=2,
            etf_size=9,
            fdog_iter=3,
            gauss_size=13,
            sigma_m=3.0,
            dog_size=5,
            sigma_c=1.0,
            rho=0.99,
            tau=0.2
        )
        print("{}-{}".format('- ' * 10, ' -' * 9))

        # show origin image and edge map
        cv_imshow(img, title="input", wait=0.1)
        cv_imshow(edge.astype('uint8'), title="output", wait=0.5,
            move=(int(img.shape[1] * 1.5), 0))

        # save result
        cv.imwrite(f"edge_{test}", edge)
        cv.destroyAllWindows()