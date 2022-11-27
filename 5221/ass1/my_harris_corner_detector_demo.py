from skimage.color import rgb2gray, gray2rgb
import skimage
import numpy as np
from scipy import ndimage as ndi
from skimage import filters
import cv2

def meanfilt(src, size):
    conv = np.ones((size, size))
    result = np.zeros((src.shape))
    H, W = src.shape
    for i in range(0, H - size + 1):
        for j in range(0, W - size + 1):
            cur_input = src[i:i + size, j:j + size]
            cur_output = conv * cur_input
            conv_sum = np.sum(cur_output)
            result[i][j] = conv_sum
    return result

def my_harris_corner_detector(img, k = 0.04, sigma = 1, distance = 1, filter_ = 'mean'):
    response = list()
    Ix = filters.sobel(img, axis = 0)
    Iy = filters.sobel(img, axis = 1)
    Ixx = Ix[:, :] ** 2
    Iyy = Iy[:, :] ** 2
    Ixy = Ix[:, :] * Iy[:, :]
    if filter_ == 'mean':
        Ixx =  meanfilt(Ixx, 3)
        Iyy =  meanfilt(Iyy, 3)
        Ixy =  meanfilt(Ixy, 3)
    else:
        Ixx = ndi.gaussian_filter(Ixx, sigma)
        Iyy = ndi.gaussian_filter(Iyy, sigma)
        Ixy = ndi.gaussian_filter(Ixy, sigma)
    cornner_point = np.zeros(img.shape, dtype = np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            M = [[Ixx[i][j], Ixy[i][j]], [Ixy[i][j], Iyy[i][j]]]
            cornner_point[i][j] = np.linalg.det(M) - k * np.trace(M) * np.trace(M)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cornner_point[i][j] > threshold and cornner_point[i][j] == np.max(cornner_point[max(0, i - distance):min(i + distance, img.shape[1] - 1), max(0, j - distance):min(j + distance, img.shape[0] - 1)] ):
                response.append([i,j])
    return response


def harris_corner_cpmpute(img_name, k = 0.04 , sigma = 1, distance = 1, filter_ = 'mean'):
    if img_name == 'astronaut':
        img = skimage.data.astronaut()
        img_ = img.copy()
        skimage.io.imsave('corner/astronaut.jpg', img)
        harris_response = skimage.feature.corner_harris(rgb2gray(img), method='k', k=k, sigma=sigma)
        response = my_harris_corner_detector(rgb2gray(img), k, sigma, distance, filter_)
    if img_name == 'checkerboard':
        img = skimage.data.checkerboard()
        skimage.io.imsave('corner/checkerboard.jpg', img)
        harris_response = skimage.feature.corner_harris(img, method='k', k=k, sigma=sigma)
        response = my_harris_corner_detector(img, k, sigma, distance, filter_)
        img = gray2rgb(img)
        img_ = img.copy()

    res = skimage.feature.corner_peaks(harris_response, min_distance = distance)
    print("the number of corner peaks detected by skimage library:", str(len(res)))
    for point in res:
        img1x = cv2.putText(img, '+', (point[1], point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255,0,0), 1)
    skimage.io.imsave('corner/' + img_name + '_corner1.jpg', img1x)

    print("the number of corner peaks detected by my implementation:", str(len(response)))
    for point in response:
        img1x = cv2.putText(img_, 'o', (point[1], point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,0,255), 1)
    skimage.io.imsave('corner/' + img_name + '_corner2.jpg', img1x)


k = 0.04
sigma = 1
distance = 1
filter_ = 'gaussian' #filter_ can be 'mean' or 'gaussian'
threshold = 0
harris_corner_cpmpute('astronaut', k, sigma, distance, filter_)
harris_corner_cpmpute('checkerboard', k, sigma, distance, filter_)