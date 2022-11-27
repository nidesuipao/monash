import skimage
import numpy as np
from skimage.color import rgb2gray
from scipy import ndimage as ndi
import cv2
import random
import os
#collect some images from a video and store it in dir 'capture_images'
# filename = 'cangjie_2.mp4'
# vid = imageio.get_reader(filename, 'ffmpeg')
# for num,im in enumerate(vid):
#     #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
#     # image = skimage.img_as_float(im).astype(np.uint8)
#     if num % 100 == 0:
#         skimage.io.imsave('./capture_images/' + str(num) + '.jpg', im)

def  create_collage(feature):
    #
    censures = skimage.feature.CENSURE()
    imgs = []
    for file in os.listdir('capture_images'):
        imgs.append(skimage.io.imread(os.path.join('capture_images',file)))

    collage_num = 5
    i = 0
    collage_imgs = []
    while True:
        index = random.randint(0,len(imgs)-1)
        img = imgs[index]
        # print(img.shape)
        img = img[150:1080-150,:]
        # img = transform.resize(img, (int(img.shape[0]/2), int(img.shape[1]/2),3)) * 255
        collage_imgs.append(img)
        imgs.pop(index)
        i += 1
        if i == collage_num:
            break
    rgb_imgs = collage_imgs.copy()


    def edge_compute(x):
        x = rgb2gray(x)
        edge = skimage.feature.canny(x, sigma=1)
        edge = np.array(edge, dtype=np.uint8)
        return edge.sum()

    def hog_compute(x):
        x = rgb2gray(x)
        normalised_blocks, hog  = skimage.feature.hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2-Hys',visualize=True)
        hog = np.array(hog, dtype=np.uint8)
        return hog.sum()


    def censure_compute(x):
        x = rgb2gray(x)
        censures.detect(x)
        return censures.keypoints.sum()
    if feature == 'edges':
        rgb_imgs = sorted(rgb_imgs, key=lambda x:edge_compute(x))
        edges = []
        for i in range(len(collage_imgs)):
            collage_imgs[i] = rgb2gray(collage_imgs[i])
            edge = skimage.feature.canny(collage_imgs[i], sigma=1)
            edge = np.array(edge, dtype=np.uint8) * 255
            edges.append(edge)

        for i in range(len(edges)):
            skimage.io.imsave('edges/edge' + str(i) +'.jpg', edges[i])
            skimage.io.imsave('edges/ori' + str(i) + '.jpg', rgb_imgs[i])
    if feature == 'hogs':
        rgb_imgs = sorted(rgb_imgs, key=lambda x:hog_compute(x))
        hogs = []
        for i in range(len(rgb_imgs)):
            normalised_blocks, hog = skimage.feature.hog(rgb2gray(rgb_imgs[i]) * 255, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), block_norm='L2-Hys',visualize=True)
            hog = np.array(hog) * 255
            hogs.append(hog)

        for i in range(len(hogs)):
            skimage.io.imsave('hogs/hog' + str(i) +'.jpg', hogs[i])
            skimage.io.imsave('hogs/ori' + str(i) + '.jpg', rgb_imgs[i])
    if feature == 'censures':
        rgb_imgs = sorted(rgb_imgs, key=lambda x:censure_compute(x))
        CENSURE = []
        for i in range(len(collage_imgs)):
            censures.detect(rgb2gray(collage_imgs[i]))
            censure = np.zeros(collage_imgs[i].shape, dtype = np.uint8)
            img_= collage_imgs[i].copy()
            for m,n in censures.keypoints:
                censure[m][n] = 255
            for point in censures.keypoints:
                img_ = cv2.putText(img_, '+', (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1)
            skimage.io.imsave('censures/censure_' + str(i) + '.jpg', img_)
            CENSURE.append(censure)

        for i in range(len(CENSURE)):
            skimage.io.imsave('censures/censure' + str(i) +'.jpg', CENSURE[i])
            skimage.io.imsave('censures/ori' + str(i) + '.jpg', rgb_imgs[i])

    img = rgb_imgs[0]
    img = np.concatenate((img, rgb_imgs[1]), axis=1)
    im = np.concatenate((img, rgb_imgs[2]), axis=1)
    img1 = skimage.transform.resize(rgb_imgs[3], (1170, 2880)) * 255
    img2 = skimage.transform.resize(rgb_imgs[4], (1170, 2880)) * 255
    img = np.concatenate((img1, img2), axis=1)
    im = np.concatenate((im, img), axis=0)

    def is_smooth(i,j, x_, y_, rapid):
        if i < x_[0] - rapid:
            if (j > y_[0] - rapid and j < y_[0] + rapid) or (j > y_[2] - rapid and j < y_[2] + rapid):
                return 1
        if i > x_[0] + rapid:
            if (j > y_[1] - rapid and j < y_[1] + rapid):
                return 1
        if i >= x_[0] - rapid and i <= x_[0] + rapid:
            return 1
        return 0

    def smooth(img, img_, rapid):
        y_ = [1920, 2880, 3840]
        x_ = [780]
        im2 = np.zeros(img.shape)
        kernel_size = 21 - rapid / 10
        im2[:, :, 0] = ndi.gaussian_filter(img[:, :, 0], (kernel_size, kernel_size))
        im2[:, :, 1] = ndi.gaussian_filter(img[:, :, 1], (kernel_size, kernel_size))
        im2[:, :, 2] = ndi.gaussian_filter(img[:, :, 2], (kernel_size, kernel_size))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if is_smooth(i,j,x_,y_, rapid):
                    img[i][j] = im2[i][j]
        return img
    im_ = im.copy()
    for i in range(200,5,-10):
        im = smooth(im_,rgb_imgs[0], i)
    skimage.io.imsave('./' + feature +'/collage.jpg', im)

create_collage('edges')
create_collage('hogs')
create_collage('censures')

