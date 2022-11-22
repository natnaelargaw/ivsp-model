from __future__ import division

import os

import numpy as np
import scipy.io
import scipy.ndimage
# from scipy.misc import imresize
import cv2 as cv
import numpy as np

from config import shape_c, shape_r


def imresize(image, shape=(224, 224)):
    img = cv.resize(image, shape)
    return img


def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = imresize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = imresize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r

        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_bin_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))
    # receives 5 frames at one time
    for i, ori_path in enumerate(paths):
        # print(path)
        original_image = cv.imread(ori_path)

        copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
        if original_image.shape == 2:
            # copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image

            original_image = copy
        padded_image = padding(original_image, shape_r, shape_c, 3)

        ims[i] = padded_image
    # print(len(ims), "Length of images in preprocess bin")

    # call process X here with 5 inputs



    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]
    # ims = ims.transpose((0, 3, 1, 2))
    # display_images(ims)

    ims = process_X(ims)
    # display_images(ims)
    return ims


# def display_images(ims):
#     # frame = cv.imread("samples/450frames0.png")
#     # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # cv.imwrite("samples/frames.png", grey)
#
#     for i in range(len(ims)):
#         image = ims[i]
#         # change this image to grey
#         # other methods do not work
#         # rgb_weights = [0.2989, 0.5870, 0.01140]
#         # grey = np.dot(image[...,:3],rgb_weights)
#         cv.imwrite("samples/"+str(len(ims))+"frames"+str(i)+".png", image)

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, ori_path in enumerate(paths):
        # print(path)
        original_image = cv.imread(ori_path)
        # bin_image = cv.imread(bin_path)
        # original_image = mpimg.imread(path)
        # original_image = imread(path)
        # if original_image.ndim == 2:
        copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
        if original_image.shape == 2:
            # copy = np.zeros((original_image.shape[0], original_image.shape[1], 3)
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image

            # copy[:, :, 0] = bin_image
            # copy[:, :, 1] = bin_image
            # copy[:, :, 2] = bin_image
            # copy[:, :, 3] = bin_image
        # else:
        #     copy[:, :, 0] = bin_image[:, :, 0]
        #     copy[:, :, 1] = bin_image[:, :, 0]
        #     copy[:, :, 2] = bin_image[:, :, 0]

            original_image = copy
        # if original_image.shape[2] ==3:
        #     original_image = merge_channels(original_image, bin_image)

        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image
        # Potentially average of frames in each channel and every video
    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]
    # ims = ims.transpose((0, 3, 1, 2))

    return ims


# def preprocess_images(paths, shape_r, shape_c):
#     ims = np.zeros((len(paths), shape_r, shape_c, 3))
#
#     for i, path in enumerate(paths):
#         # print(path)
#         original_image = cv.imread(path)
#         # original_image = mpimg.imread(path)
#         # original_image = imread(path)
#         # if original_image.ndim == 2:
#         if original_image.shape == 2:
#             copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
#
#             copy[:, :, 0] = original_image
#             copy[:, :, 1] = original_image
#             copy[:, :, 2] = original_image
#
#             original_image = copy
#         padded_image = padding(original_image, shape_r, shape_c, 3)
#         ims[i] = padded_image
#
#     ims[:, :, :, 0] -= 103.939
#     ims[:, :, :, 1] -= 116.779
#     ims[:, :, :, 2] -= 123.68
#     ims = ims[:, :, :, ::-1]
#     # ims = ims.transpose((0, 3, 1, 2))
#
#     return ims


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = cv.imread(path, 0)
        # original_map = mpimg.imread(path)
        # original_map = imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0

    return ims


def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = imresize(pred, (shape_r, new_cols))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    img = img / np.max(img) * 255

    return img


def merge_channels(rgb, binarized):
    b, g, r = cv.split(rgb)
    return cv.merge(b, g, r, binarized)


def process_X(small_batch):
    # print(len(small_batch), "Size in process batch")
    ims3 = np.zeros((len(small_batch), shape_r, shape_c, 3))

    for i in range(len(small_batch)-1):
        if i == 0:
            frame_current = small_batch[i]
            frame_ref_left = small_batch[i + 1]
            frame_ref_right = small_batch[i + 1]
            # cv.imshow("Frame 1", frame_current)
            # cv.waitKey(1)
        elif i == len(small_batch):#last element

            frame_current = small_batch[i]
            frame_ref_left = small_batch[i - 1]
            frame_ref_right = small_batch[i - 2]
        else:
            frame_current = small_batch[i]
            frame_ref_left = small_batch[i - 1]
            frame_ref_right = small_batch[i + 1]
        ims3[i] = transform(frame_current, frame_ref_left, frame_ref_right)

    return ims3

def transform(fc, fl,fr):
    # converting to grey - cv.cvtcolor doesn't work because of the numpy nature of image ?
    rgb_weights = [0.2989, 0.5870, 0.01140]
    frame_current = np.dot(fc[...,:3],rgb_weights)
    frame_past = np.dot(fl[...,:3],rgb_weights)
    frame_next = np.dot(fr[...,:3],rgb_weights)

    # frame_current = cv.cvtColor(fc, cv.COLOR_RGB2GRAY)
    # frame_past = cv.cvtColor(fl, cv.COLOR_RGB2GRAY)
    # frame_next = cv.cvtColor(fr, cv.COLOR_RGB2GRAY)


    # delta_future = cv.absdiff(frame_next, frame_current)
    delta_past = cv.absdiff(frame_past, frame_current)



    # bitwise or of temproal differences
    # img_bwo = cv.bitwise_or(delta_past, delta_future)

    # kernel = np.ones((3, 3), np.uint8)
    # img_bwo = cv.erode(img_bwo, kernel)

    # Dilating and binarizing temporal differences
    # ret, thresh = cv.threshold(img_bwo, 25, 255, cv.THRESH_BINARY)
    # dilate_frame = cv.dilate(thresh, None, iterations=2)

    # Is it required: ? to make more neighbour similarity
    # blured_copy = cv.GaussianBlur(frame_current, (21, 21), 0)

    # Shaked difference
    # subtracter = np.copy(frame_current)[1:frame_current.shape[0], 1:frame_current.shape[1]]
    # print(subtracter.shape)

    thresholded = cv.threshold(delta_past, 75,255,cv.THRESH_BINARY)[1]


    # print(subtracter.shape)
    #
    # difference = cv.absdiff(frame_current[0:frame_current.shape[0] - 1, 0:frame_current.shape[1] - 1], subtracter)
    # difference = frame_current[0:frame_current.shape[0] - 1, 0:frame_current.shape[1] - 1] - subtracter
    # shape = (360, 640)
    shape = (256, 320)

    # plain = np.zeros(shape, dtype="uint8")
    # plain[0:frame_current.shape[0] - 1, 0:frame_current.shape[1] - 1] = difference
    # plain[frame_current.shape[0]:, frame_current.shape[1]:] = subtracter[subtracter.shape[0]:, subtracter.shape[1]:]

    dilate_frame = cv.dilate(thresholded, None, iterations=2)


    # kernel = np.ones((3, 3), np.uint8)
    # eroded = cv.erode(plain, kernel)
    #
    # # additive = cv.add(eroded ,dilate_frame)
    # spatio_temporal_ready_frame = cv.add(eroded, dilate_frame)
    #
    # # eroded_max = cv.erode(maximize, kernel)
    #
    #
    r, g, b = cv.split(fc)
    r = np.maximum(r, dilate_frame)
    g = np.maximum(g, dilate_frame)
    b = np.maximum(b, dilate_frame)
    #
    spatio_temporal_ready_frame = cv.merge((r,g,b))

    # cv.imshow(" Spatio Temporally Ready Frame ", spatio_temporal_ready_frame)
    # cv.imshow(" Spatio Temporally Ready Frame ", plain)

    # cv.waitKey(0)
    # print("Completed Batch")
    return spatio_temporal_ready_frame


if __name__ == '__main__':
    videos_train_paths = ['/home/natnael/Documents/datasets/DHF1K/train_images/']
    videos = [videos_train_path + f for videos_train_path in videos_train_paths for f in
              os.listdir(videos_train_path) if os.path.isdir(videos_train_path + f)]
    videos.sort()
    for i in range(2):
        path = videos[i]
        images = [path + '/images/' + f for f in
                  os.listdir(path + '/images/') if f.endswith(('.jpg', '.jpeg', '.png'))]

        preprocess_bin_images(images, shape_r, shape_c)



