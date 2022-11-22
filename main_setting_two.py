from __future__ import division
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_bin_images, preprocess_maps, preprocess_fixmaps, \
    postprocess_predictions
from models import acl_vgg, schedule_vgg, kl_divergence, correlation_coefficient, nss
# from scipy.misc import imread, imsave
import random
from math import ceil

import cv2 as cv


# Time Distributed Layer

# Resources: https://levelup.gitconnected.com/
# hands-on-practice-with-time-distributed-layers-using-tensorflow-c776a5d78e7e
# Resources: https://medium.com/smileinnovation/
# how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00
# Documentation architecture: https://github.com/lutzroeder/netron

def generator(video_b_s, image_b_s, phase_gen='train'):
    if phase_gen == 'train':
        # videos = [videos_train_path + f for videos_train_path in videos_train_paths for f in
        #           os.listdir(videos_train_path) if os.path.isfile(videos_train_path + f)]
        videos = [videos_train_path + f for videos_train_path in videos_train_paths for f in
                  os.listdir(videos_train_path) if os.path.isdir(videos_train_path + f)]

        # print(videos[0])

        for item_video in videos:
            images = [imgs_path + item_video[-4:] + '/images/' + f for f in
                      os.listdir(imgs_path + item_video[-4:] + '/images/') if f.endswith(('.jpg', '.jpeg', '.png'))]
            # images = [video +'/' + f for video in videos for f in os.listdir(video+'/') if f.endswith(('.jpg', '.jpeg', '.png'))]

        # spatio_temporal_images = [imgs_path + 'residual_training/'+ video[-8:-4] +'/' + f for video in videos for f in os.listdir(imgs_path + 'residual_training/'+video[-8:-4] +'/') if f.endswith(('.jpg', '.jpeg', '.png'))]

        # nb_train = len(images)/image_b_s
        # print(nb_train, len(images))
        fixationmaps = [imgs_path + item_video[-4:] + '/maps/' + f for f in
                        os.listdir(imgs_path + item_video[-4:] + '/maps/') if f.endswith(('.jpg', '.jpeg', '.png'))]
        # fixationmaps = [imgs_path +'annotation/0'+video[-3:] + '/maps/' + f for video in videos for f in os.listdir(imgs_path +'annotation/0'+video[-3:] +'/maps/') if f.endswith(('.jpg', '.jpeg', '.png'))]

        fixs = [imgs_path + item_video[-4:] + '/fixation/maps/' + f for video in videos for f in
                os.listdir(imgs_path + item_video[-4:] + '/fixation/maps/') if f.endswith(('.mat'))]

        images.sort()
        fixationmaps.sort()
        fixs.sort()
        image_train_data = []
        for image, fixationmap, fix in zip(images, fixationmaps, fixs):
            annotation_data = {'image': image, 'fixmap': fixationmap, 'fix': fix}  # changed
            image_train_data.append(annotation_data)

        random.shuffle(image_train_data)

        # videos.sort()
        random.shuffle(videos)

        loop = 1
        video_counter = 0
        image_counter = 0
        while True:
            if loop % 2:
                Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
                # Xims2 = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))

                Ymaps = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1)) + 0.01
                Yfixs = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1)) + 0.01

                Img_Ymaps = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01
                Img_Yfixs = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01

                for i in range(0, video_b_s):
                    video_path = videos[(video_counter + i) % len(videos)]
                    # images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                    #           f.endswith(('.jpg', '.jpeg', '.png'))]

                    images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                              f.endswith(('.jpg', '.jpeg', '.png'))]

                    # spatio_temporal = [imgs_path + 'residual_training/' + video_path[-7:-4] + '/' + f for f in
                    #           os.listdir(imgs_path + 'residual_training/' + video_path[-7:-4] + '/') if
                    #           f.endswith(('.jpg', '.jpeg', '.png'))]

                    # print(len(images), len(spatio_temporal))

                    # merger function like images = images,spatio_temporal -- split 3 and merge 4

                    maps = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                            f.endswith(('.jpg', '.jpeg', '.png'))]

                    # maps = [imgs_path+'annotation/0' + video_path[-7:-4]+ maps_path + f for f in os.listdir(imgs_path+'annotation/0' + video_path[-7:-4]+ maps_path) if
                    #                 f.endswith(('.jpg', '.jpeg', '.png'))]

                    fixs = [video_path + fixs_path + f for f in os.listdir(video_path + fixs_path) if
                            f.endswith('.mat')]

                    # fixs = [imgs_path+'annotation/0' + video_path[-7:-4] + fixs_path + f for f in os.listdir(imgs_path+'annotation/0' + video_path[-7:-4] + fixs_path) if
                    #         f.endswith('.mat')]
                    # images.sort()
                    # maps.sort()
                    # fixs.sort()
                    start = random.randint(0, max(len(images) - num_frames, 0))
                    X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                    # X2 = preprocess_bin_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)

                    Y = preprocess_maps(maps[start:min(start + num_frames, len(images))], shape_r_out, shape_c_out)
                    Y_fix = preprocess_fixmaps(fixs[start:min(start + num_frames, len(images))], shape_r_out,
                                               shape_c_out)
                    Xims[i, 0:X.shape[0], :] = np.copy(X)
                    # Xims2[i, 0:X.shape[0], :] = np.copy(X2)

                    Ymaps[i, 0:Y.shape[0], :] = np.copy(Y)
                    Yfixs[i, 0:Y_fix.shape[0], :] = np.copy(Y_fix)

                    Xims[i, X.shape[0]:num_frames, :] = np.copy(X[-1, :, :])
                    Ymaps[i, Y.shape[0]:num_frames, :] = np.copy(Y[-1, :, :])
                    Yfixs[i, Y_fix.shape[0]:num_frames, :] = np.copy(Y_fix[-1, :, :])

                yield Xims, [Ymaps, Ymaps, Yfixs, Img_Ymaps, Img_Ymaps, Img_Yfixs]  # add second input here
                video_counter = (video_counter + video_b_s) % len(videos)
                loop = loop + 1
            else:
                Xims = np.zeros((image_b_s, 1, shape_r, shape_c, 3))
                # Xims2 = np.zeros((image_b_s, 1, shape_r, shape_c, 3))

                Ymaps = np.zeros((image_b_s, 1, shape_r_out, shape_c_out, 1)) + 0.01
                Yfixs = np.zeros((image_b_s, 1, shape_r_out, shape_c_out, 1)) + 0.01

                Img_Ymaps = np.zeros((image_b_s, 1, shape_r_attention, shape_c_attention, 1)) + 0.01
                Img_Yfixs = np.zeros((image_b_s, 1, shape_r_attention, shape_c_attention, 1)) + 0.01
                for i in range(0, image_b_s):
                    img_data = image_train_data[(image_counter + i) % len(image_train_data)]
                    # spatio_temporal_data = image_train_data[(image_counter + i) % len(image_train_data)]

                    # X = preprocess_images([img_data['image']], shape_r, shape_c)
                    X = preprocess_images([img_data['image']], shape_r, shape_c)
                    # X2 = preprocess_bin_images([img_data['image']], shape_r, shape_c)

                    Y = preprocess_maps([img_data['fixmap']], shape_r_attention, shape_c_attention)
                    Y_fix = preprocess_fixmaps([img_data['fix']], shape_r_attention, shape_c_attention)

                    Xims[i, 0, :] = np.copy(X)
                    # Xims2[i, 0, :] = np.copy(X2)

                    Img_Ymaps[i, 0, :] = np.copy(Y)
                    Img_Yfixs[i, 0, :] = np.copy(Y_fix)

                yield Xims, [Ymaps, Ymaps, Yfixs, Img_Ymaps, Img_Ymaps, Img_Yfixs]  # #
                image_counter = (image_counter + image_b_s) % len(image_train_data)
                loop = loop + 1

    elif phase_gen == 'val':
        videos = [videos_val_path + f for videos_val_path in videos_val_paths for f in
                  os.listdir(videos_val_path) if os.path.isdir(videos_val_path + f)]

        random.shuffle(videos)

        video_counter = 0
        while True:

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            # Xims2 = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))

            Ymaps = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1)) + 0.01
            Yfixs = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1)) + 0.01

            Img_Ymaps = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01
            Img_Yfixs = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01
            for i in range(0, video_b_s):
                video_path = videos[(video_counter + i) % len(videos)]
                images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]

                # images = [imgs_path + 'val_images/' + video_path[-7:-4] + '/' + f for f in
                #           os.listdir(imgs_path + 'val_images/' + video_path[-7:-4] + '/') if
                #           f.endswith(('.jpg', '.jpeg', '.png'))]

                # spatio_temporal = [imgs_path + 'residual_val/' + video_path[-7:-4] + '/' + f for f in
                #           os.listdir(imgs_path + 'residual_val/' + video_path[-7:-4] + '/') if
                #           f.endswith(('.jpg', '.jpeg', '.png'))]

                # print(len(images), len(spatio_temporal))

                maps = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                        f.endswith(('.jpg', '.jpeg', '.png'))]

                # maps = [imgs_path + 'annotation/0' + video_path[-7:-4] + maps_path + f for f in
                #         os.listdir(imgs_path + 'annotation/0' + video_path[-7:-4] + maps_path) if
                #         f.endswith(('.jpg', '.jpeg', '.png'))]

                fixs = [video_path + fixs_path + f for f in os.listdir(video_path + fixs_path) if
                        f.endswith('.mat')]

                # fixs = [imgs_path + 'annotation/0' + video_path[-7:-4] + fixs_path + f for f in
                #         os.listdir(imgs_path + 'annotation/0' + video_path[-7:-4] + fixs_path) if
                #         f.endswith('.mat')]

                start = random.randint(0, max(len(images) - num_frames, 0))
                X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                # X2 = preprocess_bin_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)

                Y = preprocess_maps(maps[start:min(start + num_frames, len(images))], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(fixs[start:min(start + num_frames, len(images))], shape_r_out,
                                           shape_c_out)
                Xims[i, 0:X.shape[0], :] = np.copy(X)
                Ymaps[i, 0:Y.shape[0], :] = np.copy(Y)
                Yfixs[i, 0:Y_fix.shape[0], :] = np.copy(Y_fix)

                Xims[i, X.shape[0]:num_frames, :] = np.copy(X[-1, :, :])
                # Xims2[i, X.shape[0]:num_frames, :] = np.copy(X2[-1, :, :])

                Ymaps[i, Y.shape[0]:num_frames, :] = np.copy(Y[-1, :, :])
                Yfixs[i, Y_fix.shape[0]:num_frames, :] = np.copy(Y_fix[-1, :, :])

            yield Xims, [Ymaps, Ymaps, Yfixs, Img_Ymaps, Img_Ymaps, Img_Yfixs]  # add second input here
            video_counter = (video_counter + video_b_s) % len(videos)
    else:
        raise NotImplementedError


def get_test(video_test_path):

    print("Video Test Path - ", video_test_path)
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]

    # print("from get test", len(images))
    images.sort()
    start = 0
    while True:
        Xims = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        # Xims2 = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        # print("Shape Xims", Xims.shape)
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        # X2 = preprocess_bin_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        Xims[0, 0:min(len(images) - start, num_frames), :] = np.copy(X)
        # Xims2[0, 0:min(len(images) - start, num_frames), :] = np.copy(X2)

        yield Xims
        # print(Xims.shape)
        start = min(start + num_frames, len(images))

if __name__ == '__main__':
    phase = 'train'
    if phase == 'train':

        # x = Input(batch_shape=(None, None, shape_r, shape_c, 3))
        x = Input(shape=(None, shape_r, shape_c, 3))
        # x2 = Input(batch_shape=(None, None, shape_r, shape_c, 3))
        stateful = False
    else:
        # x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        stateful = True


    if phase == 'train':
        if nb_train % video_b_s != 0 or nb_videos_val % video_b_s != 0:
            print("The number of training and validation images should be a multiple of the batch size. "
                  "Please change your batch size in config.py accordingly.")
            exit()

        m = Model(inputs=x, outputs=acl_vgg(x, stateful))
        print("Compiling ACL-VGG")
        m.compile(Adam(lr=1e-4),
                  loss=[kl_divergence, correlation_coefficient, nss, kl_divergence, correlation_coefficient,
                        nss])
        print("Training ACL-VGG")
        m.fit_generator(generator(video_b_s=video_b_s, image_b_s=image_b_s), nb_train, epochs=nb_epoch,
                        validation_data=generator(video_b_s=video_b_s, image_b_s=0, phase_gen='val'),
                        validation_steps=nb_videos_val,
                        callbacks=[EarlyStopping(patience=15),
                                   ModelCheckpoint('acl-vgg.{epoch:02d}-{val_loss:.4f}.h5', save_best_only=True),
                                   LearningRateScheduler(schedule=schedule_vgg)])

        m.save('ACL.h5')

    elif phase == "test":
        # Cross check and adjust
        # videos_test_path = '../DHF1K/test_imgs/'
        result_path = '/home/natnael/Documents/datasets/DHF1K/val_images/'
        # videos_test_path = '../DHF1K/val_images/'
        videos_test_path = '/home/natnael/Documents/datasets/DHF1K/val_images/'
        videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]

        # for i in videos:
        #     print(i)
        videos.sort()

        nb_videos_test = len(videos)
        # print(videos[99])

        m = Model(inputs=x, outputs=acl_vgg(x, stateful))
        print("Loading ACL weights")
        # m = Model(inputs=x, outputs=acl_vgg(x, stateful))
        # print("Loading ACL weights")

        m.load_weights('ACL.h5')
        for i in range(25, nb_videos_test):
            # print(videos[i])
            # print(videos[i])

            output_folder = videos[i] + '/saliency/'
            # print(output_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            images_names = [f for f in os.listdir(videos[i] +frames_path) if
                            f.endswith(('.jpg', '.jpeg', '.png'))]


            images_names.sort()

            print(len(images_names), "Image count")
            # print(images_names[0])
            # print(len(images_names))


            print("Predicting saliency maps for " + videos[i])
            prediction = m.predict_generator(get_test(video_test_path=videos[i]),
                                             max(ceil(len(images_names) / num_frames), 2))
            predictions = prediction[0]

            for j in range(len(images_names)):
                original_image = cv.imread(videos[i] + frames_path + images_names[j])
                x, y = divmod(j, num_frames)
                res = postprocess_predictions(predictions[x, y, :, :, 0], original_image.shape[0],
                                              original_image.shape[1])

                cv.imwrite(output_folder + '%s' % images_names[j], res.astype(int))
                # imsave(output_folder + '%s' % images_names[j], res.astype(int))
            m.reset_states()
    else:
        raise NotImplementedError
