import cv2 as cv

import os
from os import listdir
from os.path import isfile, join


def extract(video_path, output_path):
    # cap = cv.VideoCapture()
    cap = cv.VideoCapture(video_path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    count = 1
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            prefix = '000'
            img_nm = prefix + str(count)
            img_nm = img_nm[-4:]

            # print(img_nm, count)

            # cv.imshow('Frame',frame)
            cv.imwrite(output_path + '/' + img_nm + '.png', frame)

            count = count + 1
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    print('Done with ' + video_path + ' ' + str(count - 1) + ' frames found')
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()


path = '/home/natnael/Documents/datasets/DHF1K/videos/'
videos = listdir(path)
videos.sort()
for f in videos:
    output_path = '/home/natnael/Documents/datasets/DHF1K/annotation/0' + f[:-4] + '/images'
    # print(image_path)
    # Create directory
    # print(output_path)

    isExist = os.path.exists(output_path)
    #
    if not isExist:
        #     # Create a new directory because it does not exist
        os.makedirs(output_path)
    extract(path + f, output_path)
