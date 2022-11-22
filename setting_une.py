'''
Preprocessing - transform temporal information to spatial information - aggregate results and labels -
(feed orignal data as an input and preprocessed one as a residual input | or | feed preprocessed data
 residual).
'''
from __future__ import print_function

import collections
import os
from collections import deque
import sys

import cv2 as cv
import time

import numpy as np








# np.max() is pixel element wise maxima of images

# Video 7 shows two important things, the need to have scene detector. spliter, and the applicability of this technique in general

class Self_regulatory_roi:
    def __init__(self):
        self.counter = 1
        self.firstFrame = None
        # self.occupation = rospy.Publisher('/opencog/roomoccupation', String, queue_size=30)
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/usb_cam_node/image_raw",Image, self.callback)

    def get_video(self):

        d = deque()
        for i in range(1,601):
            PREFIX = '000'
            VIDEO = (PREFIX + str(i))[-3:]
            # print(VIDEO)

            cap = cv.VideoCapture('../DHF1K/Videos/train_video/' + VIDEO + '.AVI')
            # cap = cv.VideoCapture('../DHF1K/Videos/validation_video/' + VIDEO + '.AVI')
            # get total number of frames
            totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            # print(totalFrames)

            if (cap.isOpened() == False):
                print("Error opening video stream or file")
            count = 1
            # Read until video is completed
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    # cv.imshow("Frame", frame)
                    # get the number of frames here
                    # accumulate all frames in a video to a queue + implement the dequeImp fun here till count = cap size
                    if (count <= totalFrames):
                        d.append(frame)
                    else:
                        break

                    # queue = self.dequeImp(count, frame)
                    # self.display(queue)
                    count = count + 1
                else:
                    break

            # print(len(d), totalFrames)
            queue = self.dequeImp(len(d), d)
            d.clear()

            self.write_to_file(queue, VIDEO)

            cap.release()

        # Use this to iterate over batch

    def write_to_file(self, queue, VIDEO):
        # img_list = list(collections.deque(queue))
        path = '../DHF1K/residual_train/' + VIDEO + '/'
        # path = '../DHF1K/residual_val/' + VIDEO + '/'
        isExist = os.path.exists(path)
        prefix = "0000"

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
        # print(len(queue))
        for i in range(len(queue)):
            img_name = prefix + str(i + 1)
            img_name = img_name[-4:]
            cv.imwrite(path + img_name + '.png', queue[i])
        print("Finished Writing, ", VIDEO)

        # cv.imwrite(PATH + name[-4:]+'.png',queue.popleft())
        # cv.im
        # print(path + img_name)

    def process_X(self, small_batch):
        # print(len(small_batch))
        frame_current = cv.cvtColor(small_batch[0], cv.COLOR_RGB2GRAY)
        frame_past = cv.cvtColor(small_batch[1], cv.COLOR_RGB2GRAY)
        frame_next = cv.cvtColor(small_batch[2], cv.COLOR_RGB2GRAY)

        delta_future = cv.absdiff(frame_next, frame_current)
        delta_past = cv.absdiff(frame_past, frame_current)

        # bitwise or of temproal differences
        img_bwo = cv.bitwise_or(delta_past, delta_future)

        # Dilating and binarizing temporal differences
        ret, thresh = cv.threshold(img_bwo, 25, 255, cv.THRESH_BINARY)
        dilate_frame = cv.dilate(thresh, None, iterations=2)

        # Is it required: ? to make more neighbour similarity
        # blured_copy = cv.GaussianBlur(frame_current, (21, 21), 0)

        # Shaked difference
        subtracter = np.copy(frame_current)[1:frame_current.shape[0], 1:frame_current.shape[1]]
        # print(subtracter.shape)

        # print(subtracter.shape)
        #
        # difference = cv.absdiff(frame_current[0:frame_current.shape[0] - 1, 0:frame_current.shape[1] - 1], subtracter)
        difference = frame_current[0:frame_current.shape[0] - 1, 0:frame_current.shape[1] - 1] - subtracter
        shape = (360, 640)
        plain = np.zeros(shape, dtype="uint8")
        plain[0:frame_current.shape[0] - 1, 0:frame_current.shape[1] - 1] = difference
        plain[frame_current.shape[0]:, frame_current.shape[1]:] = subtracter[subtracter.shape[0]:, subtracter.shape[1]:]

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv.erode(plain, kernel)

        # additive = cv.add(eroded ,dilate_frame)
        spatio_temporal_ready_frame = cv.add(eroded, dilate_frame)

        # eroded_max = cv.erode(maximize, kernel)


        r, g, b = cv.split(small_batch[0])
        r = np.maximum(r, spatio_temporal_ready_frame)
        g = np.maximum(g, spatio_temporal_ready_frame)
        b = np.maximum(b, spatio_temporal_ready_frame)

        spatio_temporal_ready_frame = cv.merge((r,g,b))

        # print(difference.shape)

        # cv.imshow("SelfShift Difference", difference)
        # cv.imshow(" Upsampled", plain)
        # cv.imshow(" Upsampled Eroded", eroded)










        # cv.imshow(" Spatio Temporal Frame ", spatio_temporal_ready_frame)
        # cv.imshow("Orignal Frame ", frame_current)








        # cv.imshow("Or-Union", dilate_frame)
        # cv.imshow("Maximized", maximize)

        # cv.imshow("eroded Max", eroded_max)
        cv.waitKey(1)

        return spatio_temporal_ready_frame

    def dequeImp(self, frameCount, queue):
        # if count < frameConstant:# In case of 42, around 4s queue accumulation time
        queue_to_list = list(collections.deque(queue))
        queue.clear()
        for i in range(len(queue_to_list)):

            try:
                pass
            except:
                raise Exception
            if i == 0:
                frame_current = queue_to_list[i]
                frame_ref_left = queue_to_list[i + 1]
                frame_ref_right = queue_to_list[i + 1]
                # cv.imshow("Frame 1", frame_current)
                # cv.waitKey(1)
            elif i == frameCount - 1:

                frame_current = queue_to_list[i]
                frame_ref_left = queue_to_list[i - 1]
                frame_ref_right = queue_to_list[i - 2]
            else:
                frame_current = queue_to_list[i]
                frame_ref_left = queue_to_list[i - 1]
                frame_ref_right = queue_to_list[i + 1]

            small_batch = [frame_current, frame_ref_left, frame_ref_right]

            queue.append(self.process_X(small_batch))
        # print(len(queue))
        return list(collections.deque(queue))

    # def dequeImp(self, count, currentState):
    #     # if count < frameConstant:# In case of 42, around 4s queue accumulation time
    #     if count <= frameConstant:  # In case of 42, around 4s queue accumulation time
    #         d.append(currentState)
    #         return d
    #     else:
    #         d.popleft()
    #         d.append(currentState)
    #         return d

    # def display(self, queue):
    #     counter = 1
    #     queue_to_list = list(collections.deque(queue))
    #
    #     scene_size = len(queue_to_list)
    #
    #     for i in range(len(queue_to_list) - 2):
    #         # print(i)
    #         if (len(queue_to_list) > 1):
    #             frame_past = cv.cvtColor(queue_to_list[i - 1], cv.COLOR_RGB2GRAY)
    #         else:
    #             if (len(queue_to_list) < 2):
    #                 frame_past = cv.cvtColor(queue_to_list[i], cv.COLOR_RGB2GRAY)  # change this dangerous
    #             else:
    #                 frame_past = cv.cvtColor(queue_to_list[i + 1], cv.COLOR_RGB2GRAY)
    #
    #         frame_current = cv.cvtColor(queue_to_list[i], cv.COLOR_RGB2GRAY)
    #
    #         cv.imshow("Frame Current", frame_current)
    #
    #         frame_next = cv.cvtColor(queue_to_list[(i + 1) % scene_size], cv.COLOR_RGB2GRAY)
    #         cv.imshow("Frame next", frame_next)
    #         # Background for subtraction
    #         frame_delta1 = cv.absdiff(frame_next, frame_current)
    #         frame_delta2 = cv.absdiff(frame_past, frame_current)
    #
    #         img_bwo = cv.bitwise_or(frame_delta2, frame_delta1)
    #
    #         # kernel = np.ones((2,2), np.uint8)
    #         # erode_frame = cv.erode(frame_delta, kernel)
    #
    #         ret, thresh = cv.threshold(img_bwo, 25, 255, cv.THRESH_BINARY)
    #
    #         dilate_frame = cv.dilate(thresh, None, iterations=2)
    #
    #         cv.imshow("Or-Union", dilate_frame)
    #
    #         counter = counter + 1
    #
    #         if cv.waitKey(25) & 0xFF == ord('q'):
    #             break

    # def callback(self,data):
    #   try:
    #       cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #       #Frame: Convert it To Gray and apply Gaussian blur
    #       gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #       gray = cv2.GaussianBlur(gray, (21, 21), 0)
    #
    #       # Initializing the First Reference Frame
    #       if self.firstFrame is None:
    #           self.firstFrame = gray
    # Subtracting the current frame from the reference Frame and converting it to Binary img based on the threshold
    # frameDelta = cv2.absdiff(self.firstFrame, gray)
    #     thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    #
    #     #Dilating the Image to fill merge tiny white areas.Also one can use erode to discover only large objects/moves.
    #     thresh = cv2.dilate(thresh, None, iterations=25)
    #     # find contours w. greater area;> cntArea
    #     (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)# discard/add the prefix _, before cnts, if it raise error
    #     activeSub=0 # number of moving objects
    #     for c in cnts:
    #         if cv2.contourArea(c) < cntArea:
    #             continue
    #         activeSub = activeSub + 1
    #         (x, y, w, h) = cv2.boundingRect(c)
    #         cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #     # Counting Active Locomotion
    #     state = 0
    #     if activeSub >= 0:
    #         state = int(activeSub * 0.75) # the more person exist the more impact
    #     else:
    #         state= -1
    #     Data = str(activeSub)
    #     # Get the state of the room in the past (frameConstant/10) seconds
    #     # trend = self.roomsilence(self.counter,state)
    #     self.counter = self.counter + 1
    #     self.occupation.publish(str(max(self.dequeImp(self.counter,activeSub))))
    #
    #     cv2.imshow("View", thresh)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         exit(0)
    # except CvBridgeError as e:
    #     print(e)


def main(args):
    global frameConstant
    global d
    global StartTime
    global pre
    global firstFrame
    global cntArea
    cntArea = 10000
    d = deque()
    frameConstant = 3  # was 42
    pre = 0
    firstFrame = None
    StartTime = time.time()

    preprocessor = Self_regulatory_roi()
    preprocessor.get_video()

    # rospy.init_node('RoomSilence', anonymous=True)
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
