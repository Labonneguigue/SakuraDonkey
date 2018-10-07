#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import sys
import os
import cv2


class OpticalFlow(object):
    def __init__(self):
        # params for Shi-Tomasi corner detection
        self.feature_params = dict( maxCorners = 3000,
                                    qualityLevel = 0.01,
                                    minDistance = 7,
                                    blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                                maxLevel = 2,
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray = None
        self.p0 = None

    def compute(self, img):
        # Create a mask image for drawing purposes
        mask = np.zeros_like(img)
        # Conversion to gray scale image for detection
        frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if self.old_gray is not None:
            self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
            
            if p1 is None:
                self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
                p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = self.p0[st==1]
            
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv.line(mask, (a,b),(c,d),(0,128,0), 1)
                img = cv.circle(img,(a,b),5,(255,0,0),1)
            img = cv.add(img,mask)
            
            # Now update the previous frame and previous points
            self.p0 = good_new.reshape(-1,1,2)
        self.old_gray = frame_gray.copy()
        
        return img

class Renderer(object):
    def __init__(self):
        pass

    def display_frame(self, img):
        cv2.imshow('frame',img)
        k = cv2.waitKey(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s <video.mp4>" % sys.argv[0])
        exit(-1)
    cap = cv2.VideoCapture(sys.argv[1])
    # camera parameters
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    F = float(os.getenv("F", "525"))
    if os.getenv("SEEK") is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("SEEK")))

    if W > 1024:
        downscale = 1024.0/W
        F *= downscale
        H = int(H * downscale)
        W = 1024
    print("using camera %dx%d with F %f" % (W,H,F))

    of = OpticalFlow()
    renderer = Renderer()

    # camera intrinsics
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    Kinv = np.linalg.inv(K)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (W,H))
        if ret == True:
            frame = of.compute(frame)
            renderer.display_frame(frame)
        else:
            break

    cv.destroyAllWindows()
    cap.release()