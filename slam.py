#!/usr/bin/env python3
import time
import cv2
from extractor import Extractor
import numpy as np
import sys
import os

fe = Extractor()

def display_frame(img):
    cv2.imshow('frame',img)
    k = cv2.waitKey(1)

def process_frame(img):
    matches = fe.extract(img)

    print("%d matches" % (len(matches)))

    for pt1, pt2 in matches:
        u1,v1 = map(lambda x: int(round(x)), pt1)
        u2,v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))
    return img

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

    # camera intrinsics
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    Kinv = np.linalg.inv(K)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (W,H))
        if ret == True:
            frame = process_frame(frame)
            display_frame(frame)
        else:
            break


