#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    fast.py [--input=<image>] [--output=<folder>]

Options:
    -h --help       Show this screen.
    --input FILE    Input image
    --output FOLDER Output folder to store the results
"""

from docopt import docopt
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time

def fast(image, folder):

    img = cv2.imread(image,0)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    start = time.time()
    kp = fast.detect(img,None)
    elapsed = time.time() - start
    print("Detection of the features with nonmaxSuppression took ", elapsed*1000., " ms")

    img2 = np.zeros_like(img)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

    # print(all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

    basename = os.path.splitext(os.path.basename(image))[0]
    cv2.imwrite(folder + basename +'_fast_true.png',img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    start = time.time()
    kp = fast.detect(img,None)
    elapsed = time.time() - start
    print("Detection of the features without nonmaxSuppression took ", elapsed*1000., " ms")

    print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

    img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

    path_to_img3 = folder + basename +'_fast_false.png'
    cv2.imwrite(path_to_img3,img3)
    print("Saved ", path_to_img3)


if __name__ == '__main__':
    args = docopt(__doc__)
    
    image = args['--input']
    print("Input image: ", image)
    folder = args['--output']
    print("Output folder: ", folder)

    fast(image, folder)