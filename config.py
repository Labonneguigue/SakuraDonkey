""" 
CAR CONFIG 

This file is read by your car application's manage.py script to change the car
performance. 

EXMAPLE
-----------
import dk
cfg = dk.load_config(config_path='~/d2/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os
import platform

#OS
OS = platform.system()
MAC_OS = 'Darwin'
LINUX_OS = 'Linux'

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = 100000

#CAMERA
CAMERA_RESOLUTION = (120, 160) #(height, width)
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_ID = "carla"
'''
The different cameras available are the regular raspberry pi
camera, the raspberry pi camera with a wide angle and the camera
mounted in Carla (Udacity's self driving car) from which
recordings have been made.
For each camera, the camera matrix and distance coefficients
must be computed for having an accurate representation of the
world.
-   CAMERA_ID = "pi_cam"
-   CAMERA_ID = "pi_cam_wide"
-   CAMERA_ID = "carla"
'''

#STEERING
STEERING_CHANNEL = 0
STEERING_LEFT_PWM = 490
STEERING_RIGHT_PWM = 340

#THROTTLE
THROTTLE_CHANNEL = 1
THROTTLE_FORWARD_PWM = 400
THROTTLE_STOPPED_PWM = 360
THROTTLE_REVERSE_PWM = 310

#TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8


#JOYSTICK
USE_JOYSTICK_AS_DEFAULT = False
JOYSTICK_MAX_THROTTLE = 0.25
JOYSTICK_STEERING_SCALE = 1.0
AUTO_RECORD_ON_THROTTLE = True

#EXTERNAL PARTS
LANE_DETECTOR = True

# LANE DETECTOR CONFIGURATION
LD_PARAMETERS = { 'orig_points_x' : (575, 705, 1127, 203),#(617, 660, 1125, 188),
                  'orig_points_y' : 460, #434,
                  's_thresh'      : (170, 255),
                  'sx_thresh'     : (20, 100),
                  'l_white_thresh': (200, 255),
                  'margin'        : 100,
                  'detection_distance' : 0,
                  'videofile_in'  : '../videos/project_video.mp4',
                  'videofile_out' : '../processed_videos/project_video.mp4',
               #    'videofile_in'  : 'challenge_video.mp4',
               #    'videofile_out' : 'output_videos/challenge_video.mp4',
                  'output_video_as_images' : 0,
                  'debug_output'  : 0
}