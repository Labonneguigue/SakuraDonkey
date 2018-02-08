import donkeycar as dk
from donkeycar.vehicle import Vehicle
from lanedetection import LanesDetector

cfg = dk.load_config(config_path='./config.py')

print(cfg.LD_PARAMETERS["videofile_in"])

V = Vehicle()
laneDetector = LanesDetector(cfg.CAMERA_ID)

V.add(laneDetector)

V.start()