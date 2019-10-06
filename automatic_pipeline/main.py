import os
from os.path import join
from time import time

import cv2
from loguru import logger

from face_tracking.object_tracking import DlibTracker
from face_detection.face_detection import FaceDetector
from shot_boundary.shot_boundary import shot_boundary
from landmark_detection.landmark_detection import LandmarkDetection


from definitions import MODULE_PATH, DATA_PATH


def _load_test_data():
    link = "https://www.dropbox.com/s/qzs7vkjj4ckljnn/test_video.mp4"

    os.system(f"wget -P {DATA_PATH} {link}")


def main():
    video_path = join(DATA_PATH, "test_video.mp4")

    """
    Pipeline description

    1. Shots intervals detection
    2. For each interval:
        Face detection init

        for each frame
            IF The 1st frame: Face detection
            ELSE Face tracker start
        
            Landmarks detection
            Mouth region extraction
    """
    




    

if __name__ == "__main__":
    main()
