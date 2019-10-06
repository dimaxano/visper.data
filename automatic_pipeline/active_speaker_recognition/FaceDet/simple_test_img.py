from face_detection import FaceDetector
import argparse
import os
from os.path import join 
from definitions import MODULE_PATH, IMG_PATH, PROTOTXT_PATH, WEIGHTS_PATH, RESULTS_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="path to image", default=IMG_PATH)
parser.add_argument("--prototxt", help="path to prototxt", default=PROTOTXT_PATH)
parser.add_argument("--weights", help="path to .caffemodel", default=WEIGHTS_PATH)
parser.add_argument("--save_dir", help="where to save result image", default=RESULTS_PATH)
parser.add_argument("--threshold", help="confidence threshold", default=0.4)
args = parser.parse_args()

image = args.image
prototxt = args.prototxt
weights = args.weights
save_dir = args.save_dir
threshold = args.threshold


if __name__ == "__main__": 
    face_detector = FaceDetector(save_dir=RESULTS_PATH)
    face_detector.initialize()

    face_detector.detect([image])