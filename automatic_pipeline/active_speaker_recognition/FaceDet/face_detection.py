import numpy as np
import argparse
import cv2
import os
from os.path import isfile, join, dirname, abspath
from definitions import MODULE_PATH, WEIGHTS_PATH, PROTOTXT_PATH

# pylint: disable=no-member


WEIGHTS_FILE_PATH = join(WEIGHTS_PATH, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def _load_weights(save_dir=WEIGHTS_PATH):
    """
        Loading weights for OpenCV face recognition model
    """

    # path to the weights file
    path = ("https://raw.githubusercontent.com/"
            "opencv/opencv_3rdparty/"
            "dnn_samples_face_detector_20180205_fp16/"
            "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    os.system(f"wget -P {save_dir} {path}")


class FaceDetector():
    def __init__(self, prototxt=PROTOTXT_PATH,
                        weights=WEIGHTS_FILE_PATH,
                        module_path=MODULE_PATH,
                        threshold=0.5,
                        save_dir=None):

        self.prototxt = prototxt
        self.module_path = module_path
        self.weights = weights
        self.threshold = threshold
        self.save_dir = save_dir

        self.input_type = None
        self.model = None


    def initialize(self):
        """
            Loading model into RAM
        """
        if not isfile(self.weights):
            dir_name = dirname(self.weights)
            self._load_weights(dir_name)

        print("[INFO] loading model...")
        self.model = cv2.dnn.readNetFromCaffe(self.prototxt, self.weights)


    def detect(self, input_data):
        """
            Face detection method, returns
        """
        self.input_type = "path" if isinstance(input_data[0], str) else "image"
        model = self.model

        out_dict = dict()
        for img_idx, img in enumerate(input_data):
            if self.input_type == "path":
                if isfile(img):
                    image = cv2.imread(img)
                else:
                    raise FileExistsError(f"{img}")
            else:
                image = img

            # constructing image blob of size 300x300 and normalizing it
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

            print("[INFO] computing object detections...")
            model.setInput(blob)
            detections = model.forward()

            out = list()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > self.threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box of the face along with the associated
                    # probability
                    if self.save_dir is None:
                        out.append([startX, startY, endX, endY])
                    else:
                        text = "{:.2f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            if self.save_dir is None:
                out_dict[img_idx] = out
            else:
                save_path = join(self.save_dir, f"{i}.png")
                cv2.imwrite(save_path, image)

        if self.save_dir is None:
            return out_dict


    def _load_weights(self, save_dir=WEIGHTS_PATH):
        """
            Loading weights for OpenCV face recognition model
        """

        # path to the weights file
        path = ("https://raw.githubusercontent.com/"
                "opencv/opencv_3rdparty/"
                "dnn_samples_face_detector_20180205_fp16/"
                "res10_300x300_ssd_iter_140000_fp16.caffemodel")

        os.system(f"wget -P {save_dir} {path}")
