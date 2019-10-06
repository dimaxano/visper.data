import cv2
from .object_tracking import *
import dlib
import time

#pylint: disable=no-member

class App(object):
    """
    Object tracking demo using video capture from webcam.

    Press 'm' to cycle through different methods
    Press 'f' to re-detect face
    Pres ESC to quit
    """

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.face_detector = dlib.get_frontal_face_detector()
        self.methods = ['MOSSE', 'DLIB', 'BOOSTING', 'MIL', 'KCF', 'TLD',
                        'MEDIANFLOW', 'CSRT']

    def run(self):
        blue, red, green = (255, 0, 0), (0, 0, 255), (0,  255, 0)

        tracker = None
        last_box = None
        method_idx = 0

        while True:
            frame = self.next_frame()

            if tracker:
                box = tracker.track_next(frame)

                if box:
                    last_box = box
                    frame = cv2.rectangle(frame, box[:2], box[2:4], blue, 2)
                elif last_box:
                    frame = cv2.rectangle(frame, last_box[:2],
                                          last_box[2:4], red, 2)
            else:
                face = self.detect_face(frame)
                if face:
                    tracker = self.create_tracker(method_idx)
                    tracker.init(frame, face)
                    frame = cv2.rectangle(frame, face[:2], face[2:4], green, 2)

            frame = cv2.putText(frame, self.methods[method_idx],
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, green)

            cv2.imshow('frame', frame)

            ch = cv2.waitKey(10)

            if ch == 27:
                break
            elif ch == ord('m') or ch == ord('f'):
                last_box = None
                tracker = None
                if ch == ord('m'):
                    method_idx = (method_idx + 1) % len(self.methods)

        cv2.destroyAllWindows()

    def create_tracker(self, idx):
        method = self.methods[idx]
        if method == 'DLIB':
            return DlibTracker()
        else:
            return OpenCvTracker(method)

    def next_frame(self):
        _, frame = self.capture.read()
        # frame = cv2.resize(frame, cv2.Size(), 0.5, 0.5)
        return frame

    def detect_face(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(frame_gray, 0)

        if not faces:
            return None
        face = faces[0]
        return (face.left(), face.top(), face.right(), face.bottom())


if __name__ == "__main__":
    App().run()
