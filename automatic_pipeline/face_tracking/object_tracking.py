import cv2
import dlib

#pylint: disable=no-member

class ObjectTracker(object):
    """ Abstract class for object tracking """

    def track(self, frames, bbox):
        """ Tracks the object within the given frames staring from given bounding box
        Parameters:
            frames : list[numpy.ndarray]
                     List of frames to process
            bbox : tuple
                   coords (x1,y1,x2,y2) of staring bounding box containing
                   tracked object
        Returns:
            list[tuple]: list of either bbox coordinates of tracked object
                         or None if tracking failed for the respective frame
        """

        self.init(frames[0], bbox)
        return [self.track_next(x) for x in frames]

    def init(self, img, bbox):
        pass

    def track_next(self, img):
        pass

    def _to_int(self, box):
        return tuple(int(x) for x in box)


class DlibTracker(ObjectTracker):
    """ DLib implementation of object tracking """
    def __init__(self):
        self.tracker = dlib.correlation_tracker()

    def init(self, img, bbox):
        rect = dlib.rectangle(*bbox)
        self.tracker.start_track(img, rect)

    def track_next(self, img):
        self.tracker.update(img)
        rect = self.tracker.get_position()
        return self._to_int((rect.left(), rect.top(),
                             rect.right(), rect.bottom()))


class OpenCvTracker(ObjectTracker):
    """ Wraps different object tracking methods available in OpenCV """

    def __init__(self, method='MOSSE'):
        self.tracker = self._create_tracker(method)

    def init(self, img, bbox):
        h, w = bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
        self.tracker.init(img, bbox[:2] + (h, w))

    def track_next(self, img):
        ok, rect = self.tracker.update(img)
        if not ok:
            return None

        return self._to_int((rect[0], rect[1],
                             rect[0]+rect[2]-1, rect[1]+rect[3]-1))

    def _create_tracker(self, method):
        if method == 'BOOSTING':
            return cv2.TrackerBoosting_create()
        if method == 'MIL':
            return cv2.TrackerMIL_create()
        if method == 'KCF':
            return cv2.TrackerKCF_create()
        if method == 'TLD':
            return cv2.TrackerTLD_create()
        if method == 'MEDIANFLOW':
            return cv2.TrackerMedianFlow_create()
        if method == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        if method == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        if method == "CSRT":
            return cv2.TrackerCSRT_create()
