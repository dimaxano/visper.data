from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import cv2


class FramesManager:
    """ Mock class wrapping frames iteration for SceneManager"""

    def __init__(self, frames=None, fps=25):
        """ Init
        Args:
            frames (:obj:`list` of :obj:`numpy.ndarray`): List of video frames.
            fps (int, optional): FPS of source video.
        """
        self.frames = frames
        self._frame_length = len(frames)
        self.fps = fps
        self.cur_index = 0

    def read(self):
        try:
            cur_frame = self.frames[self.cur_index]
            self.cur_index += 1
            return True, cur_frame
        except IndexError:
            return False, []

    def get_base_timecode(self):
        return FrameTimecode(timecode=0, fps=self.fps)

    def set_downscale_factor(self):
        pass

    def start(self):
        pass

    def get(self, capture_prop, index=None):
        """ Get (cv2.VideoCapture method) - obtains capture properties from the current
        VideoCapture object in use.  Index represents the same index as the original
        video_files list passed to the constructor.  Getting/setting the position (POS)
        properties has no effect; seeking is implemented using VideoDecoder methods.
        Note that getting the property CAP_PROP_FRAME_COUNT will return the integer sum of
        the frame count for all VideoCapture objects if index is not specified (or is None),
        otherwise the frame count for the given VideoCapture index is returned instead.
        Arguments:
            capture_prop: OpenCV VideoCapture property to get (i.e. CAP_PROP_FPS).
            index (int, optional): Index in file_list of capture to get property from (default
                is zero). Index is not checked and will raise exception if out of bounds.
        Returns:
            float: Return value from calling get(property) on the VideoCapture object.
        """
        if capture_prop == cv2.CAP_PROP_FRAME_COUNT and index is None:
            return self._frame_length
        elif capture_prop == cv2.CAP_PROP_POS_FRAMES:
            return self.cur_index


def _get_trans_frames(video_manager):
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    return [s[1].frame_num for s in scene_list]


def _get_intervals(bound_frames):
    """
        Returns list with shots intervals

        :param trans_frame_idxs: transition frames indexes
    """
    bound_frames.insert(0, 0)

    intervals = [(bound_frames[i], bound_frames[i+1]) for i in range(0, len(bound_frames)-1)]
    return intervals

def shot_boundary(input_data, params=None): # TODO: make it return frame intervals
    """Determining change of scenes
    Arguments:
        input_data (list[numpy.ndarray] or str): Data to process, list of frames or path to video
        params (dict): Parameters for Managers
    Returns:
        list[int]: Indices of frames where scene changes
    """
    manager = None
    if type(input_data) == list:
        if str(type(input_data[0])) == "<class 'numpy.ndarray'>":
            manager = FramesManager(input_data)
    elif type(input_data) == str:
        manager = VideoManager([input_data])
    if not manager:
        raise ValueError('Wrong input')
    trans_frames = _get_trans_frames(manager)

    intervals = _get_intervals(trans_frames)
    return intervals


