from .object_tracking import *
import numpy as np
import cv2
import pytest
import os

# methods = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE',
#            'CSRT']

max_delta = 4


def create_tracker():
    return OpenCvTracker('MOSSE')


def test_track_static_image_returns_same_box():
    obj = cv2.imread('./data/bird.jpg')

    frame = np.zeros((200, 200, 3)).astype('uint8')
    frame[1:obj.shape[0]+1, 1:obj.shape[1]+1] = obj

    box = (1, 1) + obj.shape[:2]

    framecount = 25
    tracked_boxes = create_tracker().track([frame]*framecount, box)

    assert_same_boxes(tracked_boxes, [box]*framecount, delta=max_delta)


@pytest.mark.parametrize('shift_x, shift_y', [(0, 0), (0, 1), (1, 0),
                                              (0.2, 0.2), (1, 3), (3, 3),
                                              (7, 7)])
def test_track_linearly_moving_bird(shift_x, shift_y):

    background = cv2.imread('./data/graf1.png')
    obj = cv2.imread('./data/bird.jpg')

    frames, boxes = generate_scene(background, obj, shift_x, shift_y)
    tracked_boxes = create_tracker().track(frames, boxes[0])

    try:
        assert_same_boxes(tracked_boxes, boxes, delta=max_delta)
    except:
        render_video(frames, tracked_boxes, boxes,
                     f'video-s{shift_x}-{shift_y}.mp4')
        raise


@pytest.mark.parametrize("overlap", [0.05, 0.1])
def test_track_linearly_moving_object_with_occlusion(overlap):
    background = cv2.imread('./data/graf1.png')
    obj = cv2.imread('./data/bird.jpg')

    frames, boxes = generate_scene(background, obj, 1, 1,
                                   occlusion_overlap=overlap)
    tracked_boxes = create_tracker().track(frames, boxes[0])

    try:
        assert_same_boxes(tracked_boxes, boxes, delta=max_delta)
    except:
        render_video(frames, tracked_boxes, boxes, f'video-o{overlap}.mp4')
        raise


def generate_scene(background, obj, shift_x=0, shift_y=0,
                   occlusion_overlap=None, framecount=75):
    frames = []
    boxes = []

    starting_box = (1, 1) + obj.shape[:2]

    if occlusion_overlap:
        max_distance = int((framecount-1)*shift_x)
        occlusion_x = int(starting_box[0]+max_distance*0.75)
        occlusion_w = int(obj.shape[1]*occlusion_overlap)

    for i in range(framecount):
        currentBox = (starting_box[0]+int(i*shift_x),
                      starting_box[1]+int(i*shift_y),
                      starting_box[2]+int(i*shift_x),
                      starting_box[3]+int(i*shift_y))
        frame = background.copy()
        frame[currentBox[1]:currentBox[3]+1,
              currentBox[0]:currentBox[2]+1] = obj

        if occlusion_overlap:
            frame[:, occlusion_x:occlusion_x+occlusion_w+1] = (255, 255, 0)

        frames.append(frame)
        boxes.append(currentBox)

    return frames, boxes


def assert_same_boxes(tracked_boxes, expected_boxes, delta=0.1):
    assert tracked_boxes is not None, "Should not be None"
    assert type(tracked_boxes) is list, "Should be list"
    assert len(tracked_boxes), "Should not be empty"
    for i in range(len(expected_boxes)):
        assert tracked_boxes[i] is not None, f'Tracking lost on frame {i}'
        assert rect_diff(tracked_boxes[i], expected_boxes[i]) <= delta, \
            f'Mismatch on frame {i}'


def rect_diff(x, y):
    return max(abs(a-b) for a, b in zip(x, y))


def render_video(frames, aboxes, eboxes, name='video.mp4', folder='output'):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    wh = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(os.path.join(folder, name), fourcc, 29, wh)
    blue, green = (255, 0, 0), (0, 255, 0)
    for frame, abox, ebox in zip(frames, aboxes, eboxes):
        if abox:
            abox = tuple(int(x)for x in abox)
            frame = cv2.rectangle(frame, abox[:2], abox[2:4], blue, 2)

        ebox = tuple(int(x)for x in ebox)
        frame = cv2.rectangle(frame, ebox[:2], ebox[2:4], green, 1)
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
