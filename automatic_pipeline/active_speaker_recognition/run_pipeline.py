#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess
import numpy as np
import tensorflow as tf
import cv2
import scenedetect
import dlib  #FIXME not a original pipeline
from test import face_locations
from scipy.interpolate import interp1d
from utils import label_map_util
from scipy.io import wavfile
from scipy import signal

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

# FIXME - watch carefully for "min_track" - *minimum* nb of frames for facetrack duration
parser = argparse.ArgumentParser(description="FaceTracker")
parser.add_argument('--data_dir', type=str, default='/dev/shm', help='Output direcotry')
parser.add_argument('--videofile', type=str, default='', help='Input video file')
parser.add_argument('--reference', type=str, default='', help='Name of the video')
parser.add_argument('--crop_scale', type=float, default=0.5, help='Scale bounding box')
parser.add_argument('--min_track', type=int, default=24, help='Minimum facetrack duration')  # was 100
opt = parser.parse_args()

setattr(opt, 'avi_dir', os.path.join(opt.data_dir, 'pyavi'))
setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
    # Detector returns coordinates in relative position:  | Converter to get absolute positions:
    #
    # top left corner: (x, y) = (boxes[1], points[0])     | (int(boxes[3]*im_cols_x), int(boxes[2]*im_rows_y)
    # bottom right corner: (x, y) = (boxes[3], points[1]) | (int(boxes[3]*im_cols_x), int(boxes[2]*im_rows_y)
    # TODO: Not tested, but errors seem to be fixed
    xA = max(boxA[0], boxB[0])  # x_left
    yA = max(boxA[1], boxB[1])  # y_top
    xB = min(boxA[2], boxB[2])  # x_right
    yB = min(boxA[3], boxB[3])  # y_bottom

    # interArea = max(0, xB - xA) * max(0, yB - yA)  # FIXME - finction from orig. pipeline, probably gives error
    interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    # FIXME debug code below, turn on if needed
    # print('\n')
    # print(xA, yA, xB, yB)
    # print(interArea, float(boxAArea + boxBArea - interArea))
    # print(f'Current IoU = {iou}')
    return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========


def track_shot(opt, scenefaces):
    iouThres = 0.5    # Minimum IOU between consecutive face detections
    numFail = 2       # Number of missed detections allowed # was 3
    minSize = 0.05    # Minimum size of faces
    tracks = []

    while True:
        track = []
        for faces in scenefaces:
            for face in faces:
                if not track:  # track == []
                    track.append(face)
                    faces.remove(face)
                elif face[0] - track[-1][0] <= numFail:
                    iou = bb_intersection_over_union(face[1], track[-1][1])
                    if iou > iouThres:
                        track.append(face)
                        faces.remove(face)
                        continue
                else:
                    break

        if not track:  # track == []
            break
        elif len(track) > opt.min_track:
            framenum = np.array([f[0] for f in track])
            bboxes = np.array([np.array(f[1]) for f in track])
            frame_i = np.arange(framenum[0], framenum[-1]+1)
            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)

            print(bboxes_i[:, 3], bboxes_i[:, 1])
            print(np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
            if abs(np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > minSize:  # FIXME remove abs()
                tracks.append([frame_i, bboxes_i])
    print(tracks)
    return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========


def crop_video(opt, track, cropfile):
    cap = cv2.VideoCapture(os.path.join(opt.avi_dir, opt.reference, 'video.avi'))

    # FIXME: unknown error in line <<cap.set(1,track[0][0])>>
    total_frames = cap.get(7)

    cap.set(1, track[0][0])  # CHANGE THIS !!!

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  #https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
    vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, cap.get(5), (224, 224))

    fw = cap.get(3)
    fh = cap.get(4)

    dets = [[], [], []]

    for det in track[1]:
        dets[0].append(((det[3] - det[1]) * fw + (det[2] - det[0]) * fh) / 4) # H+W / 4
        dets[1].append((det[1] + det[3]) * fw / 2) # crop center x
        dets[2].append((det[0] + det[2]) * fh / 2) # crop center y

    # Smooth detections
    dets[0] = signal.medfilt(dets[0], kernel_size=5)
    dets[1] = signal.medfilt(dets[1], kernel_size=5)
    dets[2] = signal.medfilt(dets[2], kernel_size=7)

    for det in zip(*dets):
        cs = opt.crop_scale
        bs = det[0]             # Detection box size
        bsi = int(bs*(1+2*cs))  # Pad videos by this amount

        ret, frame = cap.read()

        frame = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(0, 0))
        my = det[2] + bsi  # BBox center Y
        mx = det[1] + bsi  # BBox center X

        face = frame[int(my-bs):int(my+bs*(1+2*cs)), int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

        vOut.write(cv2.resize(face, (224, 224)))

    audiotmp = os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')
    audiostart = track[0][0]/cap.get(5)
    audioend = (track[0][-1]+1)/cap.get(5)

    cap.release()
    vOut.release()

    # ========== CROP AUDIO FILE ==========
    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f %s" %
               (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), audiostart, audioend, audiotmp)) #-async 1
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    sample_rate, audio = wavfile.read(audiotmp)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile, audiotmp, cropfile))  #-async 1
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    print('Written %s' % cropfile)

    os.remove(cropfile + 't.avi')

    return [track, dets]


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './protos/frozen_inference_graph_face.pb'
    # print(f'\n\n\n\n!!!!!! {PATH_TO_CKPT}')

    # FIXME: seems unnecessary
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './protos/face_label_map.pbtxt'

    NUM_CLASSES = 2
    MIN_CONF = 0.3

    # FIXME: seems unnecessary
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # FIXME: unnecessary function?
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    cap = cv2.VideoCapture(os.path.join(opt.avi_dir, opt.reference, 'video.avi'))

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    dets = []

    # TODO implement dlib face and landmark detector
    # TODO Fix coordinate order in bboxes_list
    # frame_num = 0
    # print('>>> Running face detection')
    # while True:
    #     print(f'Processing frame {frame_num}')
    #
    #     ret, image = cap.read()
    #     if ret == 0:
    #         break
    #
    #     bboxes = face_locations(image, model='cnn', number_of_times_to_upsample=1)  #dlib.cnn_face_detection_model_v1(image)
    #     scores = [1. for _ in range(len(bboxes))]
    #
    #     dets.append([])
    #     for i in range(len(bboxes)):
    #         bboxes_list = list(bboxes[i])
    #         bboxes_list[0] = bboxes_list[0] / image.shape[0]
    #         bboxes_list[1] = bboxes_list[3] / image.shape[1]
    #         bboxes_list[2] = bboxes_list[2] / image.shape[1]
    #         bboxes_list[3] = bboxes_list[1] / image.shape[0]
    #
    #         dets[-1].append([frame_num, bboxes_list, scores[i]])
    #         print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
    #                                             frame_num, len(dets[-1]), 0))
    #     frame_num += 1

    # FIXME OLD FACE DETECTION PIPELINE
    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=detection_graph, config=config) as sess:
            frame_num = 0
            while True:
                ret, image = cap.read()
                if ret == 0:
                    break

                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection
                # Detector returns coordinates in relative position:
                # top left corner: (boxes[1], points[0]) |  (int(boxes[3]*im_cols_x), int(boxes[2]*im_rows_y)
                # bottom right corner: (boxes[3], points[1]) (int(boxes[3]*im_cols_x), int(boxes[2]*im_rows_y)

                start_time = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                elapsed_time = time.time() - start_time

                score = scores[0]

                dets.append([])
                for index in range(0, len(score)):
                  if score[index] > MIN_CONF:
                    print('\n', boxes[0][index].tolist())
                    dets[-1].append([frame_num, boxes[0][index].tolist(), score[index]])

                print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
                                                     frame_num, len(dets[-1]), (1/elapsed_time)))
                frame_num += 1

    cap.release()

    savepath = os.path.join(opt.work_dir, opt.reference, 'faces.pckl')

    with open(savepath, 'wb') as fil:
        pickle.dump(dets, fil)

    return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========


def scene_detect(opt):
    scene_list = []
    detector_list = [scenedetect.detectors.ContentDetector(threshold=24)]  # was 32 in original pipeline
    video_framerate, frames_read = scenedetect.detect_scenes_file(os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
                                                                  scene_list, detector_list)
    _save_to = os.path.join(opt.work_dir, opt.reference, 'scene.pckl')
    with open(_save_to, 'wb') as _fil:
        pickle.dump([frames_read, scene_list], _fil)
    print('%s - scenes detected %d from %d frames' %
          (os.path.join(opt.avi_dir, opt.reference, 'video.avi'), len(scene_list), frames_read))
    return [frames_read, scene_list]
    

# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========


if not(os.path.exists(os.path.join(opt.work_dir, opt.reference))):
    os.makedirs(os.path.join(opt.work_dir, opt.reference))

if not(os.path.exists(os.path.join(opt.crop_dir, opt.reference))):
    os.makedirs(os.path.join(opt.crop_dir, opt.reference))

if not(os.path.exists(os.path.join(opt.avi_dir, opt.reference))):
    os.makedirs(os.path.join(opt.avi_dir, opt.reference))

if not(os.path.exists(os.path.join(opt.tmp_dir, opt.reference))):
    os.makedirs(os.path.join(opt.tmp_dir, opt.reference))


# ========== ========== ========== ==========
# # MAIN PART
# ========== ========== ========== ==========

command = ("ffmpeg -y -i %s -qscale:v 4 -async 1 -r 25 -deinterlace %s" %
           (opt.videofile, os.path.join(opt.avi_dir, opt.reference, 'video.avi')))
# FIXME TURN BACK
output = subprocess.call(command, shell=True, stdout=None)
# FIXME TURN BACK
faces = inference_video(opt)
# FIXME TURN BACK
scene = scene_detect(opt)

# FIXME TURN OFF
# with open(os.path.join(opt.work_dir, opt.reference, 'scene.pckl'), 'rb') as fil:
#     scene = pickle.load(fil)
# FIXME TURN OFF
# with open(os.path.join(opt.work_dir, opt.reference, 'faces.pckl'), 'rb') as fil:
#     faces = pickle.load(fil)
print(scene)
scene[1] = scene[1] + [scene[0]]
print(scene)
prev_shot = 0
alltracks = []
vidtracks = []


for end_shot in scene[1]:
    if (len(faces) == scene[0]) and (end_shot - prev_shot >= opt.min_track):
        alltracks.extend(track_shot(opt, faces[prev_shot:end_shot-1]))
    prev_shot = end_shot

print(alltracks)

for ii, track in enumerate(alltracks):
    print(ii)
    vidtracks.append(crop_video(opt, track, os.path.join(opt.crop_dir, opt.reference, '%05d' % ii)))

savepath = os.path.join(opt.work_dir, opt.reference, 'tracks.pckl')

with open(savepath, 'wb') as fil:
    pickle.dump(vidtracks, fil)
