import argparse
import os
from os.path import join
import sys

import cv2
from loguru import logger

from .shot_boundary import shot_boundary
from .definitions import MODULE_PATH

# pylint: disable=no-member

#--------------------CONSTANTS--------------------
DEFAULT_VIDEO_PATH = join(MODULE_PATH, "data", "gates.mp4")
RESULTS_DIR = join(MODULE_PATH, "results")
OUT_FILE_NAME = "bound_frames.txt"


#--------------------CLI PARSER--------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="path to the video file",
                    default=DEFAULT_VIDEO_PATH)
args = parser.parse_args()


def _mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)

def get_video_frames(cap):
    frames = list()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
    
    return frames

def split_videos(shots_intervals, input_path, out_path): 
    
    cap = cv2.VideoCapture(input_path)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    SIZE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    frames = get_video_frames(cap)

    for idx, (start_fidx, end_fidx) in enumerate(shots_intervals):
        logger.info(f"video idx - {idx}")
        logger.info(f"{start_fidx}-{end_fidx}")
        logger.info(f"{end_fidx - start_fidx}")

        # output path (must be mp4, otherwize choose other codecs)
        save_path = join(out_path, f"{idx}.mp4") 
        writer = cv2.VideoWriter(save_path, FOURCC, FPS, SIZE)

        write_frames = frames[start_fidx:end_fidx]
        for frame in write_frames:
            writer.write(frame)
        
        writer.release()


if __name__ == "__main__":
    video_path = args.video_path
    shots_intervals = shot_boundary(video_path)

    results_dir = join(RESULTS_DIR, video_path.split('/')[-1].split('.')[0])
    _mkdirs(results_dir)

    split_videos(shots_intervals, video_path, results_dir)

