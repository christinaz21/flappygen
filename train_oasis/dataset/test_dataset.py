"""
time profiling
pip install line-profiler
kernprof -l -v test_dataset.py
"""

import torch
import random
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
import ijson
from torchvision.io import read_video
# from train_oasis.utils import parse_VPT_action
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import concurrent.futures
import itertools


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.right.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.right.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

CAMERA_SCALER = 360.0 / 2400.0

def extract_keys(json_string, keys):
    """
    raw_decode allows you to lazily parse the JSON and stop as soon as the desired data is found.
    only return required keys not others.
    """
    decoder = json.JSONDecoder()
    parsed = decoder.raw_decode(json_string)
    data = parsed[0]  # This is the actual parsed data
    return {key: data[key] for key in keys if key in data}

@profile
def parse_VPT_action(line:str):
    # json_action = json.loads(line)
    # json_action = orjson.loads(line)
    json_action = extract_keys(line, ['mouse', 'keyboard'])

    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
    if 1 in mouse_buttons:
        env_action["use"] = 1
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1

    # convert to onehot
    one_hot = np.zeros(len(NOOP_ACTION)+1)
    one_hot[0] = env_action["ESC"]
    one_hot[1] = env_action["back"]
    one_hot[2] = env_action["drop"]
    one_hot[3] = env_action["forward"]
    one_hot[4] = env_action["hotbar.1"]
    one_hot[5] = env_action["hotbar.2"]
    one_hot[6] = env_action["hotbar.3"]
    one_hot[7] = env_action["hotbar.4"]
    one_hot[8] = env_action["hotbar.5"]
    one_hot[9] = env_action["hotbar.6"]
    one_hot[10] = env_action["hotbar.7"]
    one_hot[11] = env_action["hotbar.8"]
    one_hot[12] = env_action["hotbar.9"]
    one_hot[13] = env_action["inventory"]
    one_hot[14] = env_action["jump"]
    one_hot[15] = env_action["left"]
    one_hot[16] = env_action["right"]
    one_hot[17] = env_action["sneak"]
    one_hot[18] = env_action["sprint"]
    one_hot[19] = env_action["swapHands"]
    one_hot[20] = env_action["camera"][0]
    one_hot[21] = env_action["camera"][1]
    one_hot[22] = env_action["attack"]
    one_hot[23] = env_action["use"]
    one_hot[24] = env_action["pickItem"]

    return one_hot

def split_idx(idx, cum_clips_per_video):
    video_idx = np.argmax(cum_clips_per_video > idx)
    frame_idx = idx - np.pad(cum_clips_per_video, (1, 0))[video_idx]
    return video_idx, frame_idx
    
# def process_actions(action_path, frame_idx, n_frames):
#     with open(action_path, "r") as f:
#         lines = f.readlines()
#         print(len(lines))
#     if frame_idx == 0:
#         actions = [parse_VPT_action(lines[0])] + [
#             parse_VPT_action(line) for line in lines[frame_idx : frame_idx + n_frames - 1]
#         ]
#     else:
#         actions = [
#             parse_VPT_action(line) for line in lines[frame_idx - 1 : frame_idx + n_frames - 1]
#         ]
#     return np.array(actions)

# @profile
# def process_actions(action_path, frame_idx, n_frames):
#     actions = []
#     print('start_idx: ', frame_idx)

#     with open(action_path, "r") as f:
#         if frame_idx == 0:
#             actions.append(parse_VPT_action(lines[0]))
#             for i, line in enumerate(f):
#                 if i < frame_idx + n_frames - 1:
#                     actions.append(parse_VPT_action(line))
#                 else:
#                     break
#         else:
#             for i, line in enumerate(f):
#                 if i >= frame_idx - 1 and i < frame_idx + n_frames - 1:
#                     actions.append(parse_VPT_action(line))
#                 elif i >= frame_idx + n_frames - 1:
#                     break  # Stop reading once we've processed the necessary lines
#     return np.array(actions)

@profile
def process_actions(action_path, frame_idx, n_frames):
    actions = []
    start_idx = frame_idx - 1 if frame_idx > 0 else 0
    end_idx = frame_idx + n_frames - 1
    print('start_idx: ', start_idx)
    line_len = []
    with open(action_path, "r") as f:
        for line in itertools.islice(f, start_idx, end_idx):
            actions.append(parse_VPT_action(line))
            line_len.append(len(line))
        print('max line length: ', max(line_len))
        if frame_idx == 0:
            # Add the first line twice
            with open(action_path, "r") as f_first:
                actions.insert(0, parse_VPT_action(next(f_first)))

    return np.array(actions)

    
save_dir = "data/VPT"
metadata_path = os.path.join(save_dir, "metadata.json")
metadata = json.load(open(metadata_path, "r"))
split = 'training'
n_frames = 20
data_paths = [Path(x["file"]) for x in metadata[split]]
lengths = [x["length"] for x in metadata[split]]
clips_per_video = np.clip(np.array(lengths) - n_frames + 1, a_min=1, a_max=None).astype(
    np.int32
)
full_length = clips_per_video.sum()
cum_clips_per_video = np.cumsum(clips_per_video)
idx_remap = list(range(full_length))
random.shuffle(idx_remap)

def profile_code(idx_remap, cum_clips_per_video, data_paths, n_frames):
    times=[]
    for i in range(1000):
        print(i)
        idx = idx_remap[i]
        t1=time.time()
        t1 = time.time()
        file_idx, frame_idx = split_idx(idx, cum_clips_per_video)
        action_path = data_paths[file_idx]
        assert action_path.exists(), f"File {action_path} does not exist"
        video_path = action_path.with_suffix(".mp4")
        start = frame_idx / 20
        end = (frame_idx + n_frames - 1) / 20
        # video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
        # video = video.contiguous().numpy()
        actions = process_actions(action_path, frame_idx, n_frames)
        t2=time.time()
        times.append(t2-t1)
        print('time: ', t2-t1)
    print('avg time: ', np.mean(times))
            
profile_code(idx_remap, cum_clips_per_video, data_paths, n_frames)

# for i in range(1000):
#     print(i)
#     idx = idx_remap[i]
#     t1=time.time()
#     file_idx, frame_idx = split_idx(idx, cum_clips_per_video)
#     action_path = data_paths[file_idx]
#     assert action_path.exists(), f"File {action_path} does not exist"
#     video_path = action_path.with_suffix(".mp4")
#     start = frame_idx / 20
#     end = (frame_idx + n_frames - 1) / 20
#     # video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
#     # video = video.contiguous().numpy()
#     actions = process_actions(action_path, n_frames)
#     t2=time.time()
#     print('time: ', t2-t1)


