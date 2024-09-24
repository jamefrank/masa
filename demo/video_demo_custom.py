'''
Author: jamefrank 42687222+jamefrank@users.noreply.github.com
Date: 2024-09-20 13:57:17
LastEditors: jamefrank 42687222+jamefrank@users.noreply.github.com
LastEditTime: 2024-09-20 15:43:36
FilePath: /masa/demo/video_demo_custom.py
Description:  测试自定义开放词汇跟踪

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''
import os, sys

from sympy import true
sys.path.insert(0, os.getcwd())

import gc
import cv2
import json
import glob
import torch
from typing import List
from tqdm import tqdm
from masa.apis import init_masa, inference_masa, build_test_pipeline
from dataclasses import dataclass
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmcv.ops.nms import batched_nms
from mmdet.registry import VISUALIZERS
from torch.multiprocessing import Pool, set_start_method

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

@dataclass
class YOLODETRES:
    img_path: str
    dets: DetDataSample
    pass

TYPE_DICT = {
    "monitor": 0,
    "person": 1
}

def _load_dets_json(json_file:str)->DetDataSample:
    det = DetDataSample()
    img_meta = dict(img_shape=(640, 480), pad_shape=(640, 480))
    pred_instances = InstanceData(metainfo=img_meta)
    bboxes = []
    labels = []
    scores = []
    with open(json_file, 'r') as file:
        data = json.load(file)
        if "objects" in data and len(data["objects"]):
            for obj in data["objects"]:
                bboxes.append(obj['box'])
                labels.append(TYPE_DICT[obj['category']])
                scores.append(obj['confidence']) 
    pred_instances.bboxes = torch.tensor(bboxes)
    pred_instances.labels = torch.tensor(labels)
    pred_instances.scores = torch.tensor(scores)
    det.pred_instances = pred_instances
    
    return det

def _load_yoloworld_results(dir:str)->List[YOLODETRES]:
    #
    jpg_files = glob.glob(os.path.join(dir, "*.jpg"))
    jpg_files = sorted(jpg_files)
    res_files = [file.replace(".jpg", "_bbox.json") for file in jpg_files]
    
    num_total = len(jpg_files)
    print(f"Total {num_total} images")
    
    dets_all = []
    for i in range(num_total):
        dets_frame = YOLODETRES(
            img_path=jpg_files[i],
            dets=_load_dets_json(res_files[i])
        )
        dets_all.append(dets_frame)
        pass
    
    return dets_all

def visualize_frame(score_thr, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=score_thr,
        fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame

def main():
    masa_config = "configs/masa-one/masa_r50_plug_and_play.py"
    masa_checkpoint = "saved_models/masa_models/masa_r50.pth"
    device = "cuda:0"
    score_thr = 0.1
    output_dir = "demo_outputs/yoloworld.mp4"
    
    # load yolo world results
    yolo_det_dir = "/home/frank/data/yoloworld_v2_s_monitor/mods"
    dets_all = _load_yoloworld_results(yolo_det_dir)
    
    #
    masa_model = init_masa(masa_config, masa_checkpoint, device=device)   
    masa_test_pipeline = build_test_pipeline(masa_model.cfg)
    masa_model.cfg.visualizer['texts'] = tuple(TYPE_DICT.keys())
    masa_model.cfg.visualizer['save_dir'] = None
    masa_model.cfg.visualizer['line_width'] = 5
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)
    
    #
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_dir, fourcc, 10, (640, 480))
    
    #
    frames = []
    fps_list = []
    instances_list = []
    for frame_idx, dets_frame in tqdm(enumerate(dets_all),total=len(dets_all)):
        frame = cv2.imread(dets_frame.img_path)
        result = dets_frame.dets
        det_bboxes, keep_idx = batched_nms(boxes=result.pred_instances.bboxes,
                                               scores=result.pred_instances.scores,
                                               idxs=result.pred_instances.labels,
                                               class_agnostic=False,
                                               nms_cfg=dict(type='nms',
                                                            iou_threshold=0.5,
                                                            class_agnostic=False,
                                                            split_thr=100000))
        det_bboxes = torch.cat([det_bboxes, result.pred_instances.scores[keep_idx].unsqueeze(1)], dim=1)
        det_labels = result.pred_instances.labels[keep_idx]
        track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=len(dets_all),
                                          test_pipeline=masa_test_pipeline,
                                          det_bboxes=det_bboxes,
                                          det_labels=det_labels,
                                          fp16=False,
                                          show_fps=False)
        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to('cpu'))
        frames.append(frame)

        
    print('Start to visualize the results...')
    num_cores = max(1, min(os.cpu_count() - 1, 16))
    print('Using {} cores for visualization'.format(num_cores))

    with Pool(processes=num_cores) as pool:
        frames = pool.starmap(
            visualize_frame, [(score_thr, visualizer, frame, track_result.to('cpu'), idx) for idx, (frame, track_result) in
                                enumerate(zip(frames, instances_list))]
        )

    for frame in frames:
        video_writer.write(frame[:, :, ::-1])

    video_writer.release()
 
 
if __name__ == '__main__':
    main()
    pass
