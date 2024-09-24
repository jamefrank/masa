'''
Author: jamefrank 42687222+jamefrank@users.noreply.github.com
Date: 2024-09-20 15:52:10
LastEditors: jamefrank 42687222+jamefrank@users.noreply.github.com
LastEditTime: 2024-09-20 16:21:39
FilePath: /masa/demo/get_dets_monogodb.py
Description:  从mongoDB中获取检测结果

yoloworld检测结果

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''
import os
import cv2
import json
from tqdm import tqdm
from task_common.io.cloud.find_frame_task import FindFrameTask, FindFrameParam

def main():
    src_database = 'db_dev'
    src_collection = 'novauto_ovd_test'
    output_dir = "/home/frank/data/yoloworld_v2_s_monitor/mods_0920"
    
    #
    os.makedirs(output_dir, exist_ok=True)
    
    #
    find_frame_task = FindFrameTask()
    frames = find_frame_task.execute([], FindFrameParam(
        database=src_database,
        collection=src_collection,
        sort=[('info.timestamp', 1)]
    ))
    
    #
    all_categorys = {}
    for frame in tqdm(frames, total=len(frames)):
        #
        img_path = frame.camera[0].frame[0].image
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_dir, frame.info.timestamp + '.jpg'), img)
        #
        dets_frame = {}
        dets_frame["ts"] = frame.info.timestamp
        dets_frame["objects"] = []
        dets_frame["width"] = frame.camera[0].param.width
        dets_frame["height"] = frame.camera[0].param.height
        for obj in frame.camera[0].frame[0].frame_label["gt_b"].label:
            x0 = obj.position_2d.x - obj.size_2d.x*0.5
            x1 = obj.position_2d.x + obj.size_2d.x*0.5
            y0 = obj.position_2d.y - obj.size_2d.y*0.5
            y1 = obj.position_2d.y + obj.size_2d.y*0.5
            score = obj.confidence
            category = obj.meta["type"]
            dets_frame["objects"].append(
                {
                    "box":[x0, y0, x1, y1],
                    "confidence":score,
                    "category":category
                }
            )
            if category not in all_categorys:
                all_categorys[category] = len(all_categorys)
        #
        with open(os.path.join(output_dir, frame.info.timestamp + '_bbox.json'), 'w') as f:
            json.dump(dets_frame, f)
    #
    with open(os.path.join(output_dir, 'all_categorys.json'), 'w') as f:
        json.dump(all_categorys, f)
    
    print(all_categorys)
    
    pass
 
 
if __name__ == '__main__':
    main()
    pass
