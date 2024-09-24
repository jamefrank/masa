'''
Author: jamefrank 42687222+jamefrank@users.noreply.github.com
Date: 2024-09-23 10:56:28
LastEditors: jamefrank 42687222+jamefrank@users.noreply.github.com
LastEditTime: 2024-09-23 11:00:55
FilePath: /masa/demo/mmdet_color_list.py
Description: 获取常见的颜色列表

Copyright (c) 2024 by Frank, All Rights Reserved. 
'''

from mmdet.visualization import DetLocalVisualizer

# 获取可视化器
visualizer = DetLocalVisualizer()

# 打印 dataset_meta 的内容
print(visualizer.dataset_meta)

# 获取颜色列表
color_list = visualizer.dataset_meta.get('palette', None)

# 打印颜色列表
for i, color in enumerate(color_list):
    print(f"Class {i}: {color}")
