import numpy as np

down_ratio = 4
scale = np.array([300, 300])
input_w, input_h = (300, 300)

# voc_input_h, voc_input_w = (512, 512)
voc_input_h, voc_input_w = (320, 320)

box_center = False
center_scope = False

point_num = 128
gt_point_num = 128

ro = 4

segm_or_bbox = 'segm'

