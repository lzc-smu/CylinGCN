import os
from lib.utils.network import prepare_utils, config_model, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, image_size):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.img_size = image_size

        self.coco = COCO(ann_file)
        self.anns = np.array(sorted(self.coco.getImgIds()))
        self.anns = self.anns[:int(self.anns.size/self.img_size)]


    def process_info(self, img_id, img_size):
        img_id = list(range(img_id*img_size, (img_id+1)*img_size, 1))
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = []
        for i in img_id:
            path.append(os.path.join(self.data_root, self.coco.loadImgs(int(i))[0]['file_name']))
        return anno, path, img_id


    def read_original_data(self, anno, path):
        firstiter = True
        for i in range(path.__len__()):
            img = cv2.imread(path[i])
            if firstiter:
                data = img[..., [0]]
                firstiter = False
            else:
                data = np.append(data, img[..., [0]], axis=2)
            contour = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        return data, contour


    def transform_original_data(self, contours, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        contours_ = []
        for contour in contours:
            polys = [poly.reshape(-1, 2) for poly in contour]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = prepare_utils.transform_polys(polys, trans_output, output_h, output_w)
            contours_.append(polys)
        return contours_


    def get_valid_polys(self, contours, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        contours_ = []
        for contour in contours:
            polys = prepare_utils.filter_tiny_polys(contour)
            polys = prepare_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            contours_.append(polys)
        return contours_


    def get_center(self, contours, hw):
        if self.split == 'train':
            center = prepare_utils.get_center(contours)
        else:
            center = np.array(hw, dtype=np.float32) / 2
        return center


    def prepare_match(self, poly, init_contour, can_init_contour, gt_contour, can_gt_contour,
                      inp_gt_contour, center, inp_out_hw, layer, rand):
        img_init = data_utils.init_uniform_cirsample(center, inp_out_hw, config_model.point_num, rand)
        img_gt= prepare_utils.uniformsample(poly, len(poly) * config_model.gt_point_num)
        img_gt = data_utils.init_gt_match(img_init, img_gt, center, config_model.point_num)
        interp_gt = prepare_utils.uniformsample(img_gt, len(img_gt) * config_model.gt_point_num)

        img_init = data_utils.add_deep_dim(img_init, layer)
        img_gt = data_utils.add_deep_dim(img_gt, layer)
        interp_gt = data_utils.add_deep_dim(interp_gt, layer)

        can_init = prepare_utils.img_poly_to_can_poly_3d(img_init)
        can_gt = prepare_utils.img_poly_to_can_poly_3d(img_gt)

        init_contour.append(img_init)
        can_init_contour.append(can_init)
        gt_contour.append(img_gt)
        can_gt_contour.append(can_gt)
        inp_gt_contour.append(interp_gt)


    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)


    def __getitem__(self, index):
        ann = self.anns[index]
        img_size = self.img_size

        anno, path, img_id = self.process_info(ann, img_size)
        img, contour = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, scale, inp_out_hw = prepare_utils.augment(img, self.split)

        contour = self.transform_original_data(contour, flipped, width, trans_output, inp_out_hw)
        contour = self.get_valid_polys(contour, inp_out_hw)
        center = self.get_center(contour, inp_out_hw[2:])

        if cfg.task == 'OCT':
            ct = np.array([[img.shape[0] / 2., img.shape[1] / 2.]], dtype=np.float32)
            center = data_utils.affine_transform(ct, trans_output)[0]

        init_contour = []
        can_init_contour = []
        gt_contour = []
        can_gt_contour = []
        inp_gt_contour = []
        r_rand = np.random.uniform(0.6, 0.8)
        if self.split != 'train':
            r_rand = cfg.test.radius_len


        for i in range(len(anno)):
            contour_ = contour[i]
            for j in range(len(contour_)):
                c = contour_[j]
                self.prepare_match(c, init_contour, can_init_contour, gt_contour, can_gt_contour, inp_gt_contour, center, inp_out_hw, i, r_rand)

        # Show initial contour
        # visualize_utils.visualize_contour(inp, gt_contour)
        # visualize_utils.visualize_contour(inp, init_contour)

        ret = {'inp': inp}
        evolution = {'ic': init_contour, 'cic': can_init_contour, 'gc': gt_contour, 'cgc': can_gt_contour, 'igc': inp_gt_contour}
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'img_size': img_size}
        ret.update(evolution)
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)