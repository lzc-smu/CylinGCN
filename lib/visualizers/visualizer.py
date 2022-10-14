from lib.utils import img_utils
from lib.utils.network import config_model
import matplotlib.pyplot as plt
from lib.config import cfg
import numpy as np
import torch
from itertools import cycle
import os
import cv2


class Visualizer:
    def visualize_testing(self, output, batch):
        img = batch['inp'][0]
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex     # rgb中选最后一个框
        ex = ex.detach().cpu().numpy() * config_model.down_ratio

        for i in range(img.size(0)):
            inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(torch.cat((img[[i]],img[[i]],img[[i]]), dim=0),
                                                                 img.mean().item(), img.std().item()).permute(1, 2, 0))
            ex_ = ex[0, [i], ..., 1:]

            fig, ax = plt.subplots(1, figsize=(10, 10))
            fig.tight_layout()
            ax.axis('off')
            ax.imshow(inp)

            colors = np.array([[40, 150, 40],
                               [40, 150, 40]]) / 255.
            np.random.shuffle(colors)
            colors = cycle(colors)
            for j in range(len(ex_)):
                color = next(colors).tolist()
                poly = ex_[j]
                poly = np.append(poly, [poly[0]], axis=0)   # 首尾相接
                ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=5)

            plt.savefig(os.path.join(cfg.test.save_dir + 'contour/fig{}.jpg'.format(i)))
            plt.show()

            mask = np.zeros((batch['inp'].shape[2], batch['inp'].shape[3]), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(ex_[0]).astype(int)], 255)
            cv2.imwrite(os.path.join(cfg.test.save_dir + 'mask/{}.jpg'.format(i + 1)), mask)

    def visualize(self, output, batch):
        self.visualize_testing(output, batch)