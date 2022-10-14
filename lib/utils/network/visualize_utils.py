import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import cycle
from lib.utils.network import config_model


def visualize_contour(batch, polys):
    ex = polys
    for i in range(batch.shape[0]):
        ex_ = ex[i]
        inp = np.concatenate((batch[[i], ...], batch[[i], ...], batch[[i], ...]), axis=0).transpose(1, 2, 0)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([[46, 160, 46],
                           [46, 160, 46]]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        color = next(colors).tolist()
        poly = ex_[:, 1:] * config_model.down_ratio
        poly = np.append(poly, [poly[0]], axis=0)
        ax.plot(poly[:, 0], poly[:, 1], color=color, marker='o', linewidth=2)
        plt.show()
