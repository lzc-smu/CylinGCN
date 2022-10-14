import torch.utils.data as data
import glob
import os
import scipy.io as io
import cv2
import numpy as np
from lib.utils.network import config_model
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers.visualizer import Visualizer


class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.ann = [0]

    def img_process(self):
        if os.path.isdir(cfg.test.demo_dir):
            self.imgs = glob.glob(os.path.join(cfg.test.demo_dir, '*'))
            self.imgs.sort(key=lambda x: int(x.replace(cfg.test.demo_dir, "").split('.')[0]))
        elif os.path.exists(cfg.test.demo_dir):
            self.imgs = [cfg.test.demo_dir]
        else:
            raise Exception('NO SUCH FILE')

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - np.mean(inp)) / np.std(inp)
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        self.img_process()
        firstiter = True
        for i in range(self.imgs.__len__()):
            data = cv2.imread(self.imgs[i])
            if firstiter:
                img = data[..., [0]]
                firstiter = False
            else:
                img = np.append(img, data[..., [0]], axis=2)

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)      # 仿射变换

        inp = self.normalize_image(inp)
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': ''}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.ann)


def test():
    network = make_network().cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.val.epoch)
    network.eval()

    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.val.batch_size)
    visualizer = Visualizer()
    contour = [torch.tensor([])]
    for batch in tqdm.tqdm(dataloader):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        batch['inp'] = batch['inp'][0, ...]
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)
        contour.append(output['py'][-1].squeeze(0).detach().cpu().numpy() * config_model.down_ratio)
    io.savemat(os.path.join(cfg.result_dir + '/contour.mat'),{'contour':contour[-1]})

if __name__ == '__main__':
    test()