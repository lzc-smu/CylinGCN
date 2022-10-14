from torch.utils.data.dataloader import default_collate
import torch
from lib.utils.network import config_model


def collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    if 'test' in meta:
        return ret

    batch_size = len(batch)
    img_size = meta['img_size']
    ic = torch.zeros([batch_size, img_size, config_model.point_num, 3], dtype=torch.float)
    cic = torch.zeros([batch_size, img_size, config_model.point_num, 3], dtype=torch.float)
    gc = torch.zeros([batch_size, img_size, config_model.gt_point_num, 3], dtype=torch.float)
    cgc = torch.zeros([batch_size, img_size, config_model.gt_point_num, 3], dtype=torch.float)
    igc = torch.zeros([batch_size, img_size, config_model.gt_point_num * config_model.gt_point_num, 3], dtype=torch.float)
    for i in range(batch_size):
        ic[i] = torch.Tensor(sum([b['ic'] for b in batch], []))
        cic[i] = torch.Tensor(sum([b['cic'] for b in batch], []))
        gc[i] = torch.Tensor(sum([b['gc'] for b in batch], []))
        cgc[i] = torch.Tensor(sum([b['cgc'] for b in batch], []))
        igc = torch.Tensor(sum([b['igc'] for b in batch], []))
    evolution = {'ic': ic, 'cic': cic, 'gc': gc, 'cgc': cgc, 'igc': igc}
    ret.update(evolution)
    return ret


def make_collator():
    return collator

