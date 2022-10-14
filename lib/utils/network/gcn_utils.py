import torch
import math
from lib.config import cfg
from lib.utils.network import config_model
from lib.csrc.extreme_utils import _ext as extreme_utils


def img_poly_to_can_poly_3d(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 1], dim=-1)[0]
    y_min = torch.min(img_poly[..., 2], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 1] = can_poly[..., 1] - x_min[..., None]
    can_poly[..., 2] = can_poly[..., 2] - y_min[..., None]
    return can_poly


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly



def init_contour(batch):
    init = {}
    init.update({'ic': batch['ic']})
    init.update({'cic': batch['cic']})
    init.update({'gc': batch['gc']})
    init.update({'cgc': batch['cgc']})
    return init


def get_3d_gcn_feature(cnn_feature, contour, d, h, w):
    contour = contour.clone()
    contour[..., -3] = contour[..., -3] / (d / 2.) - 1
    contour[..., -2] = contour[..., -2] / (h / 2.) - 1
    contour[..., -1] = contour[..., -1] / (w / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([contour.size(0), contour.size(1), cnn_feature.size(1), contour.size(2)]).to(contour.device)
    for i in range(batch_size):
        contour_ = contour[i].unsqueeze(0)
        for j in range(contour_.size(1)):
            feature = torch.nn.functional.grid_sample(cnn_feature[i:i + 1, :, j, :, :], contour_[:, j:j + 1, :, 1:])[0].permute(1, 0, 2)
            gcn_feature[i, j, ...] = feature
    return gcn_feature


def uniform_cirsample_3d(feat, num=0):
    d, h, w = feat.size(-3), feat.size(-2), feat.size(-1)
    contour = torch.zeros([d, num, 3]).cuda()
    a, b = h/2, w/2
    alpha = torch.linspace(0, 2*math.pi*(num-1)/num, num).cuda()
    r = cfg.test.radius_len * min(a, b)

    x = a + r * torch.sin(alpha)
    y = b + r * torch.cos(alpha)
    point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    for i in range(d):
        contour[i, ..., 1:] = point
        contour[i, ..., 0] = i
    return contour


def test_contour(feat):
    ic = uniform_cirsample_3d(feat, num=config_model.point_num).unsqueeze(dim=0)
    ic = ic.cuda()
    cic = img_poly_to_can_poly_3d(ic)
    init = {'ic': ic, 'cic': cic}
    return init
