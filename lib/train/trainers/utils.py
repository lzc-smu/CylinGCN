import torch
from torch import nn



class tomatch(nn.Module):
    def __init__(self):
        super(tomatch, self).__init__()

    def tomatch(self, pred_contour, gt_contour):
        loss = 0
        weight = [0.2, 0.3, 0.5]
        for n in range(len(pred_contour)):
            preds = pred_contour[n].squeeze(dim=0)
            gts = gt_contour.squeeze(dim=0)
            for i in range(len(preds)):
                pred_poly = preds[i]
                gt_poly = gts[i]
                for pred_point in pred_poly:
                    loss += torch.min(torch.norm(gt_poly - pred_point[1:], p=2, dim=1)) / len(pred_poly) / len(preds) / len(pred_contour) * weight[n]
        return loss

    def forward(self, pred, gt):
        return self.tomatch(pred, gt)


class tokey(nn.Module):
    def __init__(self):
        super(tokey, self).__init__()

    def tokey(self, pred_contour, gt_contour):
        loss = torch.nn.functional.smooth_l1_loss
        return loss(pred_contour, gt_contour)


    def forward(self, pred, gt):
        return self.tokey(pred, gt)

