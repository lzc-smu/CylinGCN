import torch.nn as nn
from .graphSAGE import GCN, construct_graph
from lib.utils.network import gcn_utils, config_model
import torch


class ResGCN(nn.Module):
    def __init__(self):
        super(ResGCN, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)
        self.init_gcn = GCN(state_dim=128, feature_dim=64 + 3)
        self.resgcn = GCN(state_dim=128, feature_dim=64 + 3)
        self.iter = 2
        for i in range(self.iter):
            resgcn = GCN(state_dim=128, feature_dim=64 + 3)
            self.__setattr__('resgcn' + str(i), resgcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_contour(self, ret, batch):
        init = gcn_utils.init_contour(batch)
        ret.update({'ic': init['ic'], 'gc': init['gc']})
        return init


    def test_init_contour(self, ret, cnn_feature):
        init = gcn_utils.test_contour(cnn_feature)
        ret.update({'ic': init['ic']})
        return init


    def evolve_contour(self, modal, cnn_feature, init_contour, can_init):
        if len(init_contour) == 0:
            return torch.zeros_like(init_contour)
        d, h, w = cnn_feature.size(-3), cnn_feature.size(-2), cnn_feature.size(-1)
        init_feature = gcn_utils.get_3d_gcn_feature(cnn_feature, init_contour, d, h, w)
        can_init[..., 1:] = can_init[..., 1:] * config_model.ro
        init_input = torch.cat([init_feature, can_init.permute(0, 1, 3, 2)], dim=2)
        graph = construct_graph(init_input)
        init_contour[..., 1:] = init_contour[..., 1:] * config_model.ro + torch.cat([modal(init_input[i], graph).permute(0, 2, 1).unsqueeze(0) for i in range(init_input.size(0))], dim=0)
        return init_contour


    def forward(self, cnn_feature, batch=None):
        ret = {}

        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.init_contour(ret, batch)

            pred = self.evolve_contour(self.resgcn, cnn_feature, init['ic'], init['cic'])
            preds = []
            preds.append(pred.clone())
            for i in range(self.iter):
                pred[..., 1:] = pred[..., 1:] / config_model.ro
                c_pred = gcn_utils.img_poly_to_can_poly_3d(pred)
                resgcn = self.__getattr__('resgcn'+str(i))
                pred = self.evolve_contour(resgcn, cnn_feature, pred, c_pred)
                preds.append(pred.clone())
            ret['gc'][..., 1:] = ret['gc'][..., 1:] * config_model.ro
            ret.update({'preds': preds})

        if not self.training:
            with torch.no_grad():
                init = self.test_init_contour(ret, cnn_feature)
                py = self.evolve_contour(self.resgcn, cnn_feature, init['ic'], init['cic'])

                pys = [py / config_model.ro]
                for i in range(self.iter):
                    py = py / config_model.ro
                    c_py = gcn_utils.img_poly_to_can_poly_3d(py)
                    resgcn = self.__getattr__('resgcn'+str(i))
                    py = self.evolve_contour(resgcn, cnn_feature, py, c_py)
                    pys.append(py / config_model.ro)
                ret.update({'py': pys})

        return ret

