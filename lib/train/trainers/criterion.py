import torch.nn as nn
from .utils import tokey, tomatch
from lib.utils.network import config_model


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.match_loss = tomatch()
        self.key_loss = tokey()

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        keyloss = 0
        matchloss = 0
        preds = output['preds']
        output['preds'] = [output['preds'][-1]]
        for i in range(len(output['preds'])):
            keyloss += self.key_loss(output['preds'][i][..., 1:], output['gc'][..., 1:]) / len(output['preds'])
            matchloss += self.match_loss(preds, batch['igc'][..., 1:] * config_model.ro) / len(output['preds'])
        scalar_stats.update({'keyloss': keyloss, 'matchloss': matchloss})
        loss += (keyloss + matchloss) * 0.5
        scalar_stats.update({'loss': loss})
        image_stats = {}


        return output, loss, scalar_stats, image_stats