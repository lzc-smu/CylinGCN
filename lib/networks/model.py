import torch.nn as nn
from .MCFE import MultiCNN
from .RGCN import ResGCN


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.multicnn = MultiCNN(pretrained=False,
                             down_ratio=4,
                             last_level=5,
                             out_channel=0)
        self.resgcn = ResGCN()

    def forward(self, x, batch=None):
        cnn_feature = self.multicnn(x)
        output = self.resgcn(cnn_feature, batch)
        return output

