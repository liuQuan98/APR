import torch.nn as nn
# import MinkowskiEngine as ME
# import MinkowskiEngine.MinkowskiFunctional as MEF
# from model.common import get_norm

class GenerativeMLP(nn.Module):
    CHANNELS = [None, 512, 128, None]

    def __init__(self, 
                 in_channel=125,
                 out_points=6,
                 bn_momentum=0.1):
        super().__init__()
        CHANNELS = self.CHANNELS
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, CHANNELS[1]),
            nn.ReLU(),
            nn.BatchNorm1d(CHANNELS[1], momentum=bn_momentum),
            nn.Linear(CHANNELS[1], CHANNELS[2]),
            nn.ReLU(),
            nn.BatchNorm1d(CHANNELS[2], momentum=bn_momentum),
            nn.Linear(CHANNELS[2], out_points*3),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.mlp(x)
        # print(y)
        return y


# class GenerativeMLP_99(GenerativeMLP):
#     CHANNELS = [None, 512, 512, None]


class GenerativeMLP_98(GenerativeMLP):
    CHANNELS = [None, 512, 256, None]


class GenerativeMLP_54(GenerativeMLP):
    CHANNELS = [None, 32, 16, None]


class GenerativeMLP_4(nn.Module):
    CHANNELS = [None, 16, None]

    def __init__(self, 
                 in_channel=125,
                 out_points=6,
                 bn_momentum=0.1):
        super().__init__()
        CHANNELS = self.CHANNELS
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, CHANNELS[1]),
            nn.ReLU(),
            nn.BatchNorm1d(CHANNELS[1], momentum=bn_momentum),
            nn.Linear(CHANNELS[1], out_points*3),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.mlp(x)
        # print(y)
        return y


class GenerativeMLP_11_10_9(nn.Module):
    CHANNELS = [None, 2048, 1024, 512, None]

    def __init__(self, 
                 in_channel=125,
                 out_points=6,
                 bn_momentum=0.1):
        super().__init__()
        CHANNELS = self.CHANNELS
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, CHANNELS[1]),
            nn.ReLU(),
            nn.BatchNorm1d(CHANNELS[1], momentum=bn_momentum),
            nn.Linear(CHANNELS[1], CHANNELS[2]),
            nn.ReLU(),
            nn.BatchNorm1d(CHANNELS[2], momentum=bn_momentum),
            nn.Linear(CHANNELS[2], CHANNELS[3]),
            nn.ReLU(),
            nn.BatchNorm1d(CHANNELS[3], momentum=bn_momentum),
            nn.Linear(CHANNELS[3], out_points*3),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.mlp(x)
        # print(y)
        return y