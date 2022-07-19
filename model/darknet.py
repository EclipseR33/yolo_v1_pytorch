import torch.nn as nn


def conv(in_ch, out_ch, k_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=False),
        nn.LeakyReLU(0.1)
    )


def make_layer(param):
    layers = []
    if not isinstance(param[0], list):
        param = [param]
    for p in param:
        layers.append(conv(*p))
    return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self, param, use_pool=True):
        super(Block, self).__init__()

        self.conv = make_layer(param)
        self.pool = nn.MaxPool2d(2)
        self.use_pool = use_pool

    def forward(self, x):
        x = self.conv(x)
        if self.use_pool:
            x = self.pool(x)
        return x


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()

        self.conv1 = Block([[3, 64, 7, 2, 3]])
        self.conv2 = Block([[64, 192, 3, 1, 1]])
        self.conv3 = Block([[192, 128, 1, 1, 0],
                            [128, 256, 3, 1, 1],
                            [256, 256, 1, 1, 0],
                            [256, 512, 3, 1, 1]])
        self.conv4 = Block([[512, 256, 1, 1, 0],
                            [256, 512, 3, 1, 1],
                            [512, 256, 1, 1, 0],
                            [256, 512, 3, 1, 1],
                            [512, 256, 1, 1, 0],
                            [256, 512, 3, 1, 1],
                            [512, 256, 1, 1, 0],
                            [256, 512, 3, 1, 1],
                            [512, 512, 1, 1, 0],
                            [512, 1024, 3, 1, 1]])
        self.conv5 = Block([[1024, 512, 1, 1, 0],
                            [512, 1024, 3, 1, 1],
                            [1024, 512, 1, 1, 0],
                            [512, 1024, 3, 1, 1],
                            [1024, 1024, 3, 1, 1],
                            [1024, 1024, 3, 2, 1]], False)
        self.conv6 = Block([[1024, 1024, 3, 1, 1],
                            [1024, 1024, 3, 1, 1]], False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


if __name__ == "__main__":
    import torch
    x = torch.randn([1, 3, 448, 448])

    net = DarkNet()
    print(net)
    out = net(x)
    print(out.size())
