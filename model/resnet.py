import torch
from torchvision.models.resnet import ResNet, Bottleneck


class ResNet_(ResNet):
    def __init__(self, block, layers):
        super(ResNet_, self).__init__(block=block, layers=layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, pretrained):
    model = ResNet_(block, layers)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
    return model


def resnet_1024ch(pretrained=None) -> ResNet:
    resnet = _resnet(Bottleneck, [3, 4, 6, 3], pretrained)
    return resnet


if __name__ == '__main__':
    x = torch.randn([1, 3, 448, 448])
    net = resnet_1024ch('resnet50-19c8e357.pth')
    print(net)

    y = net(x)
    print(y.size())
