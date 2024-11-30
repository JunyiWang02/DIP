import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, init_weights=True):
        super().__init__()

        conv_channels = [64, 128, 256, 512]

        def cbr_block(in_channel, out_channel, normalize=True, kernel_size=3, stride=1, padding=1, activation=None):
            layers = [
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False if normalize else True),
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers
        self.net = nn.Sequential(
            *cbr_block(in_channels, conv_channels[0], normalize=False),  
            *cbr_block(conv_channels[0], conv_channels[1]),             
            *cbr_block(conv_channels[1], conv_channels[2]),             
            *cbr_block(conv_channels[2], conv_channels[3]),             
            *cbr_block(conv_channels[3], 3, normalize=False, activation=nn.Sigmoid())  
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        return self.net(torch.cat((x, y), 1))
