import torch
import torch.nn as nn
from torch.nn.functional import conv2d, relu

def build_successive_convolutions(
        n_layers, 
        # activation_function,
        in_channels, 
        out_channels, 
        kernel_size, 
        **conv_kwargs):
    layers = []
    for _ in range(n_layers):
        # layers.append(activation_function(nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs))
    return layers


class SuccessiveConvolutions(nn.Module):
    def __init__(self, n_layers) -> None:
        super().__init__()
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.middle_layers = build_successive_convolutions(n_layers=n_layers-1, 
                                                           in_channels=64, 
                                                           out_channels=64, 
                                                           kernel_size=3,
                                                           **{"padding": 1})
        self.last_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        xi = relu(self.first_layer(x))
        for i, layer in enumerate(self.middle_layers):
            if i == 0:
                xm = relu(layer(xi))
            else:
                xm = relu(layer(xm))
        xf = self.last_layer(xm)
        return x + xf
    