import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
#  Convolution block, for both 2D and 3D
#  conv (+ normalization + activation)
# -------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, dim=3, in_ch=32, out_ch=32, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1,
                 bias=False, padding_mode='replicate', mode='CBR'):
        super().__init__()

        blocks = []
        for m in mode:
            if m == 'C':
                Conv = getattr(nn, f"Conv{dim}d")   # Conv2d or Conv3d
                blocks.append(Conv(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))
            elif m == 'T':
                ConvTranspose = getattr(nn, f"ConvTranspose{dim}d")
                blocks.append(ConvTranspose(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=output_padding, bias=bias,
                                            dilation=dilation))
            elif m == 'B':
                BatchNorm = getattr(nn, 'BatchNorm{}d'.format(dim))
                blocks.append(BatchNorm(out_ch))
            elif m == 'I':
                InstanceNorm = getattr(nn, 'InstanceNorm{}d'.format(dim))
                blocks.append(InstanceNorm(out_ch))

            elif m == 'R':
                blocks.append(nn.ReLU())
            elif m == 'L':
                blocks.append(nn.LeakyReLU(negative_slope=0.1))
            elif m == 'P':
                blocks.append(nn.PReLU(num_parameters=1, init=0.25))
            elif m == 'E':
                blocks.append(nn.ELU(alpha=1.0))
            else:
                raise NotImplementedError('Undefined type: {}'.format(m))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block(x)
        return out


# -------------------------------------------------------
#  Residual block
#  [(conv + normalization + activation + conv + normalization) + identity] + activation
# -------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    A Bottleneck Residual Block utilises 1x1 convolutions to create a bottleneck, reducing the number of parameters to
    increase depth, have been used as part of deeper ResNets such as ResNet-50 and ResNet-101.
    """
    def __init__(self, dim=3, in_ch=32, out_ch=64, stride=1, mode='CBR', bottleneck=True):
        super().__init__()

        # Determine the last activation function from mode
        if mode[-1] == 'R':
            self.activation = nn.ReLU()
        elif mode[-1] == 'L':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        elif mode[-1] == 'P':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif mode[-1] == 'E':
            self.activation = nn.ELU(alpha=1.0)
        else:
            mode += 'R'
            self.activation = nn.ReLU()

        # Standard or bottleneck block, downsample when stride > 1
        self.bottleneck = bottleneck
        if self.bottleneck:
            self.conv1 = ConvBlock(dim, in_ch, out_ch//4, kernel_size=1, padding=0,  mode=mode)
            self.conv2 = ConvBlock(dim, out_ch//4, out_ch//4, stride=stride, mode=mode)
            self.conv3 = ConvBlock(dim, out_ch//4, out_ch, kernel_size=1, padding=0, mode=mode[:-1])
        else:
            self.conv1 = ConvBlock(dim, in_ch, out_ch, stride=stride, mode=mode)
            self.conv2 = ConvBlock(dim, out_ch, out_ch, mode=mode[:-1])

        # Shortcut connection, use conv1x1 to adjust size or channel number if needed
        if in_ch == out_ch and stride == 1:
            self.shortcut = None
        else:
            self.shortcut = ConvBlock(dim, in_ch, out_ch, kernel_size=1, padding=0, stride=stride, bias=False,
                                      mode=mode[:-1])

    def forward(self, x):
        identity = x

        if self.bottleneck:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        if self.shortcut:
            identity = self.shortcut(identity)

        out = x + identity
        out = self.activation(out)
        return out


# -------------------------------------------------------
# Residual Dense Block (RDB)
# -------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    """
        Channel numbers for input and output should be the same.
        layers is the number of Conv layers in one dense Block, excluding the final fusion layer
        growth rate is the number of output channels at each convolutional layer inside the block
        https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.14441
    """
    def __init__(self, dim=3, in_ch=32, growth_rate=16, layers=4, mode='CBR'):
        super().__init__()

        self.layer_list = nn.ModuleList()
        pre_ch = in_ch
        for i in range(layers):
            self.layer_list.append(ConvBlock(dim, pre_ch, growth_rate, mode=mode))
            pre_ch = in_ch + (i + 1) * growth_rate

        # No relu in fusion layer
        self.local_feature_fusion = ConvBlock(dim, pre_ch, in_ch, kernel_size=1, padding=0, mode=mode[:-1])

    def forward(self, x):
        outputs = []

        for layer in self.layer_list:
            stacked = torch.cat((x, *outputs), dim=1)   # stack on channel
            x_out = layer(stacked)
            outputs.append(x_out)

        stacked = torch.cat((x, *outputs), dim=1)
        x_out = self.local_feature_fusion(stacked)

        x_out = x + x_out   # skip add for local residual learning

        return x_out
