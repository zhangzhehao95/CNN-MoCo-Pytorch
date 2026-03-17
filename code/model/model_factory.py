import torch
from torch import nn
from .blocks import ConvBlock, ResidualBlock, ResidualDenseBlock


class UNet(nn.Module):
    # https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
    def __init__(self, cf):
        super().__init__()

        dim = len([s for s in cf['patch_size'] if s > 1])
        assert dim in (2, 3), 'Only support 2 or 3 dimension.'

        # convolution blocks
        conv_mode = cf['conv_mode']

        pre_ch = 1
        # Encoder
        self.encoder = nn.ModuleList()
        for layer in range(cf['num_level'] - 1):
            cur_ch = (2 ** layer) * cf['ch_base']
            layer_list = []
            for i in range(cf['conv_per_level']):
                layer_list.append(ConvBlock(dim, pre_ch, cur_ch, mode=conv_mode))
                pre_ch = cur_ch

            if cf['down_sample'] == 'StrideConv':
                self.down = ConvBlock(dim, pre_ch, pre_ch, stride=2, mode='C')
            else:
                MaxPool = getattr(nn, 'MaxPool{}d'.format(dim))
                self.down = MaxPool(2)

            self.encoder.append(nn.ModuleList([nn.Sequential(*layer_list), self.down]))

        # Bottleneck
        cur_ch = 2 ** (cf['num_level'] - 1) * cf['ch_base']
        layer_list = []
        for i in range(cf['conv_per_level']):
            layer_list.append(ConvBlock(dim, pre_ch, cur_ch, mode=conv_mode))
            pre_ch = cur_ch
        self.bottleneck = nn.Sequential(*layer_list)

        # Decoder
        self.decoder = nn.ModuleList()
        for layer in range(cf['num_level'] - 2, -1, -1):
            if cf['up_sample'] == 'TransConv':
                self.up = ConvBlock(dim, pre_ch, pre_ch//2, kernel_size=2, stride=2, padding=0, mode='T')
            else:
                self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear' if dim == 2 else 'trilinear'),
                                        ConvBlock(dim, pre_ch, pre_ch//2, kernel_size=1, padding=0, mode='C'))
                # self.up = nn.Upsample(scale_factor=2, mode='bilinear' if dim == 2 else 'trilinear')
                # pre_ch = pre_ch + pre_ch//2

            cur_ch = (2 ** layer) * cf['ch_base']
            layer_list = []
            for i in range(cf['conv_per_level']):
                layer_list.append(ConvBlock(dim, pre_ch, cur_ch, mode=conv_mode))
                pre_ch = cur_ch
            self.decoder.append(nn.ModuleList([self.up, nn.Sequential(*layer_list)]))

        # Output
        self.out = ConvBlock(dim, pre_ch, out_ch=1, kernel_size=1, padding=0, mode='C')
        self.learn_residual = cf['learn_residual']

    def forward(self, x):
        model_input = x
        skips = []
        for process, down in self.encoder:
            x = process(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, process in self.decoder:
            x = up(x)
            x = process(torch.cat([x, skips.pop()], 1))    # Data: (B,C,D,H,W), concatenate along channel

        x = self.out(x)

        if self.learn_residual:
            x = x + model_input

        return x


class ResUNet(nn.Module):
    """
        UNet with residual block
        https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66
    """
    def __init__(self, cf):
        super().__init__()

        dim = len([s for s in cf['patch_size'] if s > 1])
        assert dim in (2, 3), 'Only support 2 or 3 dimension.'

        # convolution blocks
        conv_mode = cf['conv_mode']

        pre_ch = 1
        # Encoder
        self.encoder = nn.ModuleList()
        for layer in range(cf['num_level'] - 1):
            cur_ch = (2 ** layer) * cf['ch_base']
            layer_list = []
            for i in range(cf['conv_per_level']):
                # the 'conv_per_level' presents the number of residual blocks per level
                layer_list.append(ResidualBlock(dim, pre_ch, cur_ch, mode=conv_mode, bottleneck=cf['bottleneck']))
                pre_ch = cur_ch

            if cf['down_sample'] == 'StrideConv':
                self.down = ConvBlock(dim, pre_ch, pre_ch, stride=2, mode='C')
            else:
                MaxPool = getattr(nn, 'MaxPool{}d'.format(dim))
                self.down = MaxPool(2)

            self.encoder.append(nn.ModuleList([nn.Sequential(*layer_list), self.down]))

        # Bottleneck
        cur_ch = 2 ** (cf['num_level'] - 1) * cf['ch_base']
        layer_list = []
        for i in range(cf['conv_per_level']):
            layer_list.append(ResidualBlock(dim, pre_ch, cur_ch, mode=conv_mode))
            pre_ch = cur_ch
        self.bottleneck = nn.Sequential(*layer_list)

        # Decoder
        self.decoder = nn.ModuleList()
        for layer in range(cf['num_level'] - 2, -1, -1):
            if cf['up_sample'] == 'TransConv':
                self.up = ConvBlock(dim, pre_ch, pre_ch//2, kernel_size=2, stride=2, padding=0, mode='T')
            else:
                self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear' if dim == 2 else 'trilinear'),
                                        ConvBlock(dim, pre_ch, pre_ch//2, kernel_size=1, padding=0, mode='C'))

            cur_ch = (2 ** layer) * cf['ch_base']
            layer_list = []
            for i in range(cf['conv_per_level']):
                layer_list.append(ResidualBlock(dim, pre_ch, cur_ch, mode=conv_mode))
                pre_ch = cur_ch
            self.decoder.append(nn.ModuleList([self.up, nn.Sequential(*layer_list)]))

        # Output
        self.out = ConvBlock(dim, pre_ch, out_ch=1, kernel_size=1, padding=0, mode='C')
        self.learn_residual = cf['learn_residual']

    def forward(self, x):
        model_input = x
        skips = []
        for process, down in self.encoder:
            x = process(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, process in self.decoder:
            x = up(x)
            x = process(torch.cat([x, skips.pop()], 1))    # Data: (B,C,D,H,W), concatenate along channel

        x = self.out(x)

        if self.learn_residual:
            x = x + model_input

        return x


class DnCNN(nn.Module):
    """
        Denoising convolutional neural networks
        https://arxiv.org/pdf/1608.03981
    """

    def __init__(self, cf):
        super().__init__()

        dim = len([s for s in cf['patch_size'] if s > 1])
        assert dim in (2, 3), 'Only support 2 or 3 dimension.'

        # convolution blocks
        conv_mode = cf['conv_mode']

        self.DnCNN_depth = cf['DnCNN_depth']
        self.DnCNN_ch = cf['DnCNN_ch']

        # Input layer
        self.input_layer = ConvBlock(dim, in_ch=1, out_ch=self.DnCNN_ch, mode='CR')

        # Hidden layers
        self.hidden = nn.ModuleList()
        for i in range(self.DnCNN_depth - 2):
            self.hidden.append(ConvBlock(dim, in_ch=self.DnCNN_ch, out_ch=self.DnCNN_ch, mode=conv_mode))

        # Output layer
        self.output_layer = ConvBlock(dim, in_ch=self.DnCNN_ch, out_ch=1, mode='C')

        self.learn_residual = cf['learn_residual']

    def forward(self, x):
        model_input = x
        x = self.input_layer(x)

        for block in self.hidden:
            x = block(x)

        x = self.output_layer(x)

        if self.learn_residual:
            x = x + model_input

        return x


class RDN(nn.Module):
    """
        Residual dense network (RDN)
        https://arxiv.org/pdf/1812.10477
    """
    def __init__(self, cf):
        super().__init__()

        dim = len([s for s in cf['patch_size'] if s > 1])
        assert dim in (2, 3), 'Only support 2 or 3 dimension.'

        # convolution blocks
        conv_mode = cf['conv_mode']

        self.fea_extract_ch = cf['fea_extract_ch']
        self.RDB_num = cf['RDB_num']
        self.RDB_growth_rate = cf['RDB_growth_rate']
        self.RDB_layer = cf['RDB_layer']

        # feature extraction
        # the first feature extraction used for global residual learning, and the second used as input to RDBs
        self.feature_extraction1 = ConvBlock(dim, in_ch=1, out_ch=self.fea_extract_ch, mode=conv_mode)
        self.feature_extraction2 = ConvBlock(dim, in_ch=self.fea_extract_ch, out_ch=self.fea_extract_ch, mode=conv_mode)

        # RDBs
        self.RDB_list = nn.ModuleList()
        for i_block in range(self.RDB_num):
            self.RDB_list.append(ResidualDenseBlock(dim, in_ch=self.fea_extract_ch, growth_rate=self.RDB_growth_rate,
                                                    layers=self.RDB_layer, mode=conv_mode))

        # global feature fusion
        self.global_feature_fuse1 = ConvBlock(dim, in_ch=self.fea_extract_ch * self.RDB_num, out_ch=self.fea_extract_ch,
                                              kernel_size=1, padding=0, mode=conv_mode)
        self.global_feature_fuse2 = ConvBlock(dim, in_ch=self.fea_extract_ch, out_ch=self.fea_extract_ch, mode='C')

        # output, with global residual learning
        self.out = ConvBlock(dim, in_ch=self.fea_extract_ch, out_ch=1, mode='C')

        self.learn_residual = cf['learn_residual']

    def forward(self, x):
        model_input = x
        x_shallow_feature = self.feature_extraction1(x)
        x = self.feature_extraction2(x_shallow_feature)

        block_outputs = []
        for block in self.RDB_list:
            x = block(x)
            block_outputs.append(x)

        x = self.global_feature_fuse1(torch.cat(block_outputs, dim=1))
        x = self.global_feature_fuse2(x)

        x = self.out(x + x_shallow_feature)

        if self.learn_residual:
            x = x + model_input

        return x
