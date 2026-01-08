import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from .arch_util import srntt_init_weights
import functools
import numpy as np




class ImageDiscriminator(nn.Module):

    def __init__(self, in_nc=3, ndf=32):
        super(ImageDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True))
            return block

        self.conv_block1 = conv_block(in_nc, ndf)
        self.conv_block2 = conv_block(ndf, ndf * 2)
        self.conv_block3 = conv_block(ndf * 2, ndf * 4)
        self.conv_block4 = conv_block(ndf * 4, ndf * 8)
        self.conv_block5 = conv_block(ndf * 8, ndf * 16)

        self.out_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ndf * 16, 1024, kernel_size=1),
            nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1), nn.Sigmoid())

        srntt_init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x):
        fea = self.conv_block1(x)
        fea = self.conv_block2(fea)
        fea = self.conv_block3(fea)
        fea = self.conv_block4(fea)
        fea = self.conv_block5(fea)

        out = self.out_block(fea)

        return out
    
class Discriminator_UNet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, in_nc=3, ndf=64):
        super(Discriminator_UNet, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(in_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(ndf, 1, 3, 1, 1)
        print('using the UNet discriminator')

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
    


class Discriminator_PatchGAN(nn.Module):
    def __init__(self, in_nc=3, ndf=64, n_layers=3, norm_type='spectral'):
        '''PatchGAN discriminator, receptive field = 70x70 if n_layers = 3
        Args:
            input_nc: number of input channels 
            ndf: base channel number
            n_layers: number of conv layer with stride 2
            norm_type:  'batch', 'instance', 'spectral', 'batchspectral', instancespectral'
        Returns:
            tensor: score
        '''
        super(Discriminator_PatchGAN, self).__init__()
        self.n_layers = n_layers
        norm_layer = self.get_norm_layer(norm_type=norm_type)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[self.use_spectral_norm(nn.Conv2d(in_nc, ndf, kernel_size=kw, stride=2, padding=padw), norm_type), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[self.use_spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_type),
                          norm_layer(nf), 
                          nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[self.use_spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_type),
                      norm_layer(nf),
                      nn.LeakyReLU(0.2, True)]]

        sequence += [[self.use_spectral_norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw), norm_type)]]

        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.add_module('child' + str(n), nn.Sequential(*sequence[n]))

        self.model.apply(self.weights_init)

    def use_spectral_norm(self, module, norm_type='spectral'):
        if 'spectral' in norm_type:
            return spectral_norm(module)
        return module

    def get_norm_layer(self, norm_type='instance'):
        if 'batch' in norm_type:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif 'instance' in norm_type:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            norm_layer = functools.partial(nn.Identity)
        return norm_layer

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)