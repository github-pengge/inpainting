import torch
import torch.nn as nn 
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import os, random, math
try:
    reduce
except:
    from functools import reduce


def init_weights(net, init_type='normal'):
    if init_type in ['xavier', 'kaiming', 'orthogonal']:
        gain = 0.2
    else:
        gain = 0.02
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    net.apply(init_func)

def finetune(net):
    def set_finetune(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad = False
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = False

    net.apply(set_finetune)


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.register_buffer('conv_mask_weight', torch.ones(1, 1, *self.conv.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1).zero_())
        else:
            self.register_parameter('bias', None)

    def forward(self, x, mask):
        x = x * mask
        # mask part
        mask = F.conv2d(mask, Variable(self._buffers['conv_mask_weight'], requires_grad=False), bias=None, 
                        stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, \
                        groups=self.conv.groups)
        new_mask = torch.ones_like(mask)
        new_mask[mask == 0] = 0
        # x part
        x = self.conv(x)
        mask[mask == 0] = 1  # so that 0/0 would not occur.
        x = x / mask
        if self.bias is not None:
            x = x + self.bias
        return x, new_mask

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.conv.padding != (0,) * len(self.conv.padding):
            s += ', padding={padding}'
        if self.conv.dilation != (1,) * len(self.conv.dilation):
            s += ', dilation={dilation}'
        if self.conv.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.conv.__dict__)


def create_Generator(config):
    requires = ['g_input', 'decoder_partial_conv', 'learning_residual', 'init_type', 'block', 'progressive_growing']
    assert all([(k in config) for k in requires])
    if config['g_input'] == 'masked_X':
        encoder_partial_conv = False 
        decoder_partial_conv = False
    elif config['g_input'] == 'masked_X+mask':
        encoder_partial_conv = True 
        decoder_partial_conv = config['decoder_partial_conv']
    else:
        raise ValueError('Invalid combination of g_input: %s' % config['g_input'])

    if config['block'] == 'basic':
        block = BasicBlock
    elif config['block'] == 'residual':
        block = ResBlock
    else:
        raise ValueError('Invalid value for block: %s' % config['block'])

    if config['progressive_growing']:
        _Unet = ProgressiveGrowingUnet
    else:
        _Unet = Unet

    G = _Unet(block, config['learning_residual'], encoder_partial_conv, decoder_partial_conv, init_type=config['init_type'])
    return G


class BasicBlock(nn.Module):
    def __init__(self, upsample_first, use_pconv, no_norm, in_channels, out_channels, kernel_size, stride, padding, \
                dilation=1, groups=1, bias=True, norm=nn.BatchNorm2d, activation=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.upsample_first = upsample_first
        self.use_pconv = use_pconv
        self.no_norm = no_norm
        if self.upsample_first:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        conv = PartialConv2d if self.use_pconv else nn.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        norm_act = []
        if self.no_norm:
            norm_act += [activation]
        else:
            norm_act += [norm(out_channels), activation]
        self.norm_act = nn.Sequential(*norm_act)

    def forward(self, x, mask=None, skip=None):
        if self.upsample_first:
            x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        if self.use_pconv:
            x, mask = self.conv(x, mask)
            return self.norm_act(x), mask
        else:
            x = self.conv(x)
            return self.norm_act(x)


class ResBlock(nn.Module):
    def __init__(self, upsample_first, use_pconv, no_norm, in_channels, out_channels, kernel_size, stride, padding, \
                dilation=1, groups=1, bias=True, norm=nn.BatchNorm2d, activation=nn.ReLU()):
        super(ResBlock, self).__init__()
        self.upsample_first = upsample_first
        self.use_pconv = use_pconv
        self.no_norm = no_norm  # this is useless
        self.activation = activation
        if self.upsample_first:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        conv = PartialConv2d if self.use_pconv else nn.Conv2d
        self.conv1 = conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm1 = norm(out_channels)
        self.relu = nn.ReLU()
        p = kernel_size // 2
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, p)
        self.norm2 = norm(out_channels)
        self.downsample = self.downsample_norm = None 
        if stride != 1 or out_channels != in_channels:
            self.downsample = conv(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
            self.downsample_norm = norm(out_channels)

    def forward(self, x, mask=None, skip=None):
        if self.upsample_first:
            x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], 1)

        residual = x
        if self.use_pconv:
            if self.downsample is not None:
                residual, _ = self.downsample(x, mask)
                residual = self.downsample_norm(residual)
            x, mask = self.conv1(x, mask)
            x = self.norm1(x)
            x = self.relu(x)
            x, mask = self.conv2(x, mask)
            x = self.norm2(x)
            x = x + residual
            return self.activation(x), mask
        else:
            if self.downsample is not None:
                residual = self.downsample(x)
                residual = self.downsample_norm(residual)
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = x + residual
            return self.activation(x)


class Unet(nn.Module):
    def __init__(self, block, learning_residual, encoder_partial_conv, decoder_partial_conv, init_type):
        super(Unet, self).__init__()
        self.learning_residual = learning_residual
        self.encoder_partial_conv = encoder_partial_conv
        self.decoder_partial_conv = decoder_partial_conv
        self.init_type = init_type
        encoder_activation = nn.ReLU()
        decoder_activation = nn.LeakyReLU(0.2)
        output_activation = nn.Tanh() if self.learning_residual else nn.Sigmoid()
        stride = 2
        kernel_size = [7, 5, 5, 3, 3, 3, 3, 3]
        padding = [3, 2, 2, 1, 1, 1, 1, 1]
        in_channels = [3, 64, 128, 256, 512, 512, 512, 512]
        out_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        encoder_net = []
        for i, (ks, p, ic, oc) in enumerate(zip(kernel_size, padding, in_channels, out_channels)):
            if i == 0: # no batch norm
                encoder_net.append(block(False, self.encoder_partial_conv, True, ic, oc, ks, stride, p, activation=encoder_activation))
            else:
                encoder_net.append(block(False, self.encoder_partial_conv, False, ic, oc, ks, stride, p, activation=encoder_activation))
        self.encoder = nn.ModuleList(encoder_net)

        decoder_net = []
        decoder_in_channels = [1024, 1024, 1024, 1024, 768, 384, 192, 67]
        decoder_out_channels = [512, 512, 512, 512, 256, 128, 64, 3]
        stride = 1
        for i, (ks, p, ic, oc) in enumerate(zip(reversed(kernel_size), reversed(padding), decoder_in_channels, decoder_out_channels)):
            if i == len(kernel_size)-1: # no batch norm
                decoder_net.append(block(True, self.decoder_partial_conv, True, ic, oc, ks, stride, p, activation=output_activation))
            else:
                decoder_net.append(block(True, self.decoder_partial_conv, False, ic, oc, ks, stride, p, activation=decoder_activation))
        self.decoder = nn.ModuleList(decoder_net)
        init_weights(self, init_type)

    def forward(self, x, mask=None, **kwargs):
        input = x
        x_skips = [x]
        mask_skips = [mask]
        # encoder
        for i in range(len(self.encoder)):
            if self.encoder_partial_conv:
                x, mask = self.encoder[i](x, mask)
            else:
                x = self.encoder[i](x)
            if i != len(self.encoder)-1:
                x_skips.append(x)
                mask_skips.append(mask)
        # decoder
        for i in range(len(self.decoder)):
            if self.decoder_partial_conv:
                x, _ = self.decoder[i](x, mask_skips[-i-1], skip=x_skips[-i-1])
            else:
                x = self.decoder[i](x, skip=x_skips[-i-1])

        return torch.clamp(x+input, 0, 1) if self.learning_residual else x

    def set_mode(self, mode='train'):
        assert mode in ['train', 'finetune', 'test']
        if mode.lower() == 'train':
            self.encoder.train()
            self.decoder.train()
        elif mode.lower() == 'finetune':
            finetune(self.encoder)
            self.decoder.train()
        elif mode.lower() == 'test':
            self.encoder.eval()
            self.decoder.eval()


class ProgressiveGrowingUnet(nn.Module):
    def __init__(self, block, learning_residual, encoder_partial_conv, decoder_partial_conv, init_type):
        super(ProgressiveGrowingUnet, self).__init__()
        self.learning_residual = learning_residual
        self.encoder_partial_conv = encoder_partial_conv
        self.decoder_partial_conv = decoder_partial_conv
        self.init_type = init_type
        encoder_activation = nn.ReLU()
        decoder_activation = nn.LeakyReLU(0.2)
        output_activation = nn.Tanh() if self.learning_residual else nn.Sigmoid()
        stride = 2
        self.n_layer = 8
        kernel_size = [7, 5, 5, 3, 3, 3, 3, 3]
        padding = [3, 2, 2, 1, 1, 1, 1, 1]
        # in_channels = [3, 64, 128, 256, 512, 512, 512, 512]
        # out_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        # decoder_in_channels = [1024, 1024, 1024, 1024, 768, 384, 192, 67]
        # decoder_out_channels = [512, 512, 512, 512, 256, 128, 64, 3]
        ################# begin: 2018.06.12 use smaller network #################
        in_channels = [3, 16, 32, 64, 128, 256, 512, 512]
        out_channels = [16, 32, 64, 128, 256, 512, 512, 512]
        decoder_in_channels = [1024, 1024, 768, 384, 192, 96, 48, 19]
        decoder_out_channels = [512, 512, 256, 128, 64, 32, 16, 3]
        ################# end: 2018.06.12 use smaller network #################
        encoder_net = []
        input_layers = []
        for i, (ks, p, ic, oc) in enumerate(zip(kernel_size, padding, in_channels, out_channels)):
            input_layers.append(block(False, self.encoder_partial_conv, True, 3, ic, ks, 1, ks//2, activation=encoder_activation))
            encoder_net.append(block(False, self.encoder_partial_conv, False, ic, oc, ks, stride, p, activation=encoder_activation))
        self.encoder = nn.ModuleList(encoder_net)
        self.input_layers = nn.ModuleList(input_layers)

        decoder_net = []
        output_layers = []
        stride = 1
        for i, (ks, p, ic, oc) in enumerate(zip(reversed(kernel_size), reversed(padding), decoder_in_channels, decoder_out_channels)):
            decoder_net.append(block(True, self.decoder_partial_conv, False, ic, oc, ks, stride, p, activation=decoder_activation))
            output_layers.append(block(False, False, True, oc, 3, ks, 1, ks//2, activation=output_activation))
        self.decoder = nn.ModuleList(decoder_net)
        self.output_layers = nn.ModuleList(output_layers)
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        init_weights(self, init_type)

    def forward(self, x, mask=None, phase=0):
        k = phase  # [0, self.n_layer-1]
        lk, uk = int(math.floor(k)), int(math.ceil(k))
        input = x
        _mask = mask
        ##### encoder #####
        # input layer
        if self.encoder_partial_conv:
            x, mask = self.input_layers[lk](x, mask)
        else:
            x = self.input_layers[lk](x)
        if lk == 0:
            x_skips = [input]
            mask_skips = [_mask]
        else:
            x_skips = [x]
            mask_skips = [mask]
        for i in range(uk, len(self.encoder)):
            # progressive growing
            if i == uk and uk == lk+1:
                if self.encoder_partial_conv:
                    x, mask = self.encoder[lk](x, mask)
                else:
                    x = self.encoder[lk](x)
                if lk != len(self.encoder)-1:
                    x_skips.append(x)
                    mask_skips.append(mask)
                if self.encoder_partial_conv:
                    x_uk, _ = self.input_layers[i](self.downsample(input), self.downsample(_mask))
                else:
                    x_uk = self.input_layers[i](self.downsample(input))
                x = (k-lk) * x + (uk-k) * x_uk

            if self.encoder_partial_conv:
                x, mask = self.encoder[i](x, mask)
            else:
                x = self.encoder[i](x)
            if i != len(self.encoder)-1:
                x_skips.append(x)
                mask_skips.append(mask)
        ##### decoder #####
        for i in range(len(self.decoder)-lk):
            if self.decoder_partial_conv:
                x, _ = self.decoder[i](x, mask_skips[-i-1], skip=x_skips[-i-1])
            else:
                x = self.decoder[i](x, skip=x_skips[-i-1])
            if i == len(self.decoder)-lk-2 and uk == lk+1:
                x_out_lk = self.output_layers[i](x)
                x_out_lk = self.upsample(x_out_lk)
        # output layer
        x = self.output_layers[i](x)
        if uk == lk+1:
            x = (uk-k) * x + (k-lk) * x_out_lk

        return torch.clamp(x+input, 0, 1) if self.learning_residual else x

    def set_mode(self, mode='train'):
        assert mode in ['train', 'finetune', 'test']
        if mode.lower() == 'train':
            self.encoder.train()
            self.decoder.train()
        elif mode.lower() == 'finetune':
            finetune(self.encoder)
            self.decoder.train()
        elif mode.lower() == 'test':
            self.encoder.eval()
            self.decoder.eval()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, pix2pix_style=False, init_type='normal'):
        super(Discriminator, self).__init__()
        self.pix2pix_style = pix2pix_style
        lrelu = nn.LeakyReLU(0.2)
        first_nf = 6 if self.pix2pix_style else 3
        self.net = nn.Sequential(nn.Conv2d(first_nf, 64, 4, 2, 1), nn.BatchNorm2d(64), lrelu, \
                                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), lrelu, \
                                nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), lrelu, \
                                nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                Flatten(), nn.Linear(512*4*4, 1, bias=True), nn.Sigmoid())  # input size: 512x512
        init_weights(self.net, init_type)

    def forward(self, x, **kwargs):
        return self.net(x)


class ProgressiveGrowingDiscriminator(nn.Module):
    def __init__(self, pix2pix_style=False, init_type='normal'):
        super(ProgressiveGrowingDiscriminator, self).__init__()
        self.pix2pix_style = pix2pix_style
        self.init_type = init_type
        self.max_nf = 512
        self.img_size = 512
        self.n_layer = 8
        self.net = nn.ModuleList()
        self.input_layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.input_layers.append(self.make_block(i, True, False))
            self.net.append(self.make_block(i, False, True))
        self.downsample = nn.MaxPool2d(2, 2)
        init_weights(self.input_layers, init_type)
        init_weights(self.net, init_type)

    def make_block(self, i, is_input_layer=False, bn=True):
        is_last_block = (i == self.n_layer-1)
        ks = 3 if is_input_layer else 4
        stride = 1 if is_input_layer else 2
        padding = 1
        ic = self.get_nf(-1) if is_input_layer else self.get_nf(i)
        oc = self.get_nf(i) if is_input_layer else self.get_nf(i+1)
        block = [nn.Conv2d(ic, oc, ks, stride, padding)]
        if bn:
            block += [nn.BatchNorm2d(oc)]
        block += [nn.LeakyReLU(0.2)]
        if is_last_block:
            size = self.img_size // (2**(i+1))
            block += [Flatten(), nn.Linear(oc*size*size, 1, bias=True), nn.Sigmoid()]
        return nn.Sequential(*block)

    def get_nf(self, i):
        assert i >= -1
        if i == -1:
            return 6 if self.pix2pix_style else 3
        else:
            # return min(self.max_nf, 32*(2**i))
            return min(self.max_nf, 16*(2**i))  # 2018.06.12 use smaller network

    def forward(self, x, phase=0):
        k = (self.n_layer-1) - phase  # [0, self.n_layer-1]
        lk, uk = int(math.floor(k)), int(math.ceil(k))
        input = x
        x = self.input_layers[lk](x)
        for i in range(uk, self.n_layer):
            if i==uk and uk==lk+1:
                x = self.net[lk](x)
                x_uk = self.input_layers[uk](self.downsample(input))
                x = x * (uk-k) + x_uk * (k-lk)
            x = self.net[i](x)
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, pix2pix_style=False, init_type='normal'):
        super(PatchDiscriminator, self).__init__()
        self.pix2pix_style = pix2pix_style
        lrelu = nn.LeakyReLU(0.2)
        first_nf = 6 if self.pix2pix_style else 3
        self.net = nn.Sequential(nn.Conv2d(first_nf, 64, 4, 2, 1), nn.BatchNorm2d(64), lrelu, \
                                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), lrelu, \
                                nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), lrelu, \
                                nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), lrelu, \
                                nn.Conv2d(512, 1, 4, 1, 1), nn.Sigmoid())  # input size: 512x512
        init_weights(self.net, init_type)

    def forward(self, x, **kwargs):
        return self.net(x)


def _pair(x):
    if isinstance(x, tuple) or isinstance(x, list):
        assert len(x) == 2
        return x 
    else:
        assert not hasattr(x, '__iter__')
        return (x, x)


class PatchProposalLayer2d(nn.Module):
    def __init__(self, patch_size):
        super(PatchProposalLayer2d, self).__init__()
        self.patch_size = _pair(patch_size)
        self.register_buffer('_weight', torch.ones(1, 1, *self.patch_size))
        self._sum = self.patch_size[0] * self.patch_size[1]
        self._map = lambda x: list(map(lambda a: a.cpu().data[0], x))

    def forward(self, mask):
        res = F.conv2d(mask, Variable(self._buffers['_weight'], requires_grad=False), bias=None, stride=self.patch_size, padding=0)
        idx = [torch.nonzero(res[i, 0] < self._sum) for i in range(mask.size(0))]
        proposed = [torch.randperm(idx_i.size(0))[0] for idx_i in idx]
        left_h = [self.patch_size[0] * idx[i][proposed[i]][0] for i in range(mask.size(0))]
        left_w = [self.patch_size[1] * idx[i][proposed[i]][1] for i in range(mask.size(0))]
        return self._map(left_h), self._map(left_w)


class LocalDiscriminator(nn.Module):
    def __init__(self, patch_size=128, pix2pix_style=False, init_type='normal'):
        super(LocalDiscriminator, self).__init__()
        self.patch_size = _pair(patch_size)
        self.proposal_net = PatchProposalLayer2d(self.patch_size)
        self.pix2pix_style = pix2pix_style
        lrelu = nn.LeakyReLU(0.2)
        first_nf = 6 if self.pix2pix_style else 3
        n_layer = int(math.log2(min(self.patch_size))) - 1
        assert n_layer >= 3  # to few layer would restrict its ability
        net = []
        for i in range(n_layer):
            if i==0:
                ic = first_nf
            else:
                ic = oc
            # oc = min(2 ** (i+6), 512)
            oc = min(2 ** (i+4), 512)  # 2018.06.12 use smaller network
            net.extend([nn.Conv2d(ic, oc, 4, 2, 1), nn.BatchNorm2d(oc), lrelu])
        net.extend([Flatten(), nn.Linear(oc*2*2, 1, bias=True), nn.Sigmoid()])
        self.net = nn.Sequential(*net)
        init_weights(self.net, init_type)

    def forward(self, x, mask, is_real_x):
        if is_real_x:
            h, w = mask.size(2), mask.size(3)
            left_h = [random.randint(0, h//self.patch_size[0]-1) * self.patch_size[0] for i in range(x.size(0))]
            left_w = [random.randint(0, w//self.patch_size[1]-1) * self.patch_size[1] for i in range(x.size(0))]
        else:
            left_h, left_w = self.proposal_net(mask)
        patch = [x[i:(i+1), :, left_h[i]:(left_h[i]+self.patch_size[0]), left_w[i]:(left_w[i]+self.patch_size[1])] for i in range(x.size(0))]
        return self.net(torch.cat(patch, 0))


def make_vgg_block(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


'''Only define the part we consider'''
class VGG16(nn.Module):
    def __init__(self, batch_norm=False, pretrain_model='vgg_16.pth'):
        if not os.path.exists(pretrain_model):
            raise ValueError("Pretrain model file of VGG16 not found.")
        super(VGG16, self).__init__()
        cfg = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M']]  # up to pool3 layer
        in_channels = [3, 64, 128]
        self.vgg = nn.ModuleList([make_vgg_block(cfg[i], in_channels[i], batch_norm) for i in range(len(cfg))])
        self.vgg.load_state_dict(torch.load(pretrain_model), strict=False)
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = []
        for i in range(len(self.vgg)):
            x = self.vgg[i](x)
            features.append(x)
        return features


# class PixelLoss(nn.Module):
#     def __init__(self, p):
#         super(PixelLoss, self).__init__()
#         assert p in [1, 2]
#         self.p = p

#     def forward(self, x):
#         size = reduce(lambda a,b: a*b, x.size())
#         return torch.norm(x, p=self.p)**self.p / size
class PixelLoss(nn.Module):
    def __init__(self, p):
        super(PixelLoss, self).__init__()
        assert p in [1, 2]
        self.p = p

    def forward(self, x, gt, mask):
        return torch.norm((x-gt)*mask, p=self.p)**self.p / torch.sum(mask) / x.size(1)
        

class StyleLoss(nn.Module):
    def __init__(self, p=1):
        super(StyleLoss, self).__init__()
        assert p in [1, 2]
        self.p = p 
        self.loss = nn.L1Loss() if self.p == 1 else nn.MSELoss()
        
    def forward(self, x, gt):
        x = [xi.view(xi.size(0)*xi.size(1), -1) for xi in x]
        gt = [gti.view(gti.size(0)*gti.size(1), -1) for gti in gt]
        gm_x = [torch.mm(xi, xi.t())/xi.size(0)/xi.size(1) for xi in x]
        gm_gt = [torch.mm(gti, gti.t())/gti.size(0)/gti.size(1) for gti in gt]
        return sum([self.loss(gm_xi, gm_gti.detach()) for gm_xi, gm_gti in zip(gm_x, gm_gt)])
        

class PerceptualLoss(nn.Module):
    def __init__(self, p=1):
        super(PerceptualLoss, self).__init__()
        assert p in [1, 2]
        self.p = p 
        self.loss = nn.L1Loss() if self.p == 1 else nn.MSELoss()
        
    def forward(self, x, gt):
        return sum([self.loss(xi, gti.detach()) for xi, gti in zip(x, gt)])


class TVLoss(nn.Module):
    def __init__(self, p=1):
        super(TVLoss, self).__init__()
        assert p in [1, 2]
        self.p = p

    def forward(self, x):
        if self.p == 1:
            loss = torch.sum(torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])) + torch.sum(torch.abs(x[:,:,:,:-1] - x[:,:,:,1:]))
        else:
            loss = torch.sum(torch.sqrt((x[:,:,:-1,:] - x[:,:,1:,:])**2) + torch.sum((x[:,:,:,:-1] - x[:,:,:,1:])**2))
        loss = loss / x.size(0) / (x.size(2)-1) / (x.size(3)-1)
        return loss
