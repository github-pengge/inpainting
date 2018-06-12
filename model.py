from networks import VGG16, Discriminator, ProgressiveGrowingDiscriminator, PatchDiscriminator, \
                    LocalDiscriminator, StyleLoss, PerceptualLoss, TVLoss, PixelLoss
import os
import time
import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import Adam
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from itertools import chain
import math


class InpaintingModel:
    def __init__(self, G, data, config):
        self.G = G 
        self.data = data
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config['gpu']))
        self.use_gpu = len(self.config['gpu']) > 0
        self.initialize()

    def initialize(self):
        if self.use_gpu:
            device_ids = range(len(self.config['gpu']))
            self.G = nn.DataParallel(self.G, device_ids=device_ids).cuda(0)
        if self.config['phase'] in ['train', 'finetune']:
            self.writer = SummaryWriter(log_dir=self.config['model_dir'])
            if self.config['use_vgg']:
                self.VGG = VGG16(pretrain_model=self.config['vgg_model'])
                if self.use_gpu:
                    self.VGG = nn.DataParallel(self.VGG, device_ids=device_ids).cuda(0)
                self.VGG.eval()
                self.style_criterion = StyleLoss(p=1)
                self.perceptual_criterion = PerceptualLoss(p=1)
                if self.use_gpu:
                    self.style_criterion = self.style_criterion.cuda()
                    self.perceptual_criterion = self.perceptual_criterion.cuda()
            if self.config['use_gan']:
                if self.config['progressive_growing']:
                    disc = ProgressiveGrowingDiscriminator
                else:
                    disc = PatchDiscriminator if self.config['patchgan'] else Discriminator
                self.D = disc(pix2pix_style=self.config['pix2pix_style'], init_type=self.config['init_type'])
                if self.use_gpu:
                    self.D = nn.DataParallel(self.D, device_ids=device_ids).cuda(0)
                if self.config['use_local_d']:
                    self.D_local = LocalDiscriminator(pix2pix_style=self.config['pix2pix_style'], init_type=self.config['init_type'], \
                                                        patch_size=self.config['patch_size'])
                    if self.use_gpu:
                        self.D_local = nn.DataParallel(self.D_local, device_ids=device_ids).cuda(0)
                if self.config['use_local_d']:
                    self.trainer_d = Adam(chain(self.D.parameters(), self.D_local.parameters()), lr=self.config[self.config['phase']+'_lr'])
                else:
                    self.trainer_d = Adam(self.D.parameters(), lr=self.config[self.config['phase']+'_lr'])
                self.gan_criterion = nn.BCELoss() if self.config['gan_loss'] == 'bce' else nn.MSELoss()
                if self.use_gpu:
                    self.gan_criterion = self.gan_criterion.cuda()

            if self.config['use_tv_loss']:
                self.tv_criterion = TVLoss(p=1)
                if self.use_gpu:
                    self.tv_criterion = self.tv_criterion.cuda()

            self.pixel_criterion = PixelLoss(p=1)
            if self.use_gpu:
                self.pixel_criterion = self.pixel_criterion.cuda()

            self.trainer_g = Adam(self.G.parameters(), lr=self.config[self.config['phase']+'_lr'])

        self.restore()

    def restore(self):
        self.current_resol = self.config['from_resol'] if self.config['progressive_growing'] else self.config['to_resol']
        self.current_iter = 0
        # TODO: todo == would never do
        pass

    def update_lr(self, mode):
        if mode == 'test':
            return 
        assert hasattr(self, 'trainer_g') 
        for group in self.trainer_g.param_groups:
            group['lr'] = self.config[self.config['phase']+'_lr']

        if self.config['use_gan']:
            assert hasattr(self, 'trainer_d')
            for group in self.trainer_d.param_groups:
                group['lr'] = self.config[self.config['phase']+'_lr']

    def run(self, **kwargs):
        assert self.config['mode'] in ['train', 'train+finetune', 'finetune', 'test']
        if 'train' in self.config['mode']:
            self.train()
        if 'finetune' in self.config['mode']:
            self.finetune()
        if 'test' in self.config['mode']:
            self.test(**kwargs)

    def _training(self, mode, batch_size, from_iter, to_iter, current_resol, fade_in=False):
        self.update_lr(mode=mode)
        if self.use_gpu:
            self.G.module.set_mode(mode)
        else:
            self.G.set_mode(mode)

        self.current_resol = current_resol
        _phase = int(math.log2(current_resol)) - 2  # min resolution = 4
        if fade_in:
            _phase = _phase - 1
        phase = _phase
        use_gan = True if (from_iter >= self.config['start_gan_at'] and self.config['use_gan']) else False
        for it in range(from_iter, to_iter):
            start_time = time.time()
            if fade_in:
                phase = _phase + (it-from_iter+1)/float(to_iter-from_iter+2)
            X, M = self.data(batch_size, current_resol)
            X = Variable(torch.from_numpy(X))
            M = Variable(torch.from_numpy(M))
            if self.use_gpu:
                X = X.cuda()
                M = M.cuda()
            masked_X = X * M
            if self.config['g_input'] == 'masked_X':
                X_out = self.G(masked_X, phase=phase)
            # elif self.config['g_input'] == 'X+mask':
            #     X_out = self.G(X, M)
            elif self.config['g_input'] == 'masked_X+mask':
                X_out = self.G(masked_X, M, phase=phase)
            else:
                raise ValueError("Invalid input combination for G: %s" % self.config['g_input'])

            # update D
            if use_gan:
                if self.config['pix2pix_style']:
                    X_ = torch.cat([X, masked_X], 1)
                    X_out_ = torch.cat([X_out, masked_X], 1)
                else:
                    X_, X_out_ = X, X_out
                # X_out_detach = X_out_.detach()
                # global discriminator
                dx = self.D(X_, phase=phase)
                dx_out = self.D(X_out_.detach(), phase=phase)
                # d_loss = self.gan_criterion(dx, self._one) + self.gan_criterion(dx_out, self._zero)
                d_loss = d_loss_global = self.gan_criterion(dx, torch.ones_like(dx)) + self.gan_criterion(dx_out, torch.zeros_like(dx_out))
                # local discriminator
                if self.config['use_local_d'] and current_resol>=self.config['patch_size']:
                    dx_local = self.D_local(X_, M, True)
                    dx_out_local = self.D_local(X_out_.detach(), M, False)
                    d_loss_local = self.config['lambda_local_gan'] * \
                                (self.gan_criterion(dx_local, torch.ones_like(dx_local)) + self.gan_criterion(dx_out_local, torch.zeros_like(dx_out_local)))
                    d_loss = d_loss + d_loss_local
                self.trainer_d.zero_grad()
                d_loss.backward()
                self.trainer_d.step()

            # update G
            # pixel loss
            # g_losses = {  
            #             'pixel_loss_hole': self.config['lambda_pixel_loss_hole'] * self.pixel_criterion((1-M)*(X_out-X)),
            #             'pixel_loss_valid': self.config['lambda_pixel_loss_valid'] * self.pixel_criterion(M*(X_out-X))   
            #         }
            g_losses = {  
                        'pixel_loss_hole': self.config['lambda_pixel_loss_hole'] * self.pixel_criterion(X, X_out, 1-M),
                        'pixel_loss_valid': self.config['lambda_pixel_loss_valid'] * self.pixel_criterion(X, X_out, M)   
                    }

            if use_gan:
                dx_out = self.D(X_out_, phase=phase)
                # g_losses.update({'gan_loss': self.config['lambda_gan'] * self.gan_criterion(dx_out, self._one)})
                g_losses.update({'gan_loss_global': self.config['lambda_gan'] * self.gan_criterion(dx_out, torch.ones_like(dx_out))})
                if self.config['use_local_d'] and current_resol>=self.config['patch_size']:
                    dx_out_local = self.D_local(X_out_, M, False)
                    g_losses.update({'gan_loss_local': self.config['lambda_gan']*self.config['lambda_local_gan']*self.gan_criterion(dx_out_local, torch.ones_like(dx_out_local))})

            if self.config['use_vgg']:
                # compute X_comp
                X_comp = X_out.clone()
                # idx = (1-M).byte().expand(M.size(0), X.size(1), M.size(2), M.size(3))
                idx = M.byte().expand(M.size(0), X.size(1), M.size(2), M.size(3))
                X_comp[idx] = X[idx]
                # compute VGG feature
                vgg_X = self.VGG(X)
                vgg_X_out = self.VGG(X_out)
                vgg_X_comp = self.VGG(X_comp)

                # compute losses
                g_losses.update({  # perceptual loss
                            'perceptual_loss_out': self.config['lambda_perceptual_loss_out'] * self.perceptual_criterion(vgg_X_out, vgg_X),
                            'perceptual_loss_comp': self.config['lambda_perceptual_loss_comp'] * self.perceptual_criterion(vgg_X_comp, vgg_X),
                            # style loss
                            'style_loss_out': self.config['lambda_style_loss_out'] * self.style_criterion(vgg_X_out, vgg_X),
                            'style_loss_comp': self.config['lambda_style_loss_comp'] * self.style_criterion(vgg_X_comp, vgg_X)
                        })
            # tv loss
            if self.config['use_tv_loss']:
                g_losses.update({'tv_loss': self.config['lambda_tv_loss'] * self.tv_criterion(X_comp)})

            g_loss = sum(g_losses.values())
            self.trainer_g.zero_grad()
            g_loss.backward()
            self.trainer_g.step()

            if use_gan:
                print('%s/%s, %sx%s: d_loss: %.4f, g_loss: %.4f, time: %.2fsec' % \
                    (it, to_iter, current_resol, current_resol, d_loss.data[0], g_loss.data[0], time.time()-start_time))
            else:
                print('%s/%s. %sx%s: loss: %.4f, time: %.2fsec' % \
                    (it, to_iter, current_resol, current_resol, g_loss.data[0], time.time()-start_time))

            # register loss
            if use_gan:
                self.writer.add_scalar('d_loss/d_loss', d_loss.data[0], it)
                self.writer.add_scalar('d_loss/d_loss_global', d_loss_global.data[0], it)
                if self.config['use_local_d'] and current_resol>=self.config['patch_size']:
                    self.writer.add_scalar('d_loss/d_loss_local', d_loss_local.data[0], it)
            for tag, value in g_losses.items():
                self.writer.add_scalar('g_loss/'+tag, value.data[0], it)
            self.writer.add_scalar('g_loss/g_loss', g_loss.data[0], it)

            # register images
            if (it+1-from_iter) % self.config['sample_freq'] == 0 or (it+1) == to_iter:
                for i in range(min(X.size(0), 10)):
                    image = vutils.make_grid([X.cpu().data[i], masked_X.cpu().data[i], X_out.cpu().data[i]], normalize=True, scale_each=True)
                    self.writer.add_image('gt-masked-output/%d'%i, image, it)

            # save model
            if (it+1-from_iter) % self.config['save_freq'] == 0 or (it+1) == to_iter:
                post = '%s_%s_iter_%s.pth' % (current_resol, 'fadeIn' if fade_in else 'stabilize', it+1)
                torch.save(self.G.state_dict(), os.path.join(self.config['model_dir'], 'G_'+post))
                if use_gan:
                    torch.save(self.D.state_dict(), os.path.join(self.config['model_dir'], 'D_'+post))
                    if self.config['use_local_d'] and current_resol>=self.config['patch_size']:
                        torch.save(self.D_local.state_dict(), os.path.join(self.config['model_dir'], 'localD_'+post))

            # check whether it's time to use gan or not
            if self.config['start_gan_at'] <= it and self.config['use_gan']:
                use_gan = True

        self.current_iter = to_iter

    def train(self):
        if self.config['progressive_growing']:
            _from = int(math.log2(self.config['from_resol']))
            _to = int(math.log2(self.config['to_resol']))
            assert 2**_from == self.config['from_resol'] and 2**_to == self.config['to_resol']
            for phase in range(_from, _to+1):
                from_iter = self.current_iter
                current_resol = 2 ** phase
                bs = self.config['bs_map'][current_resol]
                num_iter = (self.config['n_real_per_phase']+bs-1) // bs
                to_iter = from_iter + num_iter 
                self._training('train', bs, from_iter, to_iter, current_resol, fade_in=False)
                if phase != _to: # fade in
                    current_resol = 2 ** (phase+1)
                    from_iter = self.current_iter
                    bs = self.config['bs_map'][current_resol]
                    num_iter = self.config['n_real_per_phase'] // bs
                    to_iter = from_iter + num_iter 
                    self._training('train', bs, from_iter, to_iter, current_resol, fade_in=True)
        else:
            from_iter = self.current_iter
            bs = self.config['bs_map'][self.config['to_resol']]
            num_iter = (self.config['n_real_per_phase']+bs-1) // bs
            to_iter = from_iter + num_iter 
            self._training('train', bs, from_iter, to_iter, self.config['to_resol'], fade_in=False)

    def finetune(self):
        bs = self.config['bs_map'][self.config['to_resol']]
        from_iter = self.current_iter
        to_iter = self.config['finetune_iter'] + from_iter
        self._training('finetune', bs, from_iter, to_iter, self.config['to_resol'], fade_in=False)

    def test(self, *args, **kwargs):
        assert 'X' in kwargs and 'M' in kwargs
        X = kwargs['X']
        M = kwargs['M']

        from scipy.misc import imsave

        X = Variable(torch.from_numpy(X))
        M = Variable(torch.from_numpy(M))
        if self.use_gpu:
            self.G.module.set_mode('test')
            X = X.cuda()
            M = M.cuda()
        else:
            self.G.set_mode('test')
        X_out = self.G(X, M, phase=0)  # only full resolution
        masked_X = X * M
        save_path = os.path.join(self.config['model_dir'], 'test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(X.size(0)):
            image = vutils.make_grid([X.cpu().data[i], masked_X.cpu().data[i], X_out.cpu().data[i]], normalize=True, scale_each=True)
            imsave(os.path.join(save_path, '%s.png'%(i+1)), image.cpu().numpy().transpose(1, 2, 0))


