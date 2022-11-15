# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from email import generator
from attr import has

from cv2 import DescriptorMatcher
import training
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import rearrange
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.diffaugment import DiffAugment

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, **kwargs): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(
        self, device, G_mapping, G_synthesis, D, 
        G_encoder=None, augment_pipe=None, D_ema=None,
        style_mixing_prob=0.9, r1_gamma=10, 
        pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, other_weights=None,
        curriculum=None, alpha_start=0.0, cycle_consistency=False, label_smooth=0,
        generator_mode='random_z_random_c', diffaugment='color,translation', img_disc_weight=1.0, video_disc_weight=1.0):

        super().__init__()
        self.device            = device
        self.G_mapping         = G_mapping
        self.G_synthesis       = G_synthesis
        self.G_encoder         = G_encoder
        self.D                 = D
        self.D_ema             = D_ema
        self.augment_pipe      = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma          = r1_gamma
        self.pl_batch_shrink   = pl_batch_shrink
        self.pl_decay          = pl_decay
        self.pl_weight         = pl_weight
        self.other_weights     = other_weights
        self.pl_mean           = torch.zeros([], device=device)
        self.curriculum        = curriculum
        self.alpha_start       = alpha_start
        self.alpha             = None
        self.cycle_consistency = cycle_consistency
        self.label_smooth      = label_smooth
        self.generator_mode    = generator_mode
        self.img_disc_weight = img_disc_weight
        self.video_disc_weight = video_disc_weight
        self.dist1 = torch.distributions.beta.Beta(2., 1., validate_args=None)
        self.dist2 = torch.distributions.beta.Beta(1., 2., validate_args=None)
        if hasattr(self.G_synthesis, "module"):
            self.z_dim_motion = self.G_synthesis.module.z_dim_motion
        else:
            self.z_dim_motion = self.G_synthesis.z_dim_motion
        self.diffaugment = diffaugment

        if self.G_encoder is not None:
            import lpips
            self.lpips_loss      = lpips.LPIPS(net='vgg').to(device=device)
    
    def convert(self, img, Ts):
        img = rearrange(img, '(b t) c h w -> b (t c) h w', t=2)
        Ts = Ts.view(-1, 2, 1, 1)[:, 1:2] - Ts.view(-1, 2, 1, 1)[:, 0:1]
        Ts = Ts * torch.ones_like(img[:, 0:1])
        img = torch.cat([img, Ts], dim=1)

        return img

    def set_alpha(self, steps):
        alpha = None
        if self.curriculum is not None:
            if self.curriculum == 'upsample':
                alpha = 0.0
            else:
                assert len(self.curriculum) == 2, "currently support one stage for now"
                start, end = self.curriculum
                alpha = min(1., max(0., (steps / 1e3 - start) / (end - start)))
                if self.alpha_start > 0:
                    alpha = self.alpha_start + (1 - self.alpha_start) * alpha
        self.alpha = alpha
        self.steps = steps
        self.curr_status = None

        def _apply(m):
            if hasattr(m, "set_alpha") and m != self:
                m.set_alpha(alpha)
            if hasattr(m, "set_steps") and m != self:
                m.set_steps(steps)
            if hasattr(m, "set_resolution") and m != self:
                m.set_resolution(self.curr_status)
        
        self.G_synthesis.apply(_apply)
        self.curr_status = self.resolution
        self.D.apply(_apply)
        if self.G_encoder is not None:
            self.G_encoder.apply(_apply)

    def run_G(self, z, c, sync, img=None, mode=None, get_loss=True):
        # Get motion codes
        batch_size = z.size(0)
        Ts = torch.cat([self.dist1.sample((batch_size, 1, 1, 1, 1)),
                               self.dist2.sample((batch_size, 1, 1, 1, 1))], dim=1).to(self.device)
        Ts = torch.cat([Ts.min(dim=1, keepdim=True)[0], Ts.max(dim=1, keepdim=True)[0]], dim=1)
        Ts = rearrange(Ts, 'b t c h w -> (b t) c h w')
        z_motion = torch.randn(batch_size, self.z_dim_motion).to(z.device)

        synthesis_kwargs = {'camera_mode': 'random'}
        generator_mode   = self.generator_mode if mode is None else mode

        if (generator_mode == 'image_z_random_c') or (generator_mode == 'image_z_image_c'):
            assert (self.G_encoder is not None) and (img is not None)
            with misc.ddp_sync(self.G_encoder, sync):
                ws  = self.G_encoder(img)['ws']
            if generator_mode == 'image_z_image_c':
                with misc.ddp_sync(self.D, False):
                    synthesis_kwargs['camera_RT'] = misc.get_func(self.D, 'get_estimated_camera')[0](img)
            with misc.ddp_sync(self.G_synthesis, sync):
                out = self.G_synthesis(ws, Ts, z_motion, **synthesis_kwargs)            
            if get_loss:  # consistency loss given the image predicted camera (train the image encoder jointly)
                out['consist_l1_loss']    = F.smooth_l1_loss(out['img'], img['img']) * 2.0   # TODO: DEBUG
                out['consist_lpips_loss'] = self.lpips_loss(out['img'],  img['img']) * 10.0  # TODO: DEBUG
            
        elif (generator_mode == 'random_z_random_c') or (generator_mode == 'random_z_image_c'):
            with misc.ddp_sync(self.G_mapping, sync):
                ws  = self.G_mapping(z, c)
                if self.style_mixing_prob > 0:
                    with torch.autograd.profiler.record_function('style_mixing'):
                        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                        cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                        ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
            if generator_mode == 'random_z_image_c':
                assert img is not None
                with torch.no_grad():
                    D = self.D_ema if self.D_ema is not None else self.D
                    with misc.ddp_sync(D, sync):
                        estimated_c = misc.get_func(D, 'get_estimated_camera')(img)[0].detach()
                        if estimated_c.size(-1) == 16:
                            synthesis_kwargs['camera_RT'] = estimated_c
                        if estimated_c.size(-1) == 3:
                            synthesis_kwargs['camera_UV'] = estimated_c
            with misc.ddp_sync(self.G_synthesis, sync):
                out = self.G_synthesis(ws, Ts, z_motion, **synthesis_kwargs)
        else:
            raise NotImplementedError(f'wrong generator_mode {generator_mode}')
        return out, ws, Ts
    
    def run_D(self, img, c, sync):
        if self.diffaugment:
            img = torch.cat([
                    rearrange(DiffAugment(img, policy=self.diffaugment), '(b t) c h w -> b (t c) h w', t=2),
                    img[:, 6:7]], dim=1
                  )

        with misc.ddp_sync(self.D, sync):
            logits, img_logits, temp_imgs = self.D(img, c, aug_pipe=self.augment_pipe)
        return logits, img_logits, temp_imgs

    def get_loss(self, outputs, module='D'):
        reg_loss, logs, del_keys = 0, [], []
        if isinstance(outputs, dict):
            for key in outputs:
                if key[-5:] == '_loss':
                    logs += [(f'Loss/{module}/{key}', outputs[key])]
                    del_keys += [key]
                    if (self.other_weights is not None) and (key in self.other_weights):
                        reg_loss = reg_loss + outputs[key].mean() * self.other_weights[key]
                    else:
                        reg_loss = reg_loss + outputs[key].mean()
            for key in del_keys:
                del outputs[key]
            for key, loss in logs:
                training_stats.report(key, loss)
        return reg_loss

    @property
    def resolution(self):
        return misc.get_func(self.G_synthesis, 'get_current_resolution')()[-1]

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, fake_img, sync, gain, scaler=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        losses   = {}
        if isinstance(real_img, dict): real_img = real_img['img']
        # Gmain: Maximize logits for generated images.
        loss_Gmain, reg_loss_Gmain, reg_loss_Gimg_main = 0, 0, 0
        if isinstance(fake_img, dict): fake_img = fake_img['img']
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, _gen_Ts = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), img=fake_img) # May get synced by Gpl.
                reg_loss_Gmain += self.get_loss(gen_img, 'G')
                if isinstance(gen_img, dict): gen_img = gen_img['img']
                gen_img = self.convert(gen_img, _gen_Ts)
                gen_logits, gen_img_logits, _ = self.run_D(gen_img, gen_c, sync=False)
                reg_loss_Gmain += self.get_loss(gen_logits, 'G') #TODO: Check if both needed
                reg_loss_Gimg_main += self.get_loss(gen_img_logits, 'G') #TODO: Check if both needed

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gimg_main = torch.nn.functional.softplus(-gen_img_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Loss/G/Iloss', loss_Gimg_main)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain = loss_Gmain + reg_loss_Gmain
                loss_Gimg_main = loss_Gimg_main + reg_loss_Gimg_main
                losses['Gmain'] = (loss_Gmain.mean() * self.video_disc_weight + loss_Gimg_main.mean() * self.img_disc_weight).mul(gain)
                loss = losses['Gmain']
                loss.backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                # batch_size = gen_z.shape[0] // self.pl_batch_shrink # TODO: Maybe max(1, batch_size)
                batch_size = max(1, gen_z.shape[0] // self.pl_batch_shrink)
                gen_img, gen_ws, _gen_Ts = self.run_G(
                    gen_z[:batch_size], gen_c[:batch_size], sync=sync,
                    img=fake_img[:batch_size] if fake_img is not None else None
                    )
                if isinstance(gen_img, dict):  gen_img = gen_img['img']
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                losses['Gpl'] = (gen_img[:batch_size, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain)
                loss = losses['Gpl']
                loss.backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen, reg_loss_Dgen, reg_loss_Dimggen = 0, 0, 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, _gen_Ts = self.run_G(gen_z, gen_c, sync=False, img=fake_img)
                reg_loss_Dgen += self.get_loss(gen_img, 'D')
                if isinstance(gen_img, dict):  gen_img = gen_img['img']
                gen_img = self.convert(gen_img, _gen_Ts)
                gen_logits, gen_img_logits, _ = self.run_D(gen_img, gen_c, sync=False)
                reg_loss_Dgen += self.get_loss(gen_logits, 'D') #TODO: Check if both needed
                reg_loss_Dimggen += self.get_loss(gen_img_logits, 'D') #TODO: Check if both needed
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                loss_Dimg_gen = torch.nn.functional.softplus(gen_img_logits)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen = loss_Dgen + reg_loss_Dgen
                loss_Dimg_gen = loss_Dimg_gen + reg_loss_Dimggen
                losses['Dgen'] = (loss_Dgen.mean() * self.video_disc_weight + loss_Dimg_gen.mean() * self.img_disc_weight).mul(gain)
                loss = losses['Dgen']
                loss.backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                #print(real_img.size())
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_img_logits, temp_imgs = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_Dimg_real = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dimg_real = torch.nn.functional.softplus(-real_img_logits)

                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    training_stats.report('Loss/D/Iloss', loss_Dimg_gen + loss_Dimg_real)

                loss_Dr1 = 0
                loss_Dimg_r1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        if self.video_disc_weight != 0.0:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()* self.video_disc_weight], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                            r1_penalty = r1_grads.square().sum([1,2,3])
                        else:
                            r1_grads = torch.zeros_like(real_logits)
                            r1_penalty = r1_grads.square().sum()
                        if self.img_disc_weight != 0.0:
                            r1_img_grads = torch.autograd.grad(outputs=[real_img_logits.sum() * self.img_disc_weight], inputs=[temp_imgs], create_graph=True, only_inputs=True)[0]
                        else:
                            r1_img_grads = torch.zeros_like(real_img_logits)
                    r1_img_penalty = r1_img_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    loss_Dimg_r1 = r1_img_penalty * (self.r1_gamma / 2)

                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                losses['Dr1'] = ((real_logits * 0 + loss_Dreal * self.video_disc_weight + loss_Dr1).mean() +
                 (real_img_logits * 0 + loss_Dimg_real * self.img_disc_weight + loss_Dimg_r1).mean()).mul(gain)
                loss = losses['Dr1']
                loss.backward()

        return losses