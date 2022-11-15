# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
import numpy as np


class Renderer(object):

    def __init__(self, generator, discriminator=None, program=None):
        self.generator = generator
        self.discriminator = discriminator
        self.sample_tmp = 0.65
        self.program = program
        self.seed = 0

        if (program is not None) and (len(program.split(':')) == 2):
            from training.dataset import ImageFolderDataset
            self.image_data = ImageFolderDataset(program.split(':')[1])
            self.program = program.split(':')[0]
        else:
            self.image_data = None

    def set_random_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def __call__(self, *args, **kwargs):
        self.generator.eval()  # eval mode...

        if self.program is None:
            if hasattr(self.generator, 'get_final_output'):
                return self.generator.get_final_output(*args, **kwargs)
            return self.generator(*args, **kwargs)
        
        if self.image_data is not None:
            batch_size = 1
            indices = (np.random.rand(batch_size) * len(self.image_data)).tolist()
            rimages = np.stack([self.image_data._load_raw_image(int(i)) for i in indices], 0)
            rimages = torch.from_numpy(rimages).float().to(kwargs['z'].device) / 127.5 - 1
            kwargs['img'] = rimages
        
        outputs = getattr(self, f"render_{self.program}")(*args, **kwargs)
        
        if self.image_data is not None:
            imgs = outputs if not isinstance(outputs, tuple) else outputs[0]
            size = imgs[0].size(-1)
            rimg = F.interpolate(rimages, (size, size), mode='bicubic', align_corners=False)
            imgs = [torch.cat([img, rimg], 0) for img in imgs]
            outputs = imgs if not isinstance(outputs, tuple) else (imgs, outputs[1])
        return outputs

    def get_camera_traj(self, t, batch_size=1, device='cpu'):
        gen = self.generator.synthesis
        range_u, range_v = gen.C.range_u, gen.C.range_v
        pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi/2
        yaw = 0.4 * np.sin(t * 2 * np.pi)
        u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
        v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
        cam = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=device)
        return cam

    def render_front_zoom(self, zoom_factor = 0.1, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov_orig = kwargs["fov"]
        if fov_orig is None:
            fov_orig = self.generator.synthesis.C.fov
        motion_seed = kwargs["motion_seed"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        fov_samples = np.linspace((1.0 + zoom_factor)*fov_orig, (1.0 - zoom_factor)*fov_orig, n_steps)
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device, seed=motion_seed)
        for step in range(n_steps):
            # Set Camera
            fov = fov_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[0.5, 0.5, 0.5], device=ws.device, fov=fov)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out
    
    def render_rotation_camera_yaw(self, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov = kwargs["fov"]
        motion_seed = kwargs["motion_seed"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        relative_range_yaw = kwargs['relative_range_yaw']
        u_samples = np.linspace(relative_range_yaw[0], relative_range_yaw[1], n_steps)
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device, seed=motion_seed)
        for step in range(n_steps):
            # Set Camera
            u = u_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[u, 0.5, 0.5], device=ws.device, fov=fov)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out
    
    def render_rotation_camera_yaw_inv(self, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov = kwargs["fov"]
        motion_seed = kwargs["motion_seed"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        relative_range_yaw = kwargs['relative_range_yaw']
        u_samples = np.linspace(relative_range_yaw[0], relative_range_yaw[1], n_steps)
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device, seed=motion_seed)
        for step in range(n_steps):
            # Set Camera
            u = u_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[u, 0.5, 0.5], device=ws.device, fov=fov)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out[::-1], cameras[::-1]
        else:
            return out[::-1]
    
    def render_rotation_camera_yaw_pitch(self, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov = kwargs["fov"]
        motion_seed = kwargs["motion_seed"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])

        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        relative_range_yaw = kwargs['relative_range_yaw']
        u_samples = np.linspace(relative_range_yaw[0], relative_range_yaw[1], n_steps)
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device, seed=motion_seed)
        for step in range(n_steps):
            # Set Camera
            u = u_samples[step]
            kwargs["camera_matrices"] = self.get_camera_traj(step/n_steps, ws.size(0), device=ws.device)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out
    
    def render_front(self, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov = kwargs["fov"]
        motion_seed = kwargs["motion_seed"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device, seed=motion_seed)
        for step in range(n_steps):
            # Set Camera
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[0.5, 0.5, 0.5], device=ws.device, fov=fov)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out
   
    def render_rotation_camera(self, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov = kwargs["fov"]
        motion_seed = kwargs["motion_seed"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        relative_range_yaw = kwargs['relative_range_yaw']
        u_samples = np.linspace(relative_range_yaw[0], relative_range_yaw[1], n_steps)
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device, seed=motion_seed)
        for step in range(n_steps):
            # Set Camera
            u = u_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[u, 0.5, 0.5], device=ws.device, fov=fov)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out
    
    def render_rotation_camera_pitch(self, *args, **kwargs):
        batch_size, n_steps = kwargs["batch_size"], kwargs["n_steps"]
        timesteps = kwargs["time_steps"]
        fov = kwargs["fov"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
            kwargs.pop('img', None) 

        out = []
        cameras = []
        relative_range_pitch = kwargs['relative_range_pitch']
        v_samples = np.linspace(relative_range_pitch[0], relative_range_pitch[1], n_steps)
        Ts, _, z_motion = self.generator.get_ts(timesteps=timesteps, batch_size = batch_size, device=ws.device)
        for step in range(n_steps):
            # Set Camera
            v = v_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size, mode=[0.5, v, 0.5], device=ws.device, fov=fov)
            cameras.append(kwargs["camera_matrices"])
            with torch.no_grad():
                out_i = gen(ws, Ts, z_motion, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out