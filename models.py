import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
from modules.base_models.autoencoder import DiagonalGaussianDistribution
import util
import json
import modules
import modules.attention
import modules.diffusion
import modules.diffusion.util
from modules.ema import LitEma

# import pytorch lightning
import pytorch_lightning as pl
from contextlib import contextmanager

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DiffusionModel(pl.LightningModule):

    def __init__(self, model, loss, flow, t_sampler, 
                 image_key, condition_key=None, condition_stage_model = None,
                 freeze_condition_stage_model=False,
                 discrete_max_timesteps=100, first_stage_model = None, 
                 ckpt_path=None, base_lr=1e-5, use_ema=True, compile_model=False):
        
        super(DiffusionModel, self).__init__()
        self.discrete_max_timesteps = discrete_max_timesteps
        self.image_key = image_key
        self.condition_key = condition_key
        self.condition_stage_model = condition_stage_model
        self.freeze_condition_stage_model = freeze_condition_stage_model

        self.model = model
        self.compile_model = compile_model

        if self.compile_model:
            self.model_compiled = [torch.compile(model)]
    
        self.flow = flow
        self.loss = loss
        self.t_sampler = t_sampler
        self.base_lr = base_lr
        self.first_stage_model = first_stage_model
        self.use_ema = use_ema

        self.configure_models()

        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.load_state_dict(ckpt["state_dict"])

    def configure_optimizers(self):
        
        ps = [
                {"params": self.model.parameters()},
            ]
        
        if self.condition_stage_model is not None and not self.freeze_condition_stage_model:
            ps.append({"params": self.condition_stage_model.parameters()})
        
        return optim.AdamW(ps, lr=self.base_lr)
    
    def configure_models(self):

        # freeze the first stage model
        if self.first_stage_model is not None:
            self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False

        if self.condition_stage_model is not None and self.freeze_condition_stage_model:
            self.condition_stage_model.eval()
            self.condition_stage_model.train = disabled_train
            for param in self.condition_stage_model.parameters():
                param.requires_grad = False
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def first_stage_encode(self, x):
        with torch.no_grad():
            distribution = self.first_stage_model.encode(x)

            if isinstance(distribution, DiagonalGaussianDistribution):
                z = distribution.sample().detach()
            elif isinstance(distribution, torch.Tensor):
                z = distribution.detach()

            del distribution

        return z
    
    def first_stage_decode(self, z):
        with torch.no_grad():
            x = self.first_stage_model.decode(z).detach()
        return x
    
    def condition_stage_encode(self, c):
        if self.condition_stage_model is not None:
            return self.condition_stage_model(c)
        return c
    
    def apply_model(self, x, t, c=None, **kwargs):

        t = (t * self.discrete_max_timesteps).long()

        input_x = x
        
        if c is not None:
            input_x = torch.cat([input_x, c], dim=1)

        if self.compile_model:
            m = self.model_compiled[0]
        else:
            m = self.model

        v_pred = m(input_x, t)

        return v_pred
    
    def preprocess_input(self, batch):
        
        x0 = batch[self.image_key] # B x H x W x C
        x0 = x0.permute(0, 3, 1, 2) # B x C x H x W
        if self.first_stage_model is not None:
            x0 = self.first_stage_encode(x0) # B x c x h x w
        
        c = None
        if self.condition_key is not None:
            c = batch[self.condition_key]
            c = c.permute(0, 3, 1, 2) # B x C x H x W

            c = self.condition_stage_encode(c)
        
        return x0, c
    
    def training_step(self, batch, batch_idx):
        
        x0, c = self.preprocess_input(batch)
        
        batch_size = x0.size(0)

        t = self.t_sampler.sample(batch_size, x0.device) # [0, 1]
        
        e = torch.randn_like(x0, device=x0.device) 

        xt = self.flow.forward(x0, e, t)
        vt = self.flow.velocity(x0, e, t)
        # sigma = 0.001
        # xt += torch.randn_like(xt) * sigma
        
        v_pred = self.apply_model(xt, t, c)

        loss = self.loss(v_pred, vt)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        if batch_idx == 0:
            self.test_step(batch=batch, batch_idx=batch_idx)
         
        x0, c = self.preprocess_input(batch)
        
        batch_size = x0.size(0)

        t = self.t_sampler.sample(batch_size, x0.device) # [0, 1]
        
        e = torch.randn_like(x0, device=x0.device) 

        xt = self.flow.forward(x0, e, t)
        vt = self.flow.velocity(x0, e, t)
        # sigma = 0.001
        # xt += torch.randn_like(xt) * sigma
        
        with self.ema_scope():
            v_pred = self.apply_model(xt, t, c)
            loss = self.loss(v_pred, vt)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    
    def test_step(self, batch, batch_idx):

        # # set torch and numpy seed
        # torch.manual_seed(0)
        # np.random.seed(0)
        
        start_str = "test/"
        log_dir = self.logger.log_dir
        image_folder = os.path.join(log_dir, start_str + "images")
        os.makedirs(image_folder, exist_ok=True)

        batch_size = batch[self.image_key].shape[0]
        device = batch[self.image_key].device

        # create a noise like image
        x0, c = self.preprocess_input(batch)
        xt = torch.randn_like(x0, device=device)
        self.log_images(x0, image_folder, 0, batch_idx, init_str="recons_")
        self.log_images(c, image_folder, 0, batch_idx, init_str="cond_")
        del x0

        with self.ema_scope():
            test_step_size = 1 / self.discrete_max_timesteps
            
            discrete_t = self.discrete_max_timesteps
            t = 1

            while t > 0: 
                
                # denoise z using the model, the model predicts the noise
                t_tensor = torch.full((batch_size, ), t, dtype=torch.float32, device=device)
                vtt = self.apply_model(xt, t_tensor, c)
                vt = vtt.detach()
                del vtt

                # denoise the image
                new_xt = xt - vt * test_step_size 

                if True:
                    new_t_tensor = torch.full((batch_size, ), t - test_step_size, dtype=torch.float32, device=device)
                    new_vt = self.apply_model(new_xt, new_t_tensor, c)

                    # apply heun's method
                    new_xt = xt - 0.5 * (vt + new_vt) * test_step_size

                    del new_t_tensor, new_vt

                # delete intermediates
                del xt, vt

                xt = new_xt

                # save the denoised image
                if (discrete_t) % 10 == 1:
                    self.log_images(xt, image_folder, discrete_t, batch_idx)

                discrete_t -= 1
                t -= test_step_size

        self.log_images(xt, image_folder, 0, batch_idx, init_str="final_recons")
        return 0
    
    def log_images(self, xt, image_folder, i, batch_idx, init_str=""):
        batch_size = xt.size(0)

        if self.first_stage_model is not None:
            decoded_image = self.first_stage_decode(xt)
        else:
            decoded_image = xt

        if i == 1:
            # save decoded image as npy
            np.save(os.path.join(image_folder, f"{init_str}decoded_image.npy"), decoded_image.cpu().detach().numpy())

        for j in range(batch_size):
            img = decoded_image[j].cpu().detach().numpy() # C x H x W
            img = img.transpose((1, 2, 0)) # H x W x C

            img = img[:,:,0] # get the first channel
            
            # normalize
            img = (img - img.min()) / (img.max() - img.min())
            
            img = (img * 255)
            
            img = img.astype(np.uint8)

            cv2.imwrite(os.path.join(image_folder, f"{init_str}{batch_idx}_{j}_{i:03d}.png"), img)

    def forward(self, x, timesteps):
        return None
        # return self.model(x, timesteps)




