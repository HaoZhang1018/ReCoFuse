import logging
from collections import OrderedDict
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

from ema_pytorch import EMA
import copy
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss,fusion_base_loss
from models.modules.TIM_model import TIM
from .base_model import BaseModel
import torch.distributed as dist

logger = logging.getLogger("base")

class DiffusionModel(BaseModel):
    def __init__(self, opt):
        super(DiffusionModel, self).__init__(opt)

        os.makedirs('image', exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt["train"]
        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)  
        self.latent_model = networks.define_L(opt).to(self.device)
        self.fusion_model = networks.define_F(opt).to(self.device)
      
        for param in self.latent_model.parameters():
            param.requires_grad = False

        if opt["dist"]:
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])
            self.fusion_model = DistributedDataParallel(self.fusion_model, device_ids=[torch.cuda.current_device()])

        self.load()

        self.encode = self.latent_model.encode
        self.decode = self.latent_model.decode

        if self.is_train:
            self.model.train()
            self.fusion_model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.loss_fn_fuse = fusion_base_loss().to(self.device)
            self.weight = opt['train']['weight']

            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params_vis = []
            optim_params_ir = []
            optim_params_fused = []

            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    if 'vis_' in k:
                        optim_params_vis.append(v)
                    elif 'ir_' in k:
                        optim_params_ir.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning(f"Unclassified model param: {k}")
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))


            for k, v in self.fusion_model.named_parameters():
                if v.requires_grad:
                    optim_params_fused.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(f"FusionModel params [{k}] will not optimize.")

            if train_opt['optimizer'] == 'Adam':
                self.optimizer_vis = torch.optim.Adam(
                    optim_params_vis,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_ir = torch.optim.Adam(
                    optim_params_ir,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_fused = torch.optim.Adam(
                    optim_params_fused,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer_vis = torch.optim.AdamW(
                    optim_params_vis,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_ir = torch.optim.AdamW(
                    optim_params_ir,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_fused = torch.optim.AdamW(
                    optim_params_fused,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer_vis = Lion(
                    optim_params_vis, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_ir = Lion(
                    optim_params_ir, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_fused = Lion(
                    optim_params_fused, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                self.optimizer_vis = torch.optim.Adam(
                    optim_params_vis,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_ir = torch.optim.Adam(
                    optim_params_ir,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_fused = torch.optim.Adam(
                    optim_params_fused,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            self.optimizers.append(self.optimizer_vis)
            self.optimizers.append(self.optimizer_ir)
            self.optimizers.append(self.optimizer_fused)

            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.ema_fusion = EMA(self.fusion_model, beta=0.995, update_every=10).to(self.device)

            self.log_dict = OrderedDict()

    def feed_data(self, vis, vis_cond, ir, ir_cond, gt_vis=None, gt_ir=None):
        self.vis = vis.to(self.device)
        self.vis_cond = vis_cond.to(self.device)
        self.ir = ir.to(self.device)
        self.ir_cond = ir_cond.to(self.device)
        self.gt_vis = gt_vis.to(self.device) if gt_vis is not None else None
        self.gt_ir = gt_ir.to(self.device) if gt_ir is not None else None


    def optimize_parameters(self, step, timesteps, h_X, h_Y, sde=None):
        print(f"[{step}] optimize_parameters called")
        timesteps = timesteps.to(self.device)

        self.optimizer_vis.zero_grad()
        self.optimizer_ir.zero_grad()
        self.optimizer_fused.zero_grad()
        timesteps_zero = torch.zeros_like(timesteps.squeeze())
        vis_ir_fused = self.fusion_model(self.vis, self.ir,timesteps.squeeze())
        vis_pred_noise, ir_pred_noise = self.model(vis_ir_fused, self.vis_cond, vis_ir_fused, self.ir_cond, timesteps.squeeze())
        mu_concat = torch.cat([self.vis_cond, self.ir_cond], dim=0)
        sde.set_mu(mu_concat)
        vis_score = sde.get_score_from_noise(vis_pred_noise, timesteps)
        ir_score = sde.get_score_from_noise(ir_pred_noise, timesteps)
        vis_expection, ir_expection = sde.reverse_sde_step_mean(torch.cat([vis_ir_fused, vis_ir_fused], dim=0), torch.cat([vis_score, ir_score], dim=0), timesteps)
        vis_optimum, ir_optimum = sde.reverse_optimum_step(torch.cat([self.vis, self.ir], dim=0), torch.cat([self.gt_vis, self.gt_ir], dim=0), timesteps)

        vis_loss = self.weight * self.loss_fn(vis_expection, vis_optimum)
        ir_loss = self.weight * self.loss_fn(ir_expection, ir_optimum)
        loss = vis_loss + ir_loss

        loss.backward()
        self.optimizer_vis.step()
        self.optimizer_ir.step()
        self.optimizer_fused.step()

        self.optimizer_fused.zero_grad()
        self.optimizer_vis.zero_grad()
        self.optimizer_ir.zero_grad()
        fuse = self.fusion_model(self.gt_vis, self.gt_ir,timesteps_zero) 
        with torch.no_grad():
            vis_GT = self.decode(self.gt_vis, h_X)
            ir_GT = self.decode(self.gt_ir, h_Y)
        h_fuse = [(hx + hy) / 2 for hx, hy in zip(h_X, h_Y)]
        fuse_decode = self.decode(fuse, h_fuse)
        loss_fuse, loss_fuse_int, loss_fuse_grad, loss_fuse_color = self.loss_fn_fuse(vis_GT, ir_GT, fuse_decode)
        loss_fuse = loss_fuse * self.weight

        loss_fuse.backward()
        self.optimizer_fused.step()

        self.ema.update()
        self.ema_fusion.update()

        self.log_dict["loss_vis"] = vis_loss.item()
        self.log_dict["loss_ir"] = ir_loss.item()
        self.log_dict["loss_fuse"] = loss_fuse.item()
        self.log_dict["loss_fuse_int"] = loss_fuse_int.item()
        self.log_dict["loss_fuse_grad"] = loss_fuse_grad.item()
        self.log_dict["loss_fuse_color"] = loss_fuse_color.item()

    def val(self, step,sde=None, hidden_X=None,hidden_Y=None, perform_ode=False,save_states=False,idx=0):
        print("call val")
        self.model.eval()
        self.fusion_model.eval()
        
        mu_concat = torch.cat([self.vis_cond, self.ir_cond], dim=0)
        sde.set_mu(mu_concat)
        
        reverse_outs = sde.reverse_sde_dual(torch.cat([self.vis, self.ir], dim=0),save_states=save_states,fusion_model=self.fusion_model) 
        B = reverse_outs.shape[0]//2
        latent_vis=reverse_outs[:B]
        latent_ir = reverse_outs[B:]

        with torch.no_grad():
                
            fuse = self.fusion_model(latent_vis,latent_ir,0)
            h_fuse = [(hx + hy) / 2 for hx, hy in zip(hidden_X, hidden_Y)]

            self.output_fuse = self.decode(fuse, h_fuse)

            self.output_vis = self.decode(latent_vis, hidden_X)
            self.output_ir = self.decode(latent_ir, hidden_Y)

        self.model.train()
        self.fusion_model.train()

        vis_filename = "vis_{}_{}.png".format(step, idx)
        out_vis_path =  os.path.join(self.opt["path"]["val_images"], vis_filename)
        tvutils.save_image(self.output_vis.data, out_vis_path, normalize=False)

        ir_filename = "ir_{}_{}.png".format(step, idx)
        out_ir_path =  os.path.join(self.opt["path"]["val_images"], ir_filename)
        tvutils.save_image(self.output_ir.data, out_ir_path, normalize=False)

        # vis_gt_filename = "vis_gt_{}.png".format(idx)
        # vis_gt_path =  os.path.join(self.opt["path"]["val_images"], vis_gt_filename)
        # tvutils.save_image(self.gt_vis.data, vis_gt_path, normalize=False)

        # ir_gt_filename = "ir_gt_{}.png".format(idx)
        # ir_gt_path =  os.path.join(self.opt["path"]["val_images"], ir_gt_filename)
        # tvutils.save_image(self.gt_ir.data, ir_gt_path, normalize=False)

        fuse_filename = "fuse_{}_{}.png".format(step, idx)
        out_fuse_path =  os.path.join(self.opt["path"]["val_images"], fuse_filename)
        tvutils.save_image(self.output_fuse.data, out_fuse_path, normalize=False)        

        
    def test(self, step,sde=None, hidden_X=None,hidden_Y=None,save_states=False,img_name=None):
        print("call test")
        self.model.eval()
        self.fusion_model.eval()
        
        mu_concat = torch.cat([self.vis_cond, self.ir_cond], dim=0)
        sde.set_mu(mu_concat)

        reverse_outs= sde.reverse_sde_dual(torch.cat([self.vis, self.ir], dim=0),save_states=save_states,fusion_model=self.fusion_model) 
       
        B = reverse_outs.shape[0]//2
        latent_vis=reverse_outs[:B]
        latent_ir = reverse_outs[B:]
        with torch.no_grad():
            fuse = self.fusion_model(latent_vis,latent_ir,0)
            h_fuse = [(hx + hy) / 2 for hx, hy in zip(hidden_X, hidden_Y)]
            self.output_fuse = self.decode(fuse, h_fuse)
            self.output_vis = self.decode(latent_vis, hidden_X)
            self.output_ir = self.decode(latent_ir, hidden_Y)

        self.model.train()
        self.fusion_model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out = OrderedDict()
        out["Input_VIS"] = self.vis.detach()[0].float().cpu()
        out["Output_VIS"] = self.output_vis.detach()[0].float().cpu()
        out["Input_IR"] = self.ir.detach()[0].float().cpu()
        out["Output_IR"] = self.output_ir.detach()[0].float().cpu()
        out["Output_Fuse"] = self.output_fuse.detach()[0].float().cpu()
        if need_GT:
            out["GT_VIS"] = self.gt_vis.detach()[0].float().cpu()
            out["GT_IR"] = self.gt_ir.detach()[0].float().cpu()
        return out

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, (nn.DataParallel, DistributedDataParallel)):
            net_struc_str = f"{self.model.__class__.__name__} - {self.model.module.__class__.__name__}"
        else:
            net_struc_str = f"{self.model.__class__.__name__}"
        if self.rank <= 0:
            logger.info(f"Network G structure: {net_struc_str}, with parameters: {n:,d}")
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        load_path_F = self.opt["path"]["pretrain_model_F"]
        load_path_VIS = self.opt["path"]["pretrain_model_VIS"]
        load_path_IR = self.opt["path"]["pretrain_model_IR"]

        if load_path_G is not None and load_path_F is not None:
            logger.info(f"Loading model for G [{load_path_G}] ...")
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])
            logger.info(f"Loading model for F [{load_path_F}] ...")
            self.load_network(load_path_F, self.fusion_model, self.opt["path"]["strict_load"])
        elif load_path_VIS and load_path_IR:
            logger.info(f"Loading VIS branch from [{load_path_VIS}] ...")
            self.load_network(
                load_path_VIS, 
                self.model, 
                partial=True,
                rename_prefix=[
                    ("encoders", "vis_encoders"),
                    ("decoders", "vis_decoders"),
                    ("intro", "vis_intro"),
                    ("ending", "vis_ending"),
                    ("middle_blks", "vis_middle_blks"),
                    ("ups", "vis_ups"),
                    ("downs", "vis_downs"),
                    ("time_mlp", "vis_time_mlp"),  
                ]
            )
            
            logger.info(f"Loading IR branch from [{load_path_VIS}] ...")
            self.load_network(
                load_path_IR, 
                self.model, 
                partial=True,
                rename_prefix=[
                    ("encoders", "ir_encoders"),
                    ("decoders", "ir_decoders"),
                    ("intro", "ir_intro"),
                    ("ending", "ir_ending"),
                    ("middle_blks", "ir_middle_blks"),
                    ("ups", "ir_ups"),
                    ("downs", "ir_downs"),
                    ("time_mlp", "ir_time_mlp"),
                ]
            )
        else:
            raise ValueError("Either both G and F pretrain models or both VIS and IR pretrain models should be provided.")
        load_path_L = self.opt["path"]["pretrain_model_L"]
        if load_path_L is not None:
            logger.info(f"Loading model for L [{load_path_L}] ...")
            self.load_network(load_path_L, self.latent_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.fusion_model, "F", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
        self.save_network(self.ema_fusion.ema_model, "EMAF", 'lastest')
