import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class DegradationEstimator(nn.Module):
    """Lightweight CNN (~3K params) that predicts sigma scale from LQ image.
    Inspired by DT-Diff (Xu et al., ESWA 2026) trajectory selector."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, lq):
        # Output [0,1] -> map to [0.6, 1.4], default ~1.0
        return self.net(lq) * 0.8 + 0.6


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            # Adaptive sigma: degradation estimator (DT-Diff inspired)
            self.adaptive_sigma = opt['train'].get('adaptive_sigma', False)
            if self.adaptive_sigma:
                self.degradation_estimator = DegradationEstimator().to(self.device)
                self.sigma_reg_weight = opt['train'].get('sigma_reg_weight', 0.1)
            else:
                self.degradation_estimator = None

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            pearson_weight = opt['train'].get('pearson_weight', 0.0)
            edge_weight = opt['train'].get('edge_weight', 0.5)
            hist_weight = opt['train'].get('hist_weight', 0.1)
            freq_weight = opt['train'].get('freq_weight', 0.1)
            self.loss_fn = MatchingLoss(loss_type, is_weighted, pearson_weight,
                                        edge_weight, hist_weight, freq_weight).to(self.device)
            self.weight = opt['train']['weight']
            self.sga_weight = opt['train'].get('sga_weight', self.weight)

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            # Add degradation estimator params to optimizer
            if self.adaptive_sigma and self.degradation_estimator is not None:
                for v in self.degradation_estimator.parameters():
                    if v.requires_grad:
                        optim_params.append(v)

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
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
            self.log_dict = OrderedDict()

    def compute_sigma_scale(self, lq):
        """Compute adaptive sigma scale from LQ image. Returns scalar."""
        if getattr(self, 'adaptive_sigma', False) and getattr(self, 'degradation_estimator', None) is not None:
            scale = self.degradation_estimator(lq)
            return scale.view(-1, 1, 1, 1).mean()  # batch-mean scalar
        return torch.tensor(1.0, device=self.device)

    def feed_data(self, state, LQ, GT=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze())
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        # Adaptive sigma regularization (DT-Diff inspired): keep scale near 1.0
        if self.adaptive_sigma:
            sigma_reg = self.sigma_reg_weight * (sde._sigma_scale - 1.0).pow(2)
            loss = loss + sigma_reg.to(loss.device)
            self.log_dict["sigma_scale"] = sde._sigma_scale.item()

        # Self-Generated Augmentation (SGA, CLOVER): recover x0 from the
        # prediction, re-noise it to a smaller timestep, and add a second
        # x0-space loss aligned to GT. Only timesteps <= SGA_MAX_T are used,
        # because get_x0_from_noise divides by exp(-thetas_cumsum[t]*dt), which
        # explodes for large t (amplification reaches ~193x at t=299).
        SGA_MAX_T = 100
        sga_mask = (timesteps.squeeze() <= SGA_MAX_T)
        do_sga = bool(sga_mask.any())

        # Detach the adaptive sigma scale for the SGA path: its second backward
        # must not traverse the sigma graph already freed by loss.backward().
        if do_sga and self.adaptive_sigma:
            sde._sigma_scale = sde._sigma_scale.detach()

        # Prepare SGA pseudo-samples before freeing the first forward graph.
        if do_sga:
            # Align mu with the masked subset; otherwise get_x0_from_noise and
            # mu_bar would broadcast (N,3,H,W) against the full (B,3,H,W) mu.
            sde.set_mu(self.condition[sga_mask])
            x0_pred = sde.get_x0_from_noise(
                self.state[sga_mask], noise[sga_mask], timesteps[sga_mask]
            ).detach()
            t_sub = timesteps[sga_mask]
            t_2 = (torch.rand_like(t_sub.float()) * t_sub.float()).floor().long() + 1
            state_2 = sde.renoise_state(x0_pred, t_2)

        # Backward the first loss first so its activations are freed before
        # the second forward, keeping peak memory close to the baseline.
        loss.backward()

        loss_sga = None
        if do_sga:
            noise_2 = sde.noise_fn(state_2, t_2.reshape(-1))
            x0_pred_2 = sde.get_x0_from_noise(state_2, noise_2, t_2)
            loss_sga = self.sga_weight * self.loss_fn(x0_pred_2, self.state_0[sga_mask])
            loss_sga.backward()

        self.optimizer.step()
        self.ema.update()

        # set log
        total_loss = loss.item() + (loss_sga.item() if loss_sga is not None else 0.0)
        self.log_dict["loss"] = total_loss

    def test(self, sde=None, save_states=False):
        sde.set_mu(self.condition)

        # Adaptive sigma for inference
        if getattr(self, 'adaptive_sigma', False) and getattr(self, 'degradation_estimator', None) is not None:
            with torch.no_grad():
                sigma_scale = self.compute_sigma_scale(self.condition)
            sde.set_sigma_scale(sigma_scale)

        self.model.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states)

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
