# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import hydra
import torch

from torchmetrics import Metric


class TemosComputeLosses(Metric):
    def __init__(self, vae: bool,
                 mode: str,
                 loss_on_both: bool = False,
                 motion_branch: bool = False,
                 loss_on_jfeats: bool = True,
                 ablation_no_kl_combine: bool = False,
                 ablation_no_motionencoder: bool = False,
                 ablation_no_kl_gaussian: bool = False,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Save parameters
        self.vae = vae
        self.mode = mode
        self.motion_branch = motion_branch

        self.loss_on_both = loss_on_both
        self.ablation_no_kl_combine = ablation_no_kl_combine
        self.ablation_no_kl_gaussian = ablation_no_kl_gaussian
        self.ablation_no_motionencoder = ablation_no_motionencoder

        self.loss_on_jfeats = loss_on_jfeats
        losses = []
        if mode == "xyz" or loss_on_jfeats:
            if motion_branch:
                losses.append("recons_jfeats2jfeats")
            losses.append("recons_text2jfeats")
        if mode == "smpl":
            if motion_branch:
                losses.append("recons_rfeats2rfeats")
            losses.append("recons_text2rfeats")
        else:
            ValueError("This mode is not recognized.")

        if vae or loss_on_both:
            kl_losses = []
            if not ablation_no_kl_combine and not ablation_no_motionencoder:
                kl_losses.extend(["kl_text2motion", "kl_motion2text"])
            if not ablation_no_kl_gaussian:
                if not motion_branch:
                    kl_losses.extend(["kl_text"])
                else:
                    kl_losses.extend(["kl_text", "kl_motion"])
            losses.extend(kl_losses)
        if not self.vae or loss_on_both:
            if motion_branch:
                losses.append("latent_manifold")
        losses.append("total")

        for loss in losses:
            self.add_state(loss, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: hydra.utils.instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss != "total"}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, ds_text=None, ds_motion=None, ds_ref=None,
               lat_text=None, lat_motion=None, dis_text=None,
               dis_motion=None, dis_ref=None):
        total: float = 0.0

        if self.mode == "xyz" or self.loss_on_jfeats:
            if self.motion_branch:
                total += self._update_loss("recons_jfeats2jfeats", ds_motion.jfeats, ds_ref.jfeats)
            total += self._update_loss("recons_text2jfeats", ds_text.jfeats, ds_ref.jfeats)
        if self.mode == "smpl":
            if self.motion_branch:
                total += self._update_loss("recons_rfeats2rfeats", ds_motion.rfeats, ds_ref.rfeats)
            total += self._update_loss("recons_text2rfeats", ds_text.rfeats, ds_ref.rfeats)

        if self.vae or self.loss_on_both:
            if not self.ablation_no_kl_combine and self.motion_branch:
                total += self._update_loss("kl_text2motion", dis_text, dis_motion)
                total += self._update_loss("kl_motion2text", dis_motion, dis_text)
            if not self.ablation_no_kl_gaussian:
                total += self._update_loss("kl_text", dis_text, dis_ref)
                if self.motion_branch:
                    total += self._update_loss("kl_motion", dis_motion, dis_ref)
        if not self.vae or self.loss_on_both:
            if self.motion_branch:
                total += self._update_loss("latent_manifold", lat_text, lat_motion)

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss)/count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"losses/{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"losses/{loss_type}/{name}/{split}"
        return log_name
