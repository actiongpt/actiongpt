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

from typing import List, Optional

import torch
import numpy as np
from hydra.utils import instantiate
from hydra.utils import get_original_cwd

from torch import Tensor
from omegaconf import DictConfig
from teach.model.utils.tools import remove_padding

from teach.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from teach.model.base import BaseModel
from gpt_annotator import GPT_Completion
import json
import os
import sys
import numpy as np

def get_gpt3_texts(labels):
    with open(f'{get_original_cwd()}/gpt3_annotations.json', 'r') as fp:
        existing=json.load(fp)
    gpt3_texts = []
    for label in labels:
        if(label in existing):
            # print("LABEL EXISTS")
            gpt3_texts.extend(existing[label])
        else:
            # print("LABEL DOESN'T EXISTS")
            results = []
            for i in range(4):
                results.append("Describe a person's body movements who is performing the action "+str(label)+" in details")
            gpt3_texts.extend(GPT_Completion(results))
    return gpt3_texts    

class TEMOS(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 motion_branch: Optional[bool] = False,
                 nvids_to_save: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)

        self.motionencoder = instantiate(motionencoder, nfeats=nfeats)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)
        self.motion_branch = motion_branch
        self._losses = MetricCollection({split: instantiate(losses, vae=vae,
                                                            motion_branch=motion_branch,
                                                            _recursive_=False)
                                         for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()
        self.nvids_to_save = nvids_to_save
        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        for k, v in self.store_examples.items():
            self.store_examples[k] = {'text': [], 'keyid': [], 'motions':[]}

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict, return_rots=False) -> List[Tensor]:
        gpt3_texts = get_gpt3_texts(batch["text"])
        datastruct_from_text = self.text_to_motion_forward(gpt3_texts,
                                                           batch["length"])
        if return_rots:
            return remove_padding(datastruct_from_text.rots.rots, batch["length"]), remove_padding(datastruct_from_text.rots.trans, batch["length"])

        return remove_padding(datastruct_from_text.joints, batch["length"])

    def forward_seq(self, texts: list[str], lengths: list[int], align_full_bodies=True, align_only_trans=False,
                    slerp_window_size=None, return_type="joints") -> List[Tensor]:

        assert not (align_full_bodies and align_only_trans)
        do_slerp = slerp_window_size is not None

        all_features = []
        for index, (text, length) in enumerate(zip(texts, lengths)):
            gpt3_texts = get_gpt3_texts([text])

            current_features = self.text_to_motion_forward(gpt3_texts, [length]).features[0]

            all_features.append(current_features)
        
        all_features = torch.cat(all_features)
        datastruct = self.Datastruct(features=all_features)
        motion = datastruct.rots
        rots, transl = motion.rots, motion.trans
        pose_rep = "matrix"
        from teach.tools.interpolation import aligining_bodies, slerp_poses, slerp_translation, align_trajectory  

        from teach.transforms.smpl import RotTransDatastruct
        final_datastruct = self.Datastruct(rots_=RotTransDatastruct(rots=rots, trans=transl))

        if return_type == "vertices":
            return final_datastruct.vertices
        elif return_type in ["joints", 'mmmns', 'mmm']:
            return final_datastruct.joints
        else:
            raise NotImplementedError

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int],
                               return_latent: bool = False,
                               return_feats: bool = False,
                               ):
        # Encode the text to the latent space
        if self.hparams.vae:
            distribution = self.textencoder(text_sentences)

            if self.sample_mean:
                latent_vector = distribution.loc
            else:
                # Reparameterization trick
                eps = distribution.rsample() - distribution.loc
                latent_vector = distribution.loc + self.fact * eps
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            if return_feats:
                return features
            else:
                return datastruct
        if return_feats:
            return features, latent_vector, distribution
        else:
            return datastruct, latent_vector, distribution

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms

        # Encode the motion to the latent space
        if self.hparams.vae:
            distribution = self.motionencoder(datastruct.features, lengths)

            if self.sample_mean:
                latent_vector = distribution.loc
            else:
                # Reparameterization trick
                eps = distribution.rsample() - distribution.loc
                latent_vector = distribution.loc + self.fact * eps
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion

        gpt3_texts = get_gpt3_texts(batch["text"])
        ret = self.text_to_motion_forward(gpt3_texts,
                                          batch["length"],
                                          return_latent=True)
        datastruct_from_text, latent_from_text, distribution_from_text = ret
        if self.motion_branch:

            # Encode the motion/decode to a motion
            ret = self.motion_to_motion_forward(batch["datastruct"],
                                                batch["length"],
                                                return_latent=True)
            datastruct_from_motion, latent_from_motion, distribution_from_motion = ret
        else:
            datastruct_from_motion = None
            latent_from_motion = None
            distribution_from_motion = None
        # GT data
        datastruct_ref = batch["datastruct"]

        # Compare to a Normal distribution
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        # Compute the losses
        loss = self.losses[split].update(ds_text=datastruct_from_text,
                                        ds_motion=datastruct_from_motion,
                                        ds_ref=datastruct_ref,
                                        lat_text=latent_from_text,
                                        lat_motion=latent_from_motion,
                                        dis_text=distribution_from_text,
                                        dis_motion=distribution_from_motion,
                                        dis_ref=distribution_ref)
        if split == "val":
            # Compute the metrics
            self.metrics.update(datastruct_from_text.detach().joints,
                                datastruct_ref.detach().joints,
                                batch["length"])

        return loss
