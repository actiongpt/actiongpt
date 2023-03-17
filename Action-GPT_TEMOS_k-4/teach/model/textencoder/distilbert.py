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

from typing import List, Union
import pytorch_lightning as pl

import torch.nn as nn
import os

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
import hydra

class DistilbertEncoderBase(pl.LightningModule):
    def __init__(self, modelpath: str,
                 finetune: bool = False) -> None:
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # workaround to work from cluster and local 
        rel_p = modelpath.split('/')
        rel_p = rel_p[rel_p.index('deps'):]
        rel_p = '/'.join(rel_p)
        modelpath = hydra.utils.get_original_cwd() + '/' + rel_p

        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.dim

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            # Don't put the model in
            if module == self.text_model and not self.hparams.finetune:
                continue
            module.train(mode)
        return self

    def get_last_hidden_state(self, texts: List[str],
                              return_mask: bool = False
                              ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)
