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

from torch.utils.data import Dataset


class Text2Motion(Dataset):
    def __init__(self, texts, lengths):
        if not isinstance(lengths, list):
            raise NotImplementedError("Texts and lengths should be batched.")

        self.texts = texts
        self.lengths = lengths

        self.N = len(self.lengths)


    def __getitem__(self, index):
        return {"text": self.texts[index],
                "length": self.lengths[index]}

    def __len__(self):
        return self.N

    def __repr__(self):
        return f"Text2Motion dataset: {len(self)} data"
