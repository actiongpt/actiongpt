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

from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from teach.transforms.joints2jfeats import Rifke
from teach.tools.geometry import matrix_of_angles
from teach.model.utils.tools import remove_padding


def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)


class ComputeMetricsBest(Metric):
    def __init__(self, jointstype: str = "mmm",
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if jointstype != "mmm":
            raise NotImplementedError("This jointstype is not implemented.")

        super().__init__()
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype,
                           normalization=False)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # APE
        self.add_state("APE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("APE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints"]

        # AVE
        self.add_state("AVE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("AVE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

    def compute(self):
        count = self.count
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}

        # Compute average of APEs
        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        count_seq = self.count_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        return {**APE_metrics, **AVE_metrics}

    def update(self, jts_text_: List[Tensor], jts_ref_: List[Tensor], lengths: List[List[int]]):
        self.count += sum(lengths[0])
        self.count_seq += len(lengths[0])

        ntrials = len(jts_text_)
        metrics = []
        for index in range(ntrials):
            jts_text, poses_text, root_text, traj_text = self.transform(jts_text_[index], lengths[index])
            jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref_[index], lengths[index])

            mets = []
            for i in range(len(lengths[index])):
                APE_root = l2_norm(root_text[i], root_ref[i], dim=1).sum()
                APE_pose = l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
                APE_traj = l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
                APE_joints = l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

                root_sigma_text = variance(root_text[i], lengths[index][i], dim=0)
                root_sigma_ref = variance(root_ref[i], lengths[index][i], dim=0)
                AVE_root = l2_norm(root_sigma_text, root_sigma_ref, dim=0)

                traj_sigma_text = variance(traj_text[i], lengths[index][i], dim=0)
                traj_sigma_ref = variance(traj_ref[i], lengths[index][i], dim=0)
                AVE_traj = l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

                poses_sigma_text = variance(poses_text[i], lengths[index][i], dim=0)
                poses_sigma_ref = variance(poses_ref[i], lengths[index][i], dim=0)
                AVE_pose = l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

                jts_sigma_text = variance(jts_text[i], lengths[index][i], dim=0)
                jts_sigma_ref = variance(jts_ref[i], lengths[index][i], dim=0)
                AVE_joints = l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

                met = [APE_root, APE_pose, APE_traj, APE_joints,
                       AVE_root, AVE_pose, AVE_traj, AVE_joints]
                mets.append(met)
            metrics.append(mets)

        # Quick hacks
        import numpy as np
        mmm = metrics[np.argmin([x[0][0] for x in metrics])]
        APE_root, APE_pose, APE_traj, APE_joints, AVE_root, AVE_pose, AVE_traj, AVE_joints = mmm[0]
        self.APE_root += APE_root
        self.APE_pose += APE_pose
        self.APE_traj += APE_traj
        self.APE_joints += APE_joints
        self.AVE_root += AVE_root
        self.AVE_pose += AVE_pose
        self.AVE_traj += AVE_traj
        self.AVE_joints += AVE_joints

    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]], rotations)
        poses = torch.stack((poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat((trajectory[..., :, [0]],
                          root_y[..., None],
                          trajectory[..., :, [1]]), dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]

        # return results in meters
        return (remove_padding(poses / 1000, lengths),
                remove_padding(poses_local / 1000, lengths),
                remove_padding(root / 1000, lengths),
                remove_padding(trajectory / 1000, lengths))
