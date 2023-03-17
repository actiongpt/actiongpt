import os
import sys
import torch.nn.functional as F

sys.path.append('.')

import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.sampling import parser
from src.visualize.visualize import sampling, get_gpu_device
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa
import json
import numpy as np
from torch.utils.data import DataLoader
from src.utils.tensors import collate
from tqdm import tqdm
from copy import deepcopy
import pickle

plt.switch_backend('agg')

def main():
    # parse options
    parameters, folder, checkpointname, epoch, generations_folder = parser()
    os.makedirs(f"{generations_folder}/ground_truth",exist_ok=True)
    os.makedirs(f"{generations_folder}/action_gpt",exist_ok=True)

    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    model, datasets = get_model_and_data(parameters, split='test')

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    print(datasets["train"])
    print(datasets["test"])
    # sys.exit()
    dataset = datasets["test"]
    iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate)

    model.eval()
    grad_env = torch.no_grad

    m2m_loss_6d = []
    m2m_loss_3d = []
    t2m_loss_6d = []
    t2m_loss_3d = []
    count = 0
    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
    
            if len(batch['clip_text']) < 2:
                continue
            batch = {key: val.to(model.device) if torch.is_tensor(val) else val for key, val in batch.items()}

            generation = sampling(model, [batch['clip_text']], 0, parameters, folder=folder)

            batch = model(batch)

            x_6d = batch["x"]
            x_3d = batch["x_xyz"]
            
            m2m_6d = batch["output"]
            m2m_3d = batch["output_xyz"]

            t2m_6d = generation["output"][0]
            t2m_3d = generation["output_xyz"][0]
            mask = batch["mask"]

            for idx in range(len(x_3d)):
                gt_sample = x_3d[i].permute(2, 0, 1)
                gen_sample = t2m_3d[i].permute(2, 0, 1)
                np.save(f"{generations_folder}/ground_truth/"+str(count)+".npy",gt_sample.cpu())
                np.save(f"{generations_folder}/action_gpt/"+str(count)+".npy",gen_sample.cpu())
                count+=1

if __name__ == '__main__':
    main()
