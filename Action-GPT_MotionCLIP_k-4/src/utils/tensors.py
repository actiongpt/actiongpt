import torch
import numpy as np
import openai
import json
from gpt_annotator import GPT_Completion

def get_gpt3_texts(labels):
    with open('gpt3_annotations.json', 'r') as fp:
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
                results.append(label)
                # results.append("Describe a person's body movements who is performing the action "+str(label)+" in details")
            gpt3_texts.extend((results))
            # gpt3_texts.extend(GPT_Completion(results))
    return gpt3_texts    

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    if len(notnone_batches) == 0:
        out_batch = {"x": [], "y": [],
                     "mask": [], "lengths": [],
                     "clip_image": [], "clip_text": [],
                     "clip_path": [], "clip_images_emb": []
                     }
        return out_batch
    databatch = [b['inp'] for b in notnone_batches]
    labelbatch = [b['target'] for b in notnone_batches]
    lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)

    # actionlabelbatch = [b[2] for b in batch]
    # actionlabelbatchTensor = np.asarray(actionlabelbatch) #torch.as_tensor(actionlabelbatch)

    out_batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
             # "y_action_names": actionlabelbatchTensor}
    if 'clip_image' in notnone_batches[0]:
        clip_image_batch = [torch.as_tensor(b['clip_image']) for b in notnone_batches]
        out_batch.update({'clip_images': collate_tensors(clip_image_batch)})

    if 'clip_text' in notnone_batches[0]:
        textbatch = [b['clip_text'] for b in notnone_batches]
        gpt3_texts = get_gpt3_texts(textbatch)
        out_batch.update({'clip_text': gpt3_texts})

    if 'clip_path' in notnone_batches[0]:
        textbatch = [b['clip_path'] for b in notnone_batches]
        out_batch.update({'clip_path': textbatch})

    if 'clip_images_emb' in notnone_batches[0]:
        clip_images_emb = torch.as_tensor([b['clip_images_emb'] for b in notnone_batches]).squeeze().float()
        out_batch.update({'clip_images_emb': clip_images_emb})

    if 'all_categories' in notnone_batches[0]:
        textbatch = [b['all_categories'] for b in notnone_batches]
        out_batch.update({'all_categories': textbatch})

    return out_batch
