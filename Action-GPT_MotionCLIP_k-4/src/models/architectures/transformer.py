import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        
        self.input_feats = self.njoints*self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        if self.ablation == "average_encoder":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Linear(self.latent_dim, self.latent_dim)
        # elif self.ablation == "extra_token":
        #     self.extra_token = nn.Parameter(torch.randn(1, self.latent_dim))
        else:
            self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
            # self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        # print("x.shape")
        # print(x.shape)

        # embedding of the skeleton
        x = self.skelEmbedding(x)
        # print(x.shape)

        # adding the mu and sigma queries
        xseq = torch.cat((self.muQuery[y][None], x), axis=0)
        # print("xseq.shape")
        # print(xseq.shape)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 1), dtype=bool, device=x.device)

        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        # print(xseq.shape)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        # print("final.shape")
        # print(final.shape)
        mu = final[0]

        if self.normalize_output:
            mu = mu / mu.norm(dim=-1, keepdim=True)

        return {"mu": mu}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats
        self.normalize_decoder_input = kargs.get('normalize_decoder_input', False)

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch, use_text_emb=False):
        z, mask, lengths = batch["z"], batch["mask"], batch["lengths"]
        if use_text_emb:
            z = batch["clip_text_emb"]
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                # print("z.shape")
                # print(z.shape)
                # print(njoints,nfeats)
                # shift the latent noise vector to be the action noise
                # z = z + self.actionBiases[y] # TODO - REMOVED HERE BIAS IN ENCODER
                z = z[None]
                # print(z.shape)
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        # print("timequeries.shape")
        # print(timequeries.shape)
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)

        # print("euuuu")
        # print(timequeries.shape)
        # print(z.shape)
        if self.normalize_decoder_input:
            z = z / torch.norm(z, dim=-1, keepdim=True)

        # print("z.shape---")
        # print(z.shape)
        # z = z
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        # print("output.shape")
        # print(output.shape)
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        if use_text_emb:
            batch["txt_output"] = output
        else:
            batch["output"] = output
        return batch
