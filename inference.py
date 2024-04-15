import argparse

import torch
from torch.utils.data import DataLoader

from Model.alphabets import SecStr8, Str3
from Model.DSSPDataset import DSSPDataset
from utils import T5_collate_fn


def preprocessing(seqs, str_type='dssp8'):
    if str_type == 'dssp8':
        dssp_encoder = SecStr8
    elif str_type == 'dssp3':
        dssp_encoder = Str3
    else:
        raise ValueError('Invalid str_type')

    labels, lengths = [], []
    for seq in seqs:
        l_ = len(seq)
        lengths.append(l_)

        label = "".join(["C"*l_])
        label = torch.from_numpy(dssp_encoder.encode(label.encode())).long()
        labels.append(label)

    return seqs, labels, lengths, dssp_encoder


def infer(seqs, str_type='dssp3', model="MASKSecondary_Proteinnet_dssp3_T5_emb_dropout_0.3_16_RPE_seg_feature_block3.jit.pt"):
    checkpoint = 'Checkpoints/' + model
    model = torch.jit.load(checkpoint)
    model = model.cuda()

    if str_type == 'dssp8':
        dssp_dim = 8
    elif str_type == 'dssp3':
        dssp_dim = 3
    seqs, labels, lengths, dssp_encoder = preprocessing(seqs, str_type=str_type)
    evaluation_data = DSSPDataset(seqs, labels, lengths)
    collate_fn = T5_collate_fn(dssp_dim)
    dataloader = DataLoader(evaluation_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    out = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_index, lengths, rep_pack, batch_labels, mask = batch
            pred = model(rep_pack.cuda(), mask)
            pred = pred.squeeze(0)
            _, p = pred.max(dim=-1)
            out.append(dssp_encoder.decode(p.cpu()).decode())

    return out
