import os, sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from dataset.data_functions import *


class ProteinDataset(Dataset):
    def __init__(self, path_fasta):
        self.path = path_fasta

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        fasta_path = self.path[idx]
        seq = read_fasta_file(fasta_path)
        protein = fasta_path.split('/')[-1].split('.')[0]

        pssm = read_pssm("data/pssm/" + protein + ".pssm", seq)
        hmm = read_hhm("data/hmm/" + protein + ".hhm", seq)
        pcp = read_pccp("data/aa_phy7.txt", seq)
        protein_len = len(seq)
        features = np.concatenate((pssm, hmm, pcp), axis=1)

        return features, protein_len, protein, seq


def text_collate_fn(data):
    """
    collate function for data read from text file
    """

    # sort data by caption length
    data.sort(key=lambda x: x[1], reverse=True)

    features, protein_len, protein, seq = zip(*data)

    features = [torch.FloatTensor(x) for x in features]

    # Pad features
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    return padded_features, protein_len, protein, seq  ### also return feats_lengths and label_lengths if using packpadd
