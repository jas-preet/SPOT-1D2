import os, sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from dataset.data_functions import *


class Proteins_Dataset(Dataset):
    def __init__(self, list):

        self.list = list
        self.protein_list = read_list(self.list)

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        protein = self.protein_list[idx]
        seq = read_fasta_file("/data/fasta/", protein + ".fasta")
        pssm = read_pssm("data/pssm/", protein + ".pssm", seq)
        hmm = read_hhm("data/hmm/", protein + ".hhm", seq)
        pcp = read_pccp( "data/aa_phy7.txt", seq)
        protein_len = len(seq)
        features = np.concatenate((pssm, hmm, pcp), axis=1)

        return features, protein_len, protein, seq


def text_collate_fn(data):
    """
    collate function for data read from text file
    """

    # sort data by caption length
    data.sort(key=lambda x: x[2], reverse=True)
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    features, protein_len, protein, seq = zip(*data)

    features = [torch.FloatTensor(x) for x in features]

    # Pad features
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    return padded_features, protein_len, protein, seq  ### also return feats_lengths and label_lengths if using packpadd
