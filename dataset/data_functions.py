import torch
import random
import pickle
import numpy as np
import pandas as pd
import csv

MISSING_LABEL = -99



def pickle_load(file_address):
    with open(file_address, "rb") as f:
        data = pickle.load(f)
    return data

def read_list(file_name):
    '''
    returns list of proteins from file
    '''
    with open(file_name, 'r') as f:
        text = f.read().splitlines()
    return text


def read_fasta_file(fname):
    """
    reads the sequence from the fasta file
    :param fname: filename (string)
    :return: protein sequence  (string)
    """
    with open(fname, 'r') as f:
        AA = ''.join(f.read().splitlines()[1:])
    return AA

def read_pssm(fname, seq):
    """
    read the pssm
    :params
        fname: filename (string)
        seq: protein sequence  (string)
    :return
        pssm (array) [L,20] (Float64)
    """
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname, 'r') as f:
        tmp_pssm = pd.read_csv(f, delim_whitespace=True, names=pssm_col_names).loc[:, '2':'21'].dropna().values[2:,
                   :].astype(float)  # 'inf' value sometimes nan...?
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm


def read_hhm(fname, seq):
    """
    read the hmm file
    :params
        fname: filename (string)
        seq: protein sequence  (string)
    :return
        hmm (array) [L,30] (Float64)
    """
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname, 'r') as f:
        hhm = pd.read_csv(f, delim_whitespace=True, names=hhm_col_names)

    pos1 = (hhm['0'] == 'HMM').idxmax() + 3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:, :num_hhm_cols].reshape([-1, 44])
    hhm[hhm == '*'] = '9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:, 2:-12].astype(float)


def phy7(fname):
    with open(fname, 'r') as f:
        pccp = f.read().splitlines()
        pccp = [i.split() for i in pccp]
        pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}
    return pccp_dic


def read_pccp(fname, seq):
    """
    reading "aa_phy7.txt" and calculating the values for the sequence
    :params
        fname: filename (string)
        seq: protein sequence (string)
    :return
        pcp: (array) [L,7] (Float64)
    """
    pccp = phy7(fname)
    return np.array([pccp[i] for i in seq])


def seed_random(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)