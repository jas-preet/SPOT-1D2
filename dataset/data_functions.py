import pickle
import numpy as np
import pandas as pd

SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'


def read_list(file_name):
    """
    read a text file to get the list of elements
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
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


def one_hot(seq):
    """
    converts a sequence to one hot encoding
    :param seq: amino acid sequence (string)
    :return: one hot encoding of the amino acid (array)[L,20]
    """
    prot_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in prot_seq])
    return feat


def pickle_load(file_address):
    with open(file_address, "rb") as f:
        data = pickle.load(f)
    return data


def get_angle_degree(preds):
    preds = preds * 2 - 1
    preds_sin = preds[:, :, 0]
    preds_cos = preds[:, :, 1]
    preds_angle_rad = np.arctan2(preds_sin, preds_cos)
    preds_angle = np.degrees(preds_angle_rad)
    return preds_angle


def get_unnorm_asa(rel_asa, seq):
    """
    :param asa_pred: The predicted relative ASA
    :param seq_list: Sequence of the protein
    :return: absolute ASA_PRED
    """
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"

    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
               185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    seq = seq[0]
    # replacing the sequence residues with the max possible residue asa
    asa_max = np.array([dict_rnam1_ASA[i] for i in seq]).astype(np.float32)
    ### multiplying with the maxllist to unnormalise.
    abs_asa = np.multiply(rel_asa.cpu().detach().numpy(), asa_max)
    return abs_asa


def get_unnorm_asa_new(rel_asa, seq):
    """
    :param asa_pred: The predicted relative ASA
    :param seq_list: Sequence of the protein
    :return: absolute ASA_PRED
    """
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"

    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
               185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    # seq = seq[0]
    max_seq_len = len(seq[0])
    array_list = []
    for i, single_seq in enumerate(list(seq)):
        rel_asa_current = rel_asa[i, :]
        seq_len_diff = max_seq_len - len(single_seq)
        single_seq = single_seq + ("X" * seq_len_diff)

        asa_max = np.array([dict_rnam1_ASA[i] for i in single_seq]).astype(np.float32)
        abs_asa = np.multiply(rel_asa_current.cpu().detach().numpy(), asa_max)
        array_list.append(abs_asa)

    final_array = np.array(array_list)
    return final_array

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
