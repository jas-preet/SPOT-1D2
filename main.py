import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from dataset.data_functions import SS3_CLASSES, SS8_CLASSES, get_angle_degree, get_unnorm_asa, get_unnorm_asa_new


def classification(data_loader, model1, model2, model3, model4, model5, mean, std, device):
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    ss3_pred_list = []
    ss8_pred_list = []
    ss3_prob_list = []
    ss8_prob_list = []
    names_list = []
    seq_list = []
    for i, data in enumerate(data_loader):
        feats, length, name, seq = data
        length = torch.tensor(length).to(device)
        feats = (feats - mean) / std
        feats = feats.to(device, dtype=torch.float)

        # print(feats.shape)
        pred1 = model1(feats)
        pred2 = model2(feats, length)
        pred3 = model3(feats, length)
        pred4 = model4(feats, length)
        pred5 = model5(feats)[None, :, :]
        pred = (pred1 + pred2 + pred3 + pred4 + pred5) / 5

        ss3_pred = F.softmax(pred[:, :, 0:3], dim=2)
        ss8_pred = F.softmax(pred[:, :, 3:], dim=2)
        name = list(name)
        for i, prot_len in enumerate(list(length)):
            prot_len_int = int(prot_len)
            ss3_pred_single = ss3_pred[i, :prot_len_int, :]
            ss3_pred_single = torch.squeeze(ss3_pred_single, dim=0).cpu().detach().numpy()
            ss3_indices = np.argmax(ss3_pred_single, axis=1)
            ss3_pred_aa = np.array([SS3_CLASSES[aa] for aa in ss3_indices])[:, None]
            ss3_pred_list.append(ss3_pred_aa)
            ss3_prob_list.append(ss3_pred_single)

            ss8_pred_single = ss8_pred[i, :prot_len_int, :]
            ss8_pred_single = torch.squeeze(ss8_pred_single, dim=0).cpu().detach().numpy()
            ss8_indices = np.argmax(ss8_pred_single, axis=1)
            ss8_pred_aa = np.array([SS8_CLASSES[aa] for aa in ss8_indices])[:, None]
            ss8_pred_list.append(ss8_pred_aa)
            ss8_prob_list.append(ss8_pred_single)
            names_list.append(name[i])

        for seq in list(seq):
            seq_list.append(np.array([i for i in seq])[:, None])

    return names_list, seq_list, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list


def write_csv(class_out, save_dir):
    names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out

    for seq, ss3, ss8, ss3_prob, ss8_prob, name in zip(seq, ss3_pred_list,
                                                       ss8_pred_list,
                                                       ss3_prob_list,
                                                       ss8_prob_list, names):
        # print(seq.shape, ss3.shape, ss8.shape, ss3_prob.shape, ss8_prob.shape)
        data = np.concatenate((seq, ss3, ss8, ss3_prob, ss8_prob), axis=1)

        save_path = os.path.join(save_dir, name + ".csv")
        pd.DataFrame(data).to_csv(save_path,
                                  header=["AA", "SS3", "SS8", "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G",
                                          "P8I", "P8E", "P8B"])
    return print(f'please find the results saved at {save_dir} with .csv extention')


if __name__ == '__main__':
    print("Please run the spot_ss.py instead")
