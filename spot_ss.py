import torch
import pickle
import numpy as np
from dataset.dataset_inference import ProteinDataset, text_collate_fn
from dataset.data_functions import pickle_load, read_list
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--save_path', default='results/', type=str, help='save path')
parser.add_argument('--device', default='cpu', type=str,
                    help='"cuda:0", or "cpu" note wont run on other gpu then gpu0 due to limitations of jit trace')

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path_list = read_list(args.file_list)
dataset = ProteinDataset(path_list)
data_loader = DataLoader(dataset, batch_size=1, collate_fn=text_collate_fn, num_workers=4)

means = pickle_load("data/stats/means.pkl")
stds = pickle_load("data/stats/stds.pkl")
means = torch.tensor(means, dtype=torch.float32)
stds = torch.tensor(stds, dtype=torch.float32)

if args.device == "cuda:0":

    model1 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/model1_class_gpu.pth").to(args.device)
    model2 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/model2_class_gpu.pth").to(args.device)
    # model3 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/model3_class_gpu.pth").to(args.device)
    model4 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/model4_class_gpu.pth").to(args.device)
    model5 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/model5_class_gpu.pth").to(args.device)
    model6 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/model6_class_gpu.pth").to(args.device)

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

for i, data in enumerate(tqdm(test_loader)):
    feats, labels, length, name, seq = data
    feats = (feats - means) / stds
    feats, labels = feats.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.float)
    length = torch.tensor(length)

    pred1 = model1(feats)
    pred2 = model2(feats, length)
    pred3 = model3(feats, length)
    pred4 = model4(feats, length)
    pred5 = model5(feats, length)
    pred6 = model6(feats)


    pred = (pred1.view(-1, 11) + pred2.view(-1, 11) + pred4.view(-1, 11) + pred5.view(-1, 11) + pred6.view(-1, 11)) / 5

    ss3_pred = pred[:, 0:3]
    ss8_pred = pred[:, 3:]
    ss3_label = labels[:, 0:3]
    ss8_label = labels[:, 3:]

    ss3_label_indices = ss3_label.argmax(1)
    ss3_label_indices[ss3_label[:, 0] == IGNORE_LABEL] = IGNORE_LABEL
    ss8_label_indices = ss8_label.argmax(1)
    ss8_label_indices[ss8_label[:, 0] == IGNORE_LABEL] = IGNORE_LABEL

