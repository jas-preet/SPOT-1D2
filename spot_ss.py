import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from main import classification, write_csv
from dataset.data_functions import pickle_load, read_list
from dataset.dataset_inference import ProteinDataset, text_collate_fn


parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--save_path', default='results/', type=str, help='save path')
parser.add_argument('--device', default='cuda:0', type=str,
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


model1 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/new_jits/model1_class_gpu.pth").to(args.device)
model2 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/new_jits/model2_class_gpu.pth").to(args.device)
model3 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/new_jits/model4_class_gpu.pth").to(args.device)
model4 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/new_jits/model5_class_gpu.pth").to(args.device)
model5 = torch.jit.load("/home/jaspreet/jaspreet_data/jits_conference/new_jits/model6_class_gpu.pth").to(args.device)



class_out = classification(data_loader, model1, model2, model3, model4, model5, means, stds, args.device)
names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out

print(len(ss3_pred_list))
write_csv(class_out, args.save_path)