from networkx.readwrite.json_graph import adjacency
import numpy as np
import time
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from easydict import EasyDict
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import dgl
from model.gcn import RNN, Linear, GraphConvolution
from datasets.dataset_gcn import SkeletonDataset

args = EasyDict({

    'model_name': 'gcn_rnn',

    # training/model params
    'lr': 0.001,
    'start_epoch': 1,
    'num_epochs': 200,

    # Dataset params
    'in_features': 2,
    'num_classes': 10,
    'batch_size': 64,
    'n_total': 37085,
    'test_n': 7416,

    # Path params
    'annotation_path': "../datasets/annotation_dict.json",
    'model_path': "./experiments/2021_11_17-14-25_gcn_rnn",

})


def test(model_gcn, model_rnn, model_linear, args):
    # Load Dataset
    skeleton_dataset = SkeletonDataset(
        annotation_dict=args.annotation_path)
    train_subset, test_subset = random_split(
        skeleton_dataset, [args.n_total-args.test_n, args.test_n], generator=torch.Generator().manual_seed(1))
    test_loader = DataLoader(
        dataset=test_subset, shuffle=False, batch_size=args.batch_size)

    # Evaluate
    num_correct = 0
    num_samples = 0
    model_gcn.eval()
    model_rnn.eval()
    model_linear.eval()
    pred_classes = []
    ground_truth = []

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for sample in pbar:
            adjacency = sample['adjacency'].float().to(device)
            in_features = sample['node_features'].float().to(device)
            labels = sample['action'].to(device)

            for i in range(16):
                if(i == 0):
                    embeded_features = model_gcn(adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn = model_rnn(embeded_features)
                else:
                    embeded_features = model_gcn(adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn = model_rnn(embeded_features, hn)
            outputs = model_linear(hn.transpose(
                0, 1).reshape(labels.size(0), -1))
            _, preds = torch.max(outputs, 1)
            labels_ = labels.argmax(1)

            num_correct += (preds == labels_).sum()
            num_samples += preds.size(0)

            pred_classes.extend(preds.detach().cpu().numpy())
            ground_truth.extend(torch.max(labels, 1)[1].detach().cpu().numpy())

        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    pred_classes = np.asarray(pred_classes)
    ground_truth = np.asarray(ground_truth)
    cf_matrix = confusion_matrix(ground_truth, pred_classes)

    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(per_cls_acc)
    print("Plot confusion matrix")

    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)
    plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("labels")
    plt.savefig(args.model_path +
                "/_test_confusion_matrix.png")


if __name__ == '__main__':

    print("Cuda Is Available: ", torch.cuda.is_available())
    print("Current Device: ", torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_gcn = GraphConvolution(args.in_features, 5).float()
    model_rnn = RNN()
    model_linear = Linear()

    checkpoints = torch.load(
        '{}/{}_best.pt'.format(args.model_path, args.model_name))
    model_gcn.load_state_dict(checkpoints['GCN'])
    model_rnn.load_state_dict(checkpoints['RNN'])
    model_linear.load_state_dict(checkpoints['Linear'])

    if torch.cuda.is_available():
        model_gcn = model_gcn.to(device)
        model_rnn = model_rnn.to(device)
        model_linear = model_linear.to(device)

    test(model_gcn, model_rnn, model_linear, args)
