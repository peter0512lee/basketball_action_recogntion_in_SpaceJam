import numpy as np
import time
import os
import datetime
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

from model.gcn import RNN, Linear, GraphConvolution
from datasets.dataset_gcn import SkeletonDataset

args = EasyDict({

    'model_name': 'gcn_rnn_r2plus1d_bone',

    # training/model params
    'lr': 0.001,
    'start_epoch': 1,
    'num_epochs': 25,

    # Dataset params
    'in_features': 2,
    'num_classes': 10,
    'batch_size': 1,
    'n_total': 37085,
    'test_n': 7416,

    # Path params
    'annotation_path': "../datasets/annotation_dict.json",
    'model_path': "",

})


def inference(model_r2plus1d, model_gcn, model_rnn, model_linear, test_loader):

    num_correct = 0
    num_samples = 0

    model_r2plus1d.eval()
    model_gcn.eval()
    model_rnn.eval()
    model_linear.eval()

    pred_classes = []
    ground_truths = []

    with torch.no_grad():

        pbar = tqdm(test_loader)

        #  ----- Iterate over data -----1
        for sample in pbar:
            since = time.time()
            video = sample['video'].to(device)
            adjacency = sample['adjacency'].float().to(device)
            in_features = sample['node_features'].float().to(device)
            labels = sample['action'].to(device)

            # ----- Forward -----

            # concat
            '''
            for i in range(16):
                if(i == 0):
                    embeded_features = model_gcn(
                        adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn_tmp1 = model_rnn(embeded_features)
                elif(i == 1):
                    embeded_features = model_gcn(
                        adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn_tmp2 = model_rnn(embeded_features, hn_tmp1)
                    hn = torch.cat((torch.unsqueeze(hn_tmp1, 0),
                                    torch.unsqueeze(hn_tmp2, 0)), 0)
                else:
                    embeded_features = model_gcn(
                        adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn_tmp2 = model_rnn(embeded_features, hn_tmp2)
                    hn = torch.cat(
                        (hn, torch.unsqueeze(hn_tmp2, 0)), 0)
            outputs = model_linear(hn.transpose(
                0, 2).reshape(labels.size(0), -1))
            loss = criterion(outputs, torch.max(labels, 1)[1])
            '''

            # r2plus1d
            embeded_features_r2plus1d = model_r2plus1d(video)
            embeded_features_r2plus1d = embeded_features_r2plus1d.view(
                labels.size(0), -1)
            # '''
            # GCN + RNN
            for i in range(16):
                if(i == 0):
                    embeded_features = model_gcn(
                        adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn = model_rnn(embeded_features)
                else:
                    embeded_features = model_gcn(
                        adjacency, in_features[:, i])
                    embeded_features = embeded_features.reshape(
                        labels.size(0), 1, -1)
                    _, hn = model_rnn(embeded_features, hn)
            hn = hn.transpose(
                0, 1).reshape(labels.size(0), -1)
            embeded_features = torch.cat(
                (embeded_features_r2plus1d, hn), 1)
            outputs = model_linear(embeded_features)
            # '''

            _, preds = torch.max(outputs, 1)

            num_correct += (preds == labels.argmax(1)).sum()ㄅ
            num_samples += preds.size(0)

            pred_classes.extend(preds.detach().cpu().numpy())
            ground_truths.extend(
                torch.max(labels, 1)[1].detach().cpu().numpy())

            time_elapsed = time.time() - since
            print(time_elapsed)
            print('Inference one batch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        model_gcn.train()
        model_rnn.train()
        model_linear.train()
        model_r2plus1d.train()

        # ----- Save Confusion Matrix -----
        pred_classes = np.asarray(pred_classes)
        ground_truth = np.asarray(ground_truths)
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

        df_cm_acc = pd.DataFrame(
            cf_matrix / np.sum(cf_matrix) * 10, class_names, class_names)
        plt.figure(figsize=(9, 6))
        sns.heatmap(df_cm_acc, annot=True, fmt="f", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("labels")
        plt.savefig(args.model_path +
                    "/_test_confusion_matrix_acc.png")

    # Calculate elapsed time
    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':

    # Check Pytorch Environment
    print("Cuda Is Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Folder
    now = datetime.datetime.now()
    tinfo = "%d_%d_%d-%d-%d" % (now.year, now.month,
                                now.day, now.hour, now.minute)
    exp_path = "experiments/"
    model_name = args.model_name
    args.model_path = exp_path + tinfo + "_" + model_name + "/"

    print("Model Path: ", args.model_path)

    # if not os.path.exists(args.model_path):
    #     os.mkdir(args.model_path)

    # Tensorboard
    writer = SummaryWriter()

    # Load Model
    model_r2plus1d = models.video.r2plus1d_18(
        pretrained=True, progress=True)

    # Modify the last fc layer to output 10 classes
    model_r2plus1d.fc = nn.Linear(512, args.num_classes, bias=True)

    # Load pretrained weights
    pretrained_dict = torch.load(
        '../model_checkpoints/r2plus1d_augmented/r2plus1d_multiclass_16_0.0001.pt')['state_dict']
    model_r2plus1d.load_state_dict(pretrained_dict)

    # Load all layers except the last fc layer
    model_r2plus1d = torch.nn.Sequential(
        *(list(model_r2plus1d.children())[:-1]))

    pretrained_dict = torch.load(
        'experiments/2021_11_24-1-12_gcn_rnn_r2plus1d/gcn_rnn_r2plus1d_best.pt')

    model_gcn = GraphConvolution(args.in_features, 5).float()

    model_rnn = RNN()

    model_linear = Linear()

    model_gcn.load_state_dict(pretrained_dict['GCN'])
    model_rnn.load_state_dict(pretrained_dict['RNN'])
    model_linear.load_state_dict(pretrained_dict['Linear'])

    # Load Dataset
    skeleton_dataset = SkeletonDataset(
        annotation_dict=args.annotation_path)

    train_subset, test_subset = random_split(
        skeleton_dataset, [args.n_total-args.test_n, args.test_n], generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(dataset=train_subset,
                              shuffle=True, batch_size=args.batch_size, num_workers=4)

    test_loader = DataLoader(
        dataset=test_subset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    dataloaders_dict = {'train': train_loader, 'test': test_loader}

    # define optimizer
    params = list(model_gcn.parameters()) + list(model_rnn.parameters()
                                                 ) + list(model_linear.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    # define loss
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        # Put model into device after updating parameters
        model_r2plus1d = model_r2plus1d.to(device)
        model_gcn = model_gcn.to(device)
        model_rnn = model_rnn.to(device)
        model_linear = model_linear.to(device)
        criterion = criterion.to(device)

    inference(model_r2plus1d, model_gcn, model_rnn, model_linear, test_loader)

    writer.flush()
