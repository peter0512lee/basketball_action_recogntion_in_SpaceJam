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

    'model_name': 'gcn_rnn_r2plus1d',

    # training/model params
    'lr': 0.001,
    'start_epoch': 1,
    'num_epochs': 25,

    # Dataset params
    'in_features': 2,
    'num_classes': 10,
    'batch_size': 8,
    'n_total': 37085,
    'test_n': 7416,

    # Path params
    'annotation_path': "../datasets/annotation_dict.json",
    'model_path': "",

})


def train(model_r2plus1d, model_gcn, model_rnn, model_linear, dataloaders, criterion, optimizer, args, num_epochs):

    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:

            print('Phase: {}'.format(phase))

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                model_r2plus1d.train()
                model_gcn.train()
                model_rnn.train()
                model_linear.train()
            else:
                model_r2plus1d.eval()
                model_gcn.eval()
                model_rnn.eval()
                model_linear.eval()
                test_pred_classes = []
                test_ground_truths = []

            pbar = tqdm(dataloaders[phase])

            #  ----- Iterate over data -----
            for sample in pbar:
                video = sample['video'].to(device)
                adjacency = sample['adjacency'].float().to(device)
                in_features = sample['node_features'].float().to(device)
                labels = sample['action'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # ----- Forward -----
                with torch.set_grad_enabled(phase == 'train'):

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
                    loss = criterion(outputs, torch.max(labels, 1)[1])
                    # '''

                    _, preds = torch.max(outputs, 1)

                    if phase == 'test':
                        test_pred_classes.extend(preds.detach().cpu().numpy())
                        test_ground_truths.extend(
                            torch.max(labels, 1)[1].detach().cpu().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * in_features.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            # ----- Write History to Tensorboard -----
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Acc/" + phase, epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # ----- Save Model -----
            stat = {
                'GCN': model_gcn.state_dict(),
                'RNN': model_rnn.state_dict(),
                'Linear': model_linear.state_dict(),
            }
            if phase == 'train':
                print("Saving model ...")
                torch.save(stat, '{}/{}_{}.pt'.format(args.model_path,
                                                      args.model_name, epoch + 1))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print("Saving best model ...")
                torch.save(stat, '{}/{}_best.pt'.format(
                    args.model_path, args.model_name))

                # ----- Save Confusion Matrix -----
                pred_classes = np.asarray(test_pred_classes)
                ground_truth = np.asarray(test_ground_truths)
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
                            "/{}_test_confusion_matrix.png".format(epoch + 1))

            print('Best test Acc: {:4f}'.format(best_acc))
            print('-' * 10)

    # Calculate elapsed time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))


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

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Tensorboard
    writer = SummaryWriter()

    # Load Model
    model_r2plus1d = models.video.r2plus1d_18(
        pretrained=True, progress=True)

    model_r2plus1d = torch.nn.Sequential(
        *(list(model_r2plus1d.children())[:-1]))

    # let pretrained net all parameters not to learn
    for param in model_r2plus1d.parameters():
        param.requires_grad = False

    for name, param in model_r2plus1d.named_parameters():
        for layer in ['layer3', 'layer4', 'fc']:
            if layer in name:
                param.requires_grad = True

    params_to_update = model_r2plus1d.parameters()
    # print("Params to learn:")
    params_to_update = []
    for name, param in model_r2plus1d.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # print("\t", name)

    model_gcn = GraphConvolution(args.in_features, 5).float()

    model_rnn = RNN()

    model_linear = Linear()

    # Load Dataset
    skeleton_dataset = SkeletonDataset(
        annotation_dict=args.annotation_path)

    train_subset, test_subset = random_split(
        skeleton_dataset, [args.n_total-args.test_n, args.test_n], generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(dataset=train_subset,
                              shuffle=True, batch_size=args.batch_size)

    test_loader = DataLoader(
        dataset=test_subset, shuffle=False, batch_size=args.batch_size)

    dataloaders_dict = {'train': train_loader, 'test': test_loader}

    # define optimizer
    params = list(params_to_update) + list(model_gcn.parameters()) + list(model_rnn.parameters()
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

    train(model_r2plus1d, model_gcn, model_rnn, model_linear, dataloaders_dict, criterion,
          optimizer, args, num_epochs=args.num_epochs)

    writer.flush()
