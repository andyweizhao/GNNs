#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:56:28 2022

@author: zhaowi
"""

import torch
import random 
import argparse
import os
from torch import nn
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SGConv, GINConv

from torch_geometric.utils import add_remaining_self_loops
from torch.nn import Parameter
from data_utils import CustomDataset

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler 
from sklearn.metrics import accuracy_score
from pytorchtools import EarlyStopping

seed = random.randint(1, 1e3)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class LinearClassifier(nn.Module):
     def __init__(self, args):
         super().__init__()
     
         self.proj = nn.Linear(args.hidden_dims, args.num_classes)
         self.dropout = nn.Dropout(args.dropout)

     def forward(self, mat_feats):        
         mat_feats = self.dropout(mat_feats)
         preds = self.proj(mat_feats)
         return preds    

class SVMClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        self.W = Parameter(torch.Tensor(args.hidden_dims, args.num_classes))
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, mat_feats):        
        
        mat_feats = self.dropout(mat_feats)
        
        output = mat_feats @ self.W
        
        return output, self.W, mat_feats
    
    def b_trace(self, a):    
        return a.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
            
class CentroidClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.dims = args.hidden_dims
        self.n_centroids = args.n_centroids
        self.centroids = Parameter(torch.Tensor(self.n_centroids, self.dims))

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(self.n_centroids, args.num_classes)

        glorot(self.centroids)

    def forward(self, node_feats):
        
        node_feats = self.dropout(node_feats)
        
        distances = torch.norm(node_feats.unsqueeze(1) - self.centroids, p=2, dim=-1)
        
        return self.linear(distances) 

class LogisticClassifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.dims = args.hidden_dims
        self.n_centroids = args.n_centroids
        
        self.means = Parameter(torch.Tensor(self.n_centroids, self.dims))
                  
        self.sigma = torch.diag_embed(torch.ones(self.n_centroids, self.dims)).cuda()        
        
        self.bias = Parameter(torch.ones(self.n_centroids))                
        
        self.means.data.copy_(torch.ones(self.n_centroids, self.dims))
        
    def forward(self, node_feats):                    

        v = node_feats.unsqueeze(1) - self.means

        v = v.unsqueeze(-2)    
        
        distances = (v @ self.sigma @ v.transpose(-1, -2)).squeeze()
        
        distances = distances + self.bias
        
        return distances
    
        
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        if args.models == 'gcn':
            self.conv1 = GCNConv(args.num_node_features, args.hidden_dims)
            self.conv2 = GCNConv(args.hidden_dims, args.hidden_dims)
            
        if args.models == 'gat':        
            self.conv1 = GATConv(args.num_node_features, args.hidden_dims)
            self.conv2 = GATConv(args.hidden_dims, args.hidden_dims)
            
        if args.models == 'cheb':   
            self.conv1 = ChebConv(args.num_node_features, args.hidden_dims, K=2)
            self.conv2 = ChebConv(args.hidden_dims, args.hidden_dims, K=2)    

        if args.models == 'sgc':   
            self.conv1 = SGConv(args.num_node_features, args.hidden_dims, K=1)
            self.conv2 = SGConv(args.hidden_dims, args.hidden_dims, K=1) 
            
        if args.models == 'gin':  
            self.conv1 = GINConv(nn=torch.nn.Linear(args.num_node_features, args.hidden_dims), eps = 0, train_eps = False)
            self.conv2 = GINConv(nn=torch.nn.Linear(args.hidden_dims, args.hidden_dims), eps = 0, train_eps = False)
                
        
        classifs = {
            "linear": LinearClassifier,
            "centroid": CentroidClassifier,
            "logistic": LogisticClassifier,
            "svm": SVMClassifier,                    
        }
        
        self.classifier = classifs[args.classifier](args)
        
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        
        y = self.classifier(x2)
        return x1, x2, y
        
def SVM_loss(W, preds, Y, all_X, C=0.05):  
    
    Y = 2 * F.one_hot(Y, num_classes = args.num_classes) - 1
    distances = 1 - Y * preds 
    distances[distances < 0] = 0 
    
    hinge_loss = distances.mean()
    
    loss = C * torch.norm(W.transpose(0,1) , p=2, dim=-1).mean() + hinge_loss  
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dims", default=6, type=int, help="the dimension of SPD.")
    parser.add_argument("--dataset", default='airport', type=str) 
    parser.add_argument("--models", default='sgc', type=str)
    parser.add_argument("--classifier", default='logistic', type=str)
    parser.add_argument("--batchsize", default=-1, type=int) 
    parser.add_argument("--patience", default=100, type=int) 
    parser.add_argument("--learningrate", default=0.01, type=float) 
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--task", default='nc', type=str)
    parser.add_argument("--weight_decay", default=5e-3, type=float) 
    
    parser.add_argument("--epochs", default=200, type=int)
    
    parser.add_argument("--val-prop", default=0.05, type=float, help='proportion of validation edges for link prediction')
    parser.add_argument("--test-prop", default=0.1, type=float, help='proportion of test edges for link prediction')
    parser.add_argument("--use-feats", default=1, type=float, help='whether to use node features or not')
    
    parser.add_argument("--normalize-feats", default=0, type=float, help='whether to normalize input node features')
    
    parser.add_argument("--normalize-adj", default=1, type=float, help='whether to row-normalize the adjacency matrix')
    parser.add_argument("--split-seed", default=1234, type=float, help='seed for data splits (train/test/val)')
    
    args = parser.parse_args()
    
    import json
    with open(f'json/{args.dataset}.json',) as f:
        hyper_parameters = json.load(f)[args.models]
        
    args.learningrate = hyper_parameters['learningrate']
    args.dropout = hyper_parameters['dropout']
    args.weight_decay = hyper_parameters['weight_decay']
    args.hidden_dims = hyper_parameters['hidden_dims']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CustomDataset(args, os.path.join('data', args.dataset))
    
    data = dataset.to(device)
    
    args.device = device
    args.num_node_features = data.num_node_features
    args.num_classes = data.y.max().item() + 1
    if args.batchsize == -1:
        args.batchsize = data.x.shape[0]
        
    if args.classifier == 'centroid':
        args.n_centroids = 100
    else:
        args.n_centroids = args.num_classes
        
    print(args)
    
    if len(data.train_mask.shape) > 1:  # Use the first split if a dataset contains many splits.
        data.train_mask = data.train_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    
    # A = A + Id 
    data.edge_index, _ = add_remaining_self_loops(data.edge_index) 
    
    
    tensor_dataset = TensorDataset(torch.nonzero(data.train_mask).squeeze())
    train_loader = DataLoader(dataset=tensor_dataset, batch_size=args.batchsize, sampler=RandomSampler(tensor_dataset))
    
    loss_function = torch.nn.CrossEntropyLoss()     
    
    model = Net(args).to(device) 
    
    checkpoint_path =  'save/' + args.models + '-' + args.classifier + '-' + args.dataset + '.pt'
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=checkpoint_path)    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay, amsgrad=False)
    
    loss_function = torch.nn.CrossEntropyLoss() 
    
    for epoch in range(args.epochs):
        model.train()
        
        optimizer.zero_grad()
        
        if args.classifier == 'svm':
            _, _, (pred_y, W, all_X) = model(data)        
            loss = SVM_loss(W, pred_y[data.train_mask], data.y[data.train_mask], None)
        else:            
            pred_y = model(data)[2]
            loss = loss_function(pred_y[data.train_mask], data.y[data.train_mask])            
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():   
            
            if args.classifier == 'svm':
                _, _, (pred_y, _, _) = model(data)
            else:
                pred_y = model(data)[2]
    
            loss_val = loss_function(pred_y[data.val_mask], data.y[data.val_mask])
    
            acc_val = accuracy_score(pred_y[data.val_mask].argmax(dim=1).cpu(), data.y[data.val_mask].cpu())   
    
            print(f'Accuracy: {acc_val:.4f}, {loss_val:.4f}')
    
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            break
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    
    if args.classifier == 'svm':
        node_feats1, node_feats2, (pred_y, _, _) = model(data)
    else:
        node_feats1, node_feats2, pred_y = model(data)
    
    
    test_acc = accuracy_score(pred_y[data.test_mask].argmax(dim=1).cpu(), data.y[data.test_mask].cpu())
    
    print(f'Accuracy on test set: {test_acc:.4f}')
    
    with open('results/' + args.models + '-' + args.classifier + '-' + args.dataset, 'a') as f:
        f.write('-1,' + str(100 * test_acc) + '\n')
