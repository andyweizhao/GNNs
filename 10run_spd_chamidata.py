
import linalg as lalg
import torch
import os 
import random
import argparse
import time
import gc
from torch_geometric.nn.inits import glorot
from pytorchtools import EarlyStopping
from torch import nn
from classifiers import *
from spdgnn import SPDGCNConv, SPDGATConv, SPDChebConv, SPDSGConv, SPDGINConv
from data_utils import CustomDataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score

from transform_wei import SPDIsometry, get_logging
from spdgnn import expmap_id

import warnings
warnings.filterwarnings("ignore")

seed = random.randint(1, 1e3)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.set_default_tensor_type(torch.DoubleTensor)

class SquaredVec2SymMat(nn.Module):

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dimred = torch.nn.Linear(in_features=input_dims, out_features=output_dims * output_dims)        
        
    def forward(self, node_feats):

        node_feats = self.dimred(node_feats)                
        node_feats = node_feats.reshape(-1, self.output_dims, self.output_dims)
        node_feats = lalg.sym(node_feats)

        return node_feats

class TriangularVec2SymMat(nn.Module):

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        proj_dims = output_dims * (output_dims + 1) // 2
        
        self.dimred = torch.nn.Linear(in_features=input_dims, out_features=proj_dims) 
       
    def forward(self, node_feats):

        node_feats = self.dimred(node_feats)    
        triu_indices = torch.triu_indices(row=self.output_dims, col=self.output_dims)

        mat_feats = torch.zeros((len(node_feats), self.output_dims, self.output_dims),
                                device=node_feats.device, dtype=node_feats.dtype)
        
        mat_feats[:, triu_indices[0], triu_indices[1]] = node_feats
        
        mat_feats[:, triu_indices[1], triu_indices[0]] = node_feats        
        
        return mat_feats
    
class SPDGNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        if args.sym == 'squared':
            self.vec2sym = SquaredVec2SymMat(input_dims = args.num_node_features, output_dims = args.hidden_dims)        
        else:
            self.vec2sym = TriangularVec2SymMat(input_dims = args.num_node_features, output_dims = args.hidden_dims)   
        
        spdmodels = {
            "spdgcn": SPDGCNConv,
            "spdgat": SPDGATConv,
            "spdcheb": SPDChebConv,
            "spdsg": SPDSGConv,
            "spdgin": SPDGINConv,
        }
        self.layer_one = spdmodels[args.models](args.hidden_dims, args)
        self.layer_two = spdmodels[args.models](args.hidden_dims, args)

        self.dropout = nn.Dropout(args.dropout)
        
        classifs = {
            "spdvector": VectorNodeClassifier,
            "spdcentroid": SPDCentroidNodeClassifier,
            "spddia": SPDLogisticClassifier,
            "spddia_e": SPDLogisticClassifier_euclidean,
            "spdsvm_g": SVMClassifier,                   
        }
        
        self.classifier = classifs[args.classifier](args)
        
        if args.nonlinear=='reeig':
            self.nonlinear = self.ReEig
        else:
            self.nonlinear = self.TgReLU
    
    def TgReLU(self, mat_feats, delta = 0.1):
        
        e_x, v_x = torch.linalg.eigh(mat_feats, UPLO='U')
        
        dims = e_x.size(-1)
        
        e_x = torch.log(e_x)

        step_noise = delta * torch.arange(start=0, end=dims, step=1,
                                      dtype=e_x.dtype, device=e_x.device)
        
        step_noise = step_noise.expand_as(e_x)
        
        e_x = torch.where(e_x > 0, e_x, step_noise)
        
        e_x = torch.exp(e_x)
        
        out =  v_x @ torch.diag_embed(e_x) @ v_x.transpose(-1, -2)
        
        return out

    def ReEig(self, mat_feats, delta = 0.1):
        
        e_x, v_x = torch.linalg.eigh(mat_feats, UPLO='U')
        
        dims = e_x.size(-1)        

        e_x = torch.clamp(e_x, min=0.5)
        
        step_noise = delta * torch.arange(start=0, end=dims, step=1,
                                      dtype=e_x.dtype, device=e_x.device)
                
        e_x = e_x + step_noise
        
        out =  v_x @ torch.diag_embed(e_x) @ v_x.transpose(-1, -2)
        
        return out
            
    def forward(self, node_feats, edge_index):

        mat_feats = self.vec2sym(node_feats)      
        
        mat_feats = lalg.sym(self.dropout(mat_feats))
        
        mat_feats = expmap_id(mat_feats)        
        
        mat_feats_1, mat_feats_2 = self.apply_gnn_layers(mat_feats, edge_index)    
        
        pred_y = self.classifier(mat_feats_2)        
        
        return mat_feats_1, mat_feats_2, pred_y

    def apply_gnn_layers(self, mat_feats, edge_index):
                        
        mat_feats_1 = self.nonlinear(self.layer_one(mat_feats, edge_index))

        mat_feats_2 = self.nonlinear(self.layer_two(mat_feats_1, edge_index))
        
        return mat_feats_1, mat_feats_2     

def orthogonalize(W):
    beta = 0.001
    W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

def iterative_orthogonalization(model):
    attrs = vars(model)
    for att, module in attrs['_modules'].items():
        if isinstance(module, SPDIsometry):
            orthogonalize(getattr(module, 'isom_params').data)

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dims", default=3, type=int, help="the dimension of SPD.")
parser.add_argument("--dataset", default='disease_nc', type=str) 
parser.add_argument("--models", default='spdgcn', type=str)
parser.add_argument("--batchsize", default=-1, type=int) 
parser.add_argument("--patience", default=200, type=int) 
parser.add_argument("--classifier", default='spdvector', type=str)
parser.add_argument("--learningrate", default=0.01, type=float) 
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--cheb_incl_1st", default=True, type=bool)
parser.add_argument("--task", default='nc', type=str)

parser.add_argument("--nonlinear", default='reeig', type=str)
parser.add_argument("--spd_norm", default=None, type=str)

parser.add_argument("--sym", default='squared', type=str)

parser.add_argument("--transform", default="post-hoc", type=str, help="the choice of producing an orthogonal matrix",
                        choices=["post-hoc", "cayley", "qr", 'rot', 'ref']) 

parser.add_argument("--epoch", default=500, type=int)

parser.add_argument("--val-prop", default=0.05, type=float, help='proportion of validation edges for link prediction')
parser.add_argument("--test-prop", default=0.1, type=float, help='proportion of test edges for link prediction')
parser.add_argument("--use-feats", default=1, type=float, help='whether to use node features or not')

parser.add_argument("--normalize-feats", default=0, type=float, help='whether to normalize input node features')

parser.add_argument("--normalize-adj", default=1, type=float, help='whether to row-normalize the adjacency matrix')
parser.add_argument("--split-seed", default=1234, type=float, help='seed for data splits (train/test/val)')

parser.add_argument("--has_bias", default=1, type=int)

parser.add_argument("--c", default=0.0005, type=float)

args = parser.parse_args()

import json
print(f'json/{args.classifier}/{args.dataset}.json')

with open(f'json/{args.classifier}/{args.dataset}.json',) as f:
    hyper_parameters = json.load(f)[args.models]
    
    args.learningrate = hyper_parameters['learningrate']
    args.dropout = hyper_parameters['dropout']
    args.weight_decay = hyper_parameters['weight_decay']

    args.nonlinear = hyper_parameters['nonlinear']
    args.hidden_dims = hyper_parameters['hidden_dims']
    args.spd_norm = hyper_parameters['spd_norm']


args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CustomDataset(args, os.path.join('data', args.dataset))

data = dataset.to(args.device)

args.num_node_features = data.num_node_features
args.num_classes = data.y.max().item() + 1
if args.batchsize == -1:
    args.batchsize = data.x.shape[0]
    
if args.classifier in ['spdlogistic_covid_linear', 'spdlogistic_cov_linear', 'spdcentroid']:
    args.n_centroids = 200
else:    
    args.n_centroids = args.num_classes

print(args)
print(data.edge_index.shape)            

if len(data.train_mask.shape) > 1:  # When dataset contains cross-validation splits, we only use the first one.
    data.train_mask = data.train_mask[:, 0]
    data.test_mask = data.test_mask[:, 0]
    
#A = A + Id
data.edge_index, _ = add_remaining_self_loops(data.edge_index) 

tensor_dataset = TensorDataset(torch.nonzero(data.train_mask).squeeze())
train_loader = DataLoader(dataset=tensor_dataset, batch_size=args.batchsize, sampler=RandomSampler(tensor_dataset))


if args.classifier in ['spdsvm', 'spddia', 'spdsvm_g', 'spddia_e']:
    loss_function = nn.MultiMarginLoss()
else:    
    loss_function = torch.nn.CrossEntropyLoss()                 


log = get_logging()
torch.set_printoptions(profile="full")

model = SPDGNNModel(args).to(device)  

checkpoint_path = f'save/{args.models}-{args.classifier}-{args.dataset}-{args.learningrate}-{args.dropout}-{args.weight_decay}-{args.transform}-{args.has_norm}-{args.c}.pt'

early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=checkpoint_path)    
optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay, amsgrad=False)

for epoch in range(args.epoch):     
    
    model.train()
    tr_loss = 0 
    
    start = time.time()
    for step, node_idx in enumerate(train_loader):
        optimizer.zero_grad()
        
        if args.classifier == 'spdsvm_g':
            pred_y, g_invariant = model(data.x, data.edge_index)[2]
            loss = loss_function(pred_y[node_idx], data.y[node_idx]) + args.c * g_invariant.mean()
        else:
            pred_y = model(data.x, data.edge_index)[2]
            loss = loss_function(pred_y[node_idx], data.y[node_idx])
        
        loss.backward()
        
        optimizer.step()
        
        if args.transform == 'post-hoc':
            iterative_orthogonalization(model.layer_one)
            iterative_orthogonalization(model.layer_two)
            
        tr_loss += loss.item()
    xe_loss = tr_loss / len(train_loader)
    end = time.time()
    
    model.eval()
    with torch.no_grad():                        

        if args.classifier == 'spdsvm_g':
            pred_y, g_invariant = model(data.x, data.edge_index)[2]            
            loss_val = loss_function(pred_y[data.val_mask], data.y[data.val_mask]) + args.c * g_invariant.mean()
        else:
            pred_y = model(data.x, data.edge_index)[2]
            loss_val = loss_function(pred_y[data.val_mask], data.y[data.val_mask])
            
        acc_train = accuracy_score(pred_y[data.train_mask].argmax(dim=1).cpu(), 
                                data.y[data.train_mask].cpu())        
        acc_test = accuracy_score(pred_y[data.test_mask].argmax(dim=1).cpu(), 
                                data.y[data.test_mask].cpu())                     
        acc_val = accuracy_score(pred_y[data.val_mask].argmax(dim=1).cpu(), 
                                data.y[data.val_mask].cpu())   
    log.info(
            'running stats: {'
            f'"epoch": {epoch}, '
            f'"elapsed": {end - start:.2f}, '
            f'"acc_train": {acc_train*100.0:.2f}%, '
            f'"acc_val": {acc_val*100.0:.2f}%, '
            f'"acc_test": {acc_test*100.0:.2f}%, '
            f'"loss": {loss_val:.4f}, '
            '}'
        )         
    gc.collect()            
    torch.cuda.empty_cache()    

    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        break

model.load_state_dict(torch.load(checkpoint_path))

model.eval()
with torch.no_grad():   
    if args.classifier == 'spdsvm_g':
        pred_y, _ = model(data.x, data.edge_index)[2]
    else:
        node_feats1, node_feats2, pred_y = model(data.x, data.edge_index)
    test_acc = accuracy_score(pred_y[data.test_mask].argmax(dim=1).cpu(), data.y[data.test_mask].cpu())

log.info(f"Final Results: acc_test: {test_acc * 100:.2f}")

with open(f'results/{args.models}-{args.classifier}-{args.dataset}-{args.learningrate}-{args.dropout}-{args.weight_decay}-{args.transform}-{args.has_norm}-{args.c}', 'a') as f:
    f.write('-1,' + str(100 * test_acc) + '\n')
