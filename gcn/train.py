import time
import os
import argparse
import numpy as np
import tqdm
import sys 

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=4e-6,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Initial epoch.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--author_weight', type=float, default=0.25, help='(weight for author).')
parser.add_argument('--patience', default=50, type=int, help=' Time to wait without improvement.')
parser.add_argument('--max_degree', default=-1, type=int, help=' the max degree possible for a node.')
parser.add_argument('--min_degree', default=-1, type=int, help=' the max degree possible for a node.')
parser.add_argument('--minc', default=5, type=int, help='minimum examples for the labels.')
parser.add_argument('--dim', default=128, type=int, help='dimension of embedding space (features)')
parser.add_argument('--init_embed', default="embed", type=str, help='type of embedding (random|resnet|embed). Embed is for both node2vec and transe')
parser.add_argument('--graph_embs', default="../SemArt/Data/semart-artgraph-node2vec.model", type=str)
parser.add_argument('--verbose', action='store_true', default=True, help='verbose.')
parser.add_argument('--multitask', action='store_true', default=False, help='multitask learning.')
parser.add_argument('--focal', action='store_true', default=False, help='use focal loss.')
parser.add_argument('--att', default='time', type=str, help='Attribute classifier (type | school | time | author) .')
parser.add_argument('--corrupted', default='', type=str, help='Corrupted Attribute classifier (type | school | time | author) .')
parser.add_argument('--device_id', default="0", type=str, help='device_id to use .')
parser.add_argument('--output', default="", type=str, help='output to save the gcn embedding.')
parser.add_argument('--reducer', default="tsne", type=str, help='dimension reduction algo.')
parser.add_argument('--dataset', default="", type=str, help='the used dataset.')
parser.add_argument('--model', default="GCN", type=str, help='the used model (GCN|GAT).')
parser.add_argument('--atts', default='timesizetypedimension', type=str, help='Attribute classifier (type | school | time | author) .')
parser.add_argument('--csvtrain', default="ukiyoe_train.csv", type=str, help='csv for train')
parser.add_argument('--csvtest', default="ukiyoe_test.csv", type=str, help='csv for test.')
parser.add_argument('--csvval', default="ukiyoe_val.csv", type=str, help='csv for val.')
parser.add_argument('--dir_dataset', default="", type=str, help='dir of the dataset .')
parser.add_argument('--resume', default="", type=str, help='dir of the dataset .')
parser.add_argument('--save_embedding', action='store_true', default=False, help='verbose.')



args = parser.parse_args()
args.cuda = torch.cuda.is_available()


from utils import load_data, accuracy, encode_onehot
dataset="./"
args.dir_dataset = "./"


from models import GCN, FocalLoss, GAT

# from dgl import DGLGraph

# if args.att=="author":
#     args.patience = 1000
#     args.epochs = 10000

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    if args.verbose:
        print("cuda on")
    # torch.cuda.device(args.device_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    torch.cuda.manual_seed(args.seed)
else:
    if args.verbose:
        print("cuda off")

# create_nn()
# plot_embedding(dataset, "nn-embedding-{}.npy".format(args.att.replace(" ","_")), args, algo ="nn")
best_val = 0
# Load data

G, valid_labels, slabels, adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset, args)
# Model and optimizer

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
focalloss = FocalLoss(gamma=2)#, alpha=0.75)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()




def train_epoch(epoch):
    t = time.time()
    # adj[idx_train] = adj[idx_train]/len(adj)
    model.train()
    optimizer.zero_grad()

    if args.model == "GCN":
        output = model(features, adj)
        # print("shape {}".format(output.shape))
    else:
        output = model(features)

    if args.focal:
        loss_train = focalloss(output[idx_train], labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode and args.verbose and epoch%10==0:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        if args.model == "GCN":
            output = model(features, adj)
        else:
            output = model(features)
        
        if args.focal:
            loss_test = focalloss(output[idx_test], labels[idx_test])
        else:
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])

        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("{} Test set results:".format(args.att),
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))


    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    if args.focal:
        loss_val = focalloss(output[idx_val], labels[idx_val])
    else:
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if args.verbose:
        if epoch%10==0:
            print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
            sys.stdout.flush()
    return acc_val


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if args.verbose:
        print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    else:
        print("{},{:.4f}".format(args.att,acc_test.item()))



def train():
   
    t_total = time.time()
    pat_track=0
    global best_val
    # best_val = float(0)
    bar = tqdm.tqdm(total = args.epochs)
    
    for epoch in range(args.start_epoch, args.epochs):
        if not args.verbose:
            bar.update(1)
        acc_val = train_epoch(epoch)
        if acc_val <= best_val:
            pat_track += 1
        else:
            pat_track = 0
            
        if acc_val > 0.99:# or pat_track >= args.patience : #
            break
        best_val = max(acc_val, best_val)
    if args.verbose:
        print()
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    output = model(features, adj)
    transform_prediction(list(G.nodes()), slabels, labels, idx_test, output)
    # plot_embedding(dataset, "gcn-embedding-{}.npy".format(args.att.replace(" ","_")), args, algo ="gcn")
    # plot_embedding(dataset, "gcn-embedding-{}.npy".format(args.att.replace(" ","_")), args.reducer, args.att, algo ="gcn")

def transform_prediction(nodes, slabels, olabels, idx_test, embedding):
        
    preds = embedding[idx_test].max(1)[1].type_as(olabels[idx_test])
    print(preds[0])
    elabels, dicl = encode_onehot(slabels)
    rdicl = {dicl[key]:key for key in dicl}
    preds = [rdicl[p].replace("-"," ") for p in preds.cpu().detach().numpy()]
    print(preds[:10])
    df = pd.read_csv("pred.csv", sep=",")
    ids =  list(df["Id"])
    
    
    idx_test = idx_test.cpu().detach().numpy()
    test_tokens = [nodes[idx]for idx in idx_test]
    data = {"Id":ids,"Token":test_tokens, "Label":preds }
    dft = pd.DataFrame.from_dict(data)
    dft.to_csv("submission_ciad_{}.csv".format("gcn"), sep=",", index=False)
    # verifyOutput()

def verifyOutput():

    df = pd.read_csv("pred.csv", sep=",")
    ids1 =  list(df["Id"])
    tokens1 =  list(df["Token"])
    df = pd.read_csv("test_submission_ciad_gcn.csv", sep=",")
    ids2 =  list(df["Id"])
    tokens2 =  list(df["Token"])
    missed=0
    for i in range(len(ids1)):
        if ids1[i]==ids2[i] and tokens1[i]!=tokens2[i]:
            missed += 1
    print("missed: {}".format(missed))

    
# Train model
if __name__ == "__main__":
    train()
    
    # plot_embedding(dataset, "gcn-embedding-{}.npy".format(args.att.replace(" ","_")), args.reducer, args.att, algo ="gcn")
# Testing
# test()

