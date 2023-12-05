import sys

import matplotlib
matplotlib.use('Agg')

import operator
import networkx as nx
import pandas as pd
import ast
import numpy as np
import tqdm , os
import random


import scipy.sparse as sp
import torch.nn as nn
from torchvision import models
import torch
from torchvision import transforms

import heapq
import collections
import random
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from nodevectors import Node2Vec

def intersection(lst1, lst2):
    n = max(len(lst1), len(lst2))
    return list(set(lst1) & set(lst2))

def create_graph(G, path="./",filename="train.csv", algo="", delete_nodes=[], corrupted=[], window=2):
    # Load data
    # G = nx.Graph()
    init_nodes=G.number_of_nodes()
    df = pd.read_csv(os.path.join(path, filename), delimiter=',')#,  encoding='Cp1252')
    num_paint = len(df)
    tokens = df["Token"]
    labels = df["Label"]
    tokens = [t.replace('"','').lower() for t in tokens]
   
    # Edges between paintings and their attributes
    for index in range(num_paint):
        neigbors = tokens[max(0, index-window):min(index+window, num_paint)]
        
        # Add edge type
        for t in neigbors:

            G.add_edge(tokens[index], intersection(t,tokens[index]))
        
        # G.add_edge(tokens[index], labels[index].replace(" ","-"))

    return tokens #[nodes.index(im) for im in paintings if im not in delete_nodes]
 

oov = 0
def load_data(semart, args, labels_only=False):

    dfs=[]
    for filename in ["train_2.csv","test.csv"]:
        textfile = os.path.join(semart, filename)
        dfs += [pd.read_csv(textfile, delimiter=',')]#, encoding='Cp1252')]
    df=pd.concat(dfs)
    #dftest = pd.read_csv(os.path.join(semart, "pred.csv"), delimiter=',')#, encoding='Cp1252')
    att = list(df["Label"])
    #att += list(dftest["Label"])
    # print(list(dftest[algo])[:10])

    tokens = list(df['Token']) #+ list(dftest['Token'])
    # features = get_resnet_features(imageurls)
    dic={tokens[i]:i for i in tqdm.tqdm(range(len(tokens)))}
    _delete_nodes = [tokens[i] for i in range(len(att)) if att[i]=="UNK"]
   
    #random_prediction(semart)
    # Create graph
    G = nx.Graph()
    train, val  = [create_graph(G,path=semart, filename="{}.csv".format(x), delete_nodes=_delete_nodes) for x in ["train_2"]]
    test = create_graph(G,filename="test.csv", path=semart, algo="", delete_nodes=_delete_nodes, corrupted=args.corrupted)
      
    features = []
    slabels = []
    
    # plot_deg(G.degree(), Gt.degree(), att, args);exit()

    nodes = list(G.nodes())
    print(nodes[:10])
    idx_train, idx_val, idx_test = [[nodes.index(im) for im in dataset if im not in _delete_nodes]for  dataset in [train, val, test]]
    for x in nodes:
        try:
            # imges got labels
            slabels += [att[dic[x]]] 
        except: 
            # non imges are labels, won't be used, fake labels to have the correct shape
            slabels +=  ["unk"] 

    # print(slabels)
    
    labels, _ = encode_onehot(slabels)
    print("labels : {}".format(labels.shape))
    labels = torch.LongTensor(np.where(labels)[1])
    if labels_only:
        return labels

    features = get_embedding_features(G, args)#get_embedding(args, imageurls, nodes)
    # print("features : {}".format(features.shape))
    features = sp.csr_matrix(np.array(features))
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # build symmetric adjacency matrix
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = adj + sp.eye(adj.shape[0])


    print("adj : {}".format(adj.shape))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print("train : {}, val:{}, test:{}".format(idx_train.shape, idx_val.shape, idx_test.shape))

    return G, att, slabels, adj, features, labels, idx_train, idx_val, idx_test

def get_embedding(args, imageurls, nodes):
    if args.init_embed=="random":
        return get_random_features(len(nodes),args.dim)
    elif args.init_embed=="embed":
        return get_embedding_features(args.graph_embs, nodes, dim=args.dim)
    else:
        print("init module unk")
        exit()

def get_random_features(n_nodes,dim=128):
    return  np.array([np.array([random.random() for i in range(dim)]) for n in range(n_nodes)])

def get_embedding_features(G, args):
    
    def get_vector(e):
        try: return graphEm[e]
        except: return np.array([random.random() for i in range(args.dim)])
    def get_vector2(e):
        try: return graphEm.predict(e)
        except: return np.array([random.random() for i in range(args.dim)])
    # graphEm = FastText.load_fasttext_format(modelname)
    # 
    # modelname = os.path.join(args.dir_dataset, "node2vec.model")
    modelname = os.path.join(args.dir_dataset, "node2vec_{}".format(args.att))
    try:
        # graphEm = KeyedVectors.load(modelname)
        graphEm =  Node2Vec.load(modelname)
        # print(graphEm)
        # graphEm =  {} #np.load("Models/node2vec_artist.npy", allow_pickle=True).item()
    except:
        graphEm =  Node2Vec(G, args.dim)
        graphEm.save(modelname)
    # return  np.array([get_vector2(n) for n in G.nodes()])
    if args.save_embedding:
        nodes = list(G.nodes())
        dic = {nodes[n]:get_vector2(n) for n in range(len(nodes))}
        np.save(modelname, dic)
        # exit()
    return  np.array([get_vector2(n) for n in range(len(list(G.nodes())))])



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    classes_dict = {c:i for i, c in enumerate(classes)}
    return labels_onehot, classes_dict
