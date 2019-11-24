#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:14:03 2019

@author: sreekuk1
"""
#Libraries required
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import pca
import scipy.sparse.linalg as la
from scipy.sparse import csr_matrix 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from collections import Counter

#reading graph file into csv file- contains only edges
f=open("ca-AstroPh.txt") 
graph=np.array(pd.read_csv(f,sep=' ',header=None,skiprows=1))

#reading the number of vertices,edges and partitions required
with open('ca-AstroPh.txt') as f: 
    first_line = f.readline()
graph_key=[int(s) for s in first_line.split() if s.isdigit()]

n_vert=graph_key[0]
n_edge=graph_key[1]
n_partition=graph_key[2]

#Constructing graph
G=nx.Graph()
G.add_edges_from(graph)

#Computing laplacian 
L=nx.normalized_laplacian_matrix(G)
L=csr_matrix.astype(L,dtype='f')

# Computing eigen vectors
vec=[]
eig_val, eig_vec=la.eigsh(L,6) 

# Computing k-means
#kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
km=KMeans(n_clusters=n_partition,max_iter=5000)
km.fit(eig_vec)


#objective function
cluster=[]
cluster.append(km.labels_)
cluster[0]=list(cluster[0])
denom=Counter(km.labels_).values()
phi=0
a=[0]*n_partition
phi=0
for i in range(n_partition):
    for node in range(n_vert):
        if(cluster[0][node]==i):
            for n in G.neighbors(node):
                if(cluster[0][n]!=i):
                    a[i]+=1
phi=sum(np.array(a)/np.array(denom))
