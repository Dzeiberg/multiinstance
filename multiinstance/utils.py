# AUTOGENERATED! DO NOT EDIT! File to edit: 03_Dataset_Utils.ipynb (unless otherwise specified).

__all__ = ['estimator', 'getBootstrapSample', 'getEsts', 'getBagAlphaHats', 'getCliqueAlphaHats', 'getAlphaPrime',
           'getKSMatrixPMatrix', 'getAllCliques', 'clusterByLeidenAlg', 'getOptimalAdjacency']

# Cell
from .data.syntheticData import buildDataset
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import numpy as np

import community as community_louvain
import networkx as nx
import igraph as ig
import leidenalg
from itertools import chain

from dist_curve.curve_constructor import makeCurve, plotCurve
from dist_curve.model import getTrainedEstimator
from tqdm.notebook import tqdm
import seaborn as sns

# Cell
import os
if os.path.isdir("/ssdata/"):
    pth = "/ssdata/ClassPriorEstimation/model.hdf5"
else:
    pth = "/data/dzeiberg/ClassPriorEstimation/model.hdf5"
estimator = getTrainedEstimator(pth)

# Cell
def getBootstrapSample(p,u):
    ps = np.random.choice(np.arange(p.shape[0]), size=len(p), replace=True)
    ps = p[ps]
    us = np.random.choice(np.arange(u.shape[0]), size=len(u), replace=True)
    us = u[us]
    return ps, us

def getEsts(p,u, numbootstraps):
    curves = np.zeros((numbootstraps, 100))
    alphaHats = np.zeros(numbootstraps)
    for i in range(numbootstraps):
        ps, us = getBootstrapSample(p,u)
        curve = makeCurve(ps,us).reshape((1,-1))
        curves[i] = curve
        curve /= curve.sum()
        est = estimator(curve)
        alphaHats[i] = est
    return alphaHats, curves

def getBagAlphaHats(ds, numbootstraps=100):
    alphaHats =np.zeros((ds.N, numbootstraps))
    curves =np.zeros((ds.N, numbootstraps, 100))
    for bagIdx in tqdm(range(ds.N), total=ds.N, desc="getting bag estimates",leave=False):
        _,u = ds.getBag(bagIdx)
        ps, _ = list(zip(*[ds.getBag(int(i)) for i in range(ds.N)]))
        p = np.concatenate(ps)
        alphaHats[bagIdx], curves[bagIdx] = getEsts(p,u, numbootstraps)
    return alphaHats, curves


def getCliqueAlphaHats(ds, cliques, numbootstraps=10):
    Nc = len(cliques)
    alphaHats = np.zeros((Nc, numbootstraps))
    curves = np.zeros((Nc, numbootstraps, 100))
    for cnum, clique in tqdm(enumerate(cliques), total=Nc, desc="getting clique alpha ests", leave=False):
        _, us = list(zip(*[ds.getBag(int(i)) for i in clique]))
        ps, _ = list(zip(*[ds.getBag(int(i)) for i in range(ds.N)]))
        p = np.concatenate(ps)
        u = np.concatenate(us)
        alphaHats[cnum], curves[cnum] = getEsts(p,u, numbootstraps)
    return alphaHats, curves

def getAlphaPrime(cliques, cliqueEsts):
    bagNums = sorted(set(chain.from_iterable(cliques)))
    alphaPrime = np.zeros(len(bagNums))
    for bn in bagNums:
        inClique = [bn in c for c in cliques]
        alphaPrime[bn] = cliqueEsts[inClique].mean()
    return alphaPrime

# Cell
def getKSMatrixPMatrix(samples):
    "Get Kolmogrov-Smirnov adjacency matrix from lists of lists of samples for each bag"
    N = samples.shape[0]
    pmat = np.zeros((N,N))
    for bag0Idx in tqdm(range(N),total=N, desc="making kolmogorov-smirnov adj matrix", leave=False):
        for bag1Idx in range(bag0Idx+ 1,N):
            stat,p = ks_2samp(samples[bag0Idx], samples[bag1Idx])
            pmat[bag0Idx, bag1Idx] = p
            pmat[bag1Idx, bag0Idx] = p
    return pmat

def getAllCliques(mat, cutoffval=0.05):
    """
    given matrix of pairwise test p-values,
    make adjacency matrix using specified
    confidence level then find all cliques for each bag
    """
    adj = mat > cutoffval
    g = nx.from_numpy_array(adj)
    return list(nx.algorithms.clique.find_cliques(g))

def clusterByLeidenAlg(similarityMatrix, resolution_parameter = 1.5):
    """
    https://medium.com/@ciortanmadalina
    This method partitions input data by applying the Leiden algorithm
    on a given distance matrix.
    """
    # convert distance matrix to similariy matrix
    distanceMatrix = similarityMatrix
    edges = np.unravel_index(np.arange(distanceMatrix.shape[0]*distanceMatrix.shape[1]), distanceMatrix.shape)
    edges = list(zip(*edges))
    weights = distanceMatrix.ravel()

    g = ig.Graph(directed=False)
    g.add_vertices(distanceMatrix.shape[0])  # each observation is a node
    g.add_edges(edges)

    g.es['weight'] = weights
    weights = np.array(g.es["weight"]).astype(np.float64)
    partition_type = leidenalg.RBConfigurationVertexPartition
    partition_kwargs = {}
    partition_kwargs["weights"] = weights
    partition_kwargs["resolution_parameter"] = resolution_parameter
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groupAssignment = np.array(part.membership)
    groups = [np.where(groupAssignment==g)[0] for g in np.unique(groupAssignment)]
    return groups

# Cell
def getOptimalAdjacency(trueAlphas):
    N = trueAlphas.shape[0]
    adj = np.zeros((N,N))
    for i,a0 in enumerate(trueAlphas):
        for j,a1 in enumerate(trueAlphas[i+1:],start=i+1):
            adj[i,j] = np.abs(a0 - a1)
            adj[j,i] = np.abs(a0 - a1)
    return adj