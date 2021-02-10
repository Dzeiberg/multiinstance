# AUTOGENERATED! DO NOT EDIT! File to edit: 04_Distribution_Distance_Approaches.ipynb (unless otherwise specified).

__all__ = ['addTransformScores', 'splitIntoBags', 'getTransformScores', 'fitKDE', 'KLD', 'JSD', 'getJSDDistMat',
           'getKLDMat', 'getWassersteinMat', 'getOptimalAdjacency']

# Cell
from .utils import *

import seaborn as sns

import community as community_louvain
import networkx as nx

from .data.syntheticData import buildDataset,getBag
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from scipy.special import logsumexp
import scipy.stats as ss
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Cell

def addTransformScores(ds):
    P,U = list(zip(*[ds.getBag(i) for i in range(len(ds.numP))]))

    P = np.concatenate(P)
    U = np.concatenate(U)

    X = np.concatenate((P,U))
    Y = np.concatenate((np.ones(P.shape[0]),
                        np.zeros(U.shape[0])))

    clf = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100, max_samples=X.shape[0],
                            max_features=X.shape[1], bootstrap=True, bootstrap_features=False, oob_score=True).fit(X,Y)

    probP = clf.oob_decision_function_[:,1]

    roc_auc_score(Y, probP)

    Pprobs, Uprobs = splitIntoBags(probP,ds.numP, ds.numU)
    ds.Pprobs = Pprobs
    ds.Uprobs = Uprobs
    return ds

def splitIntoBags(probs, numP, numU):
    probsP, probsU = probs[:numP.sum()], probs[numP.sum():]
    pUpperIndices = np.concatenate(([0],np.cumsum(numP)))
    uUpperIndices = np.concatenate(([0],np.cumsum(numU)))
    P = np.zeros((len(numP), numP.max()))
    U = np.zeros((len(numU), numU.max()))
    for b in range(len(numP)):
        P[b,:numP[b]] = probsP[pUpperIndices[b]:pUpperIndices[b+1]]
        U[b,:numU[b]] = probsU[uUpperIndices[b] : uUpperIndices[b+1]]
    return P,U

def getTransformScores(ds,i):
    p = ds.Pprobs[i,:ds.numP[i]]
    u = ds.Uprobs[i,:ds.numU[i]]
    return p,u

# Cell
def fitKDE(vec):
    kde = KernelDensity(kernel="gaussian").fit(vec)
    return kde

def KLD(lnDensI,lnDensJ):
        return ss.entropy(np.exp(lnDensI), qk=np.exp(lnDensJ),base=2)

def JSD(ds, kdeI, i, j):
    _,uI = getTransformScores(ds,i)
    uI = uI.reshape((-1,1))
    _,uJ = getTransformScores(ds,j)
    uJ = uJ.reshape((-1,1))
    kdeJ = fitKDE(uJ)
    lnDensI0 = kdeI.score_samples(uI)
    lnDensJ0 = kdeJ.score_samples(uI)
    lnDensM0 = np.array([logsumexp((ldi,ldj),
                                       b=np.array([.5,.5])) for ldi,ldj in zip(lnDensI0, lnDensJ0)])
    lnDensI1 = kdeI.score_samples(uJ)
    lnDensJ1 = kdeJ.score_samples(uJ)
    lnDensM1 = np.array([logsumexp((ldi,ldj),
                                       b=np.array([.5,.5])) for ldi,ldj in zip(lnDensI1, lnDensJ1)])
    x = KLD(lnDensI0,lnDensM0)
    y = KLD(lnDensJ1, lnDensM1)
    return x + y

def getJSDDistMat(ds):
    N = ds.N
    dist = np.zeros((N,N))
    for i in range(N):
        _, uI = getTransformScores(ds,i)
        kdeI = fitKDE(uI.reshape((-1,1)))
        for j in range(i+1, N):
            jsd = JSD(ds, kdeI, i,j)
            dist[i,j] = jsd
            dist[j,i] = jsd
    return dist

def getKLDMat(ds):
    N = ds.N
    dist = np.zeros((N,N))
    for i in range(N):
        _, uI = getTransformScores(ds,i)
        uI = uI.reshape((-1,1))
        kdeI = fitKDE(uI)
        for j in range(N):
            _,uJ = getTransformScores(ds,j)
            uJ = uJ.reshape((-1,1))
            kdeJ = fitKDE(uJ)
            lnDensI = kdeI.score_samples(uI)
            lnDensJ = kdeJ.score_samples(uI)
            dist[i,j] = KLD(lnDensI, lnDensJ)
    return dist

def getWassersteinMat(ds):
    N = ds.N
    dist = np.zeros((N,N))
    for i in range(N):
        _, uI = getTransformScores(ds,i)
#         uI = uI.reshape((-1,1))
        for j in range(N):
            _,uJ = getTransformScores(ds,j)
#             uJ = uJ.reshape((-1,1))
            dist[i,j] = ss.wasserstein_distance(uI,uJ)
    return dist

def getOptimalAdjacency(trueAlphas):
    N = trueAlphas.shape[0]
    adj = np.zeros((N,N))
    for i,a0 in enumerate(trueAlphas):
        for j,a1 in enumerate(trueAlphas[i+1:],start=i+1):
            adj[i,j] = np.abs(a0 - a1)
            adj[j,i] = np.abs(a0 - a1)
    return adj