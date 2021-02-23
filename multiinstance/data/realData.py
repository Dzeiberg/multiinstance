# AUTOGENERATED! DO NOT EDIT! File to edit: 11_Construct_Datasets.ipynb (unless otherwise specified).

__all__ = ['getParams', 'getBagDict', 'Dataset', 'buildDataset']

# Cell
import numpy as np

import mat73

from scipy.io import loadmat
import h5py

# Cell
def getParams(nPDistr=lambda: np.random.poisson(25) + 1,
              nUDistr=lambda: np.random.poisson(30) + 1,
              alphaDistr=lambda: np.random.beta(2,10)):
    nP = nPDistr()
    nU = nUDistr()
    alpha = alphaDistr()
    numUnlabeledPos = max(1,int(alpha * nU))
    numUnlabeledNeg = max(1,nU - numUnlabeledPos)

    return nP, nU, alpha, numUnlabeledPos, numUnlabeledNeg

# Cell
def getBagDict(nP, nU, alpha, numUnlabeledPos, numUnlabeledNeg, d):
    # get indices of all positives and negatives
    posIdxs = np.where(d["y"] == 1)[0]
    negIdxs = np.where(d["y"] == 0)[0]
    # sample positives
    posSampleIDXS = np.random.choice(posIdxs,replace=True,size=nP)
    # sample unlabeled
    unlabeledPosSampleIDXS = np.random.choice(posIdxs,replace=True,size=numUnlabeledPos)
    unlabeledNegSampleIDXS = np.random.choice(negIdxs,replace=True,size=numUnlabeledNeg)
    unlabeledSampleIDXS = np.concatenate((unlabeledPosSampleIDXS, unlabeledNegSampleIDXS))
    posInstances = d["X"][posSampleIDXS]
    unlabeledInstances = d["X"][unlabeledSampleIDXS]
    hiddenLabels = np.concatenate((np.ones(numUnlabeledPos),
                                   np.zeros(numUnlabeledNeg)))
    return {"positiveInstances": posInstances,
            "unlabeledInstances": unlabeledInstances,
            "hiddenLabels": hiddenLabels,
            "alpha_i": alpha,
            "nP": nP,
            "nU": nU}

# Cell
class Dataset:
    def __init__(self, d):
        self.positiveInstances = d["positiveInstances"]
        self.unlabeledInstances = d["unlabeledInstances"]
        self.trueAlphas = d["alpha_i"]
        self.N = self.positiveInstances.shape[0]
        self.numP = d["numP"]
        self.numU = d["numU"]
        self.hiddenLabels = d["hiddenLabels"]

    def getBag(self,idx):
        p = self.positiveInstances[idx, :self.numP[idx]]
        u = self.unlabeledInstances[idx, :self.numU[idx]]
        return p,u

    def getAlpha(self,idx):
        return self.trueAlphas[idx]

    def __len__(self):
        return self.N

# Cell
def buildDataset(dsPath, size,
                 nPDistr=lambda: np.random.poisson(25) + 1,
                 nUDistr=lambda: np.random.poisson(30) + 1,
                 alphaDistr=lambda: np.random.beta(2,10)):
    try:
        ds = loadmat(dsPath)
    except:
        ds= {}
        for k,v in h5py.File(dsPath,"r").items():
            ds[k] = np.array(v)
    bags = []
    for bag in range(size):
        nP, nU, alpha, numUnlabeledPos, numUnlabeledNeg = getParams(nPDistr=nPDistr,
                                                                    nUDistr=nUDistr,
                                                                    alphaDistr=alphaDistr)
        bagDict = getBagDict(nP, nU, alpha, numUnlabeledPos, numUnlabeledNeg, ds)
        bags.append(bagDict)
    # calculate max num Pos and Unlabeled to set sizes for matrices
    maxP = np.max([d["nP"] for d in bags])
    maxU = np.max([d["nU"] for d in bags])
    dim = bags[0]["positiveInstances"].shape[1]
    # init matrices
    posMats = np.zeros((len(bags), maxP, dim))
    unlabeledMats = np.zeros((len(bags), maxU, dim))
    hiddenLabelMats = np.zeros((len(bags), maxU))
    alphas = np.zeros((len(bags), 1))
    numPos = np.zeros(len(bags),dtype=int)
    numU = np.zeros(len(bags),dtype=int)
    # fill matrices with bags
    for bagNum,bag in enumerate(bags):
        posPadding = maxP - bag["nP"]
        unlabeledPadding = maxU - bag["nU"]
        p_mat= np.concatenate((bag["positiveInstances"],
                               np.zeros((posPadding, dim))), axis=0)
        posMats[bagNum] = p_mat
        u_mat= np.concatenate((bag["unlabeledInstances"],
                               np.zeros((unlabeledPadding, dim))), axis=0)
        unlabeledMats[bagNum] = u_mat
        hiddenLabelMats[bagNum] = np.concatenate((bag["hiddenLabels"],
                                                  np.zeros(unlabeledPadding)))
        alphas[bagNum] = bag["alpha_i"]
        numPos[bagNum] = bag["nP"]
        numU[bagNum] = bag["nU"]
    dataset = Dataset({
        "positiveInstances": posMats,
        "unlabeledInstances": unlabeledMats,
        "alpha_i": alphas,
        "numP": numPos,
        "numU": numU,
        "hiddenLabels": hiddenLabelMats
    })
    return dataset