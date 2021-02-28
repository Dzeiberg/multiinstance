# AUTOGENERATED! DO NOT EDIT! File to edit: 16_Likelihood_Method.ipynb (unless otherwise specified).

__all__ = ['normalizedlogLik', 'getChildren', 'treeNegativeLogLikelihood']

# Cell
import autograd
from autograd import grad,jacobian,hessian
from autograd.scipy import stats as agss
import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import scipy.stats as ss

from scipy.optimize import minimize

# Cell
def normalizedlogLik(xi,mu,sigma):
    return (1/len(xi))*(-len(xi)/2 * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2)) * np.sum((xi - mu)**2))

def getChildren(idx,N):
    if idx > N - 1:
        return np.array([idx])
    left = 2 * idx + 1
    right = left + 1

    return np.concatenate([getChildren(left,N),getChildren(right,N)])

def treeNegativeLogLikelihood(x,leafN):
    def LL(leafMeans,bagSigma):
        NBags = len(bagSigma)
        NInternal_Nodes = np.floor(NBags/2)
#         NLeaves = NBags - NInternal_Nodes
        ll = 0
        for idx in range(NBags):
            leafIndices = (getChildren(idx, NInternal_Nodes) - NInternal_Nodes).astype(int)
            ln = leafN[leafIndices]
            mu = np.dot(leafMeans[leafIndices],ln)/np.sum(ln)
            sigma = bagSigma[idx]
            ll = ll + normalizedlogLik(x[idx],mu,sigma)
        return -1 * ll
    return LL