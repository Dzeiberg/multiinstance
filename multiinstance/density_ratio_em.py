# AUTOGENERATED! DO NOT EDIT! File to edit: 39_DREM.ipynb (unless otherwise specified).

__all__ = ['DensityRatioEM']

# Cell
from .em import generateBags
from .utils import estimate
from .nnpu import getPosterior as getNNPUPosterior

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from easydict import EasyDict
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm

# Cell
class DensityRatioEM:
    def __init__(self, bags,n_clusters):
        self.bags = bags
        self.n_clusters = n_clusters

    def findGlobalClusters(self):
        "Run K-Means on the positives from all bags then assign each unlabeled point to a cluster based on the resulting clusters of K-Means"
        globalPositives = np.concatenate([b.X_pos for b in bags])
        kmeans = KMeans(n_clusters=self.n_clusters).fit(globalPositives)
        # Cluster Unlabeled
        for bagNum,b in enumerate(self.bags):
            self.bags[bagNum].unlabeled_cluster_assignment = kmeans.predict(b.x_unlabeled)
            self.bags[bagNum].positive_cluster_assignment = kmeans.predict(b.X_pos)
        self.kmeans = kmeans

    def getClusterEstimates(self,componentInfo=None):
        "Estimate the class prior and density ratios of the unlabeled points for each cluster"
        self.clusterAlphaHats= np.zeros(self.n_clusters)
        # NClusters x NBags size list containing the density ratio for the unlabeled points
        # from the specified bag in the specified cluster
        self.bagRatios = []
        for cnum in range(self.n_clusters):
            unlabeledInCluster = [b.x_unlabeled[b.unlabeled_cluster_assignment == cnum] for b in self.bags]
            posInCluster = [b.X_pos[b.positive_cluster_assignment == cnum] for b in self.bags]
            unlabeled = np.concatenate(unlabeledInCluster)
            positive = np.concatenate(posInCluster)
            # estimate class prior
            tau, aucpu = getOptimalTransform(np.concatenate((positive, unlabeled)),
                                         np.concatenate((np.ones(positive.shape[0]),
                                                         np.zeros(unlabeled.shape[0]))))
            tau_pos = np.ascontiguousarray(tau[:positive.shape[0]].reshape((-1,1)))
            tau_u = np.ascontiguousarray(tau[positive.shape[0]:].reshape((-1,1)))
            self.clusterAlphaHats[cnum],_ = estimate(tau_pos, tau_u)
            ####
            #self.clusterAlphaHats[cnum],_ = estimate(positive, unlabeled)
            # Estimate density ratio for all unlabeled points in each bag that are in this cluster
            self.bagRatios.append(self.estimateClusterDensityRatio(posInCluster,
                                                                   unlabeledInCluster,
                                                                   cnum,
                                                                   componentInfo))

    def ratioFromPosteriorVec(self, posts, alpha):
        return (alpha * (1 - posts)) / (posts * (1 - alpha))

    def estimateClusterDensityRatio(self,posInCluster,unlabeledInCluster,cnum,componentInfo=None,
                                 args=EasyDict(d={'batchsize': 128,
                                                  'hdim': 4,
                                                  'epochs': 100,
                                                  'lr': 0.001,
                                                  'weightDecayRate': 0.005})):
        p = np.concatenate(posInCluster)
        u = np.concatenate(unlabeledInCluster)
        # PU Labels {1: pos, -1: unlabeled}
        y = np.concatenate((np.ones(p.shape[0]),
                            np.ones(u.shape[0])*-1)).astype(np.int32)
        # Run NNPU
        if componentInfo is None:
            posteriors = getNNPUPosterior(np.concatenate((p,u)).astype(np.float32),
                                          y,
                                          self.clusterAlphaHats[cnum],
                                          args = args)
            # convert cluster posterior to density ratio
            ratios = np.nan_to_num(self.ratioFromPosteriorVec(posteriors, self.clusterAlphaHats[cnum]))
        else:
            clusterMap = cdist(self.kmeans.cluster_centers_, componentInfo.posMeans).argmin(1)
            # pos
            cnum = clusterMap[cnum]
            f1 = ss.multivariate_normal.pdf(np.concatenate((p,u)),
                                 mean=componentInfo.posMeans[cnum],
                                 cov=componentInfo.posCovs[cnum])
            # Neg
            f0 = ss.multivariate_normal.pdf(np.concatenate((p,u)),
                                 mean=componentInfo.negMeans[cnum],
                                 cov=componentInfo.negCovs[cnum])
            ratios = f0/f1

        # Remove positive points from posterior list
        ratios = ratios[p.shape[0]:]
        # Store the ratios for the unlabeled set of each bag
        bagRatios = []
        # Get ratios for unlabeled sets of each bag
        idx = 0
        for bagNum in range(len(posInCluster)):
            numU = unlabeledInCluster[bagNum].shape[0]
            bagRatios.append(ratios[idx:idx+numU])
            idx += numU
        return bagRatios

    def EM(self,NIters=500):
        self.eta = np.zeros((len(self.bags), self.n_clusters))
        for cnum in range(self.n_clusters):
            for bagNum, b in enumerate(self.bags):
                ratios = self.bagRatios[cnum][bagNum]
                eta_i_j = np.array(.5)
                for em_iter in range(NIters):
                    den = eta_i_j + (1 - eta_i_j) * ratios
                    eta_i_j = np.mean(eta_i_j / den)
                self.eta[bagNum,cnum] = eta_i_j

    def run(self,componentInfo=None):
        self.findGlobalClusters()
        self.getClusterEstimates(componentInfo)
        self.EM()
        self.estimateBagParameters()

    def posterior(self, bagNum, clusterNum):
        eta_i_j = self.eta[bagNum, clusterNum]
        densityRatios = self.bagRatios[clusterNum][bagNum]
        return eta_i_j / (eta_i_j + (1 - eta_i_j)*densityRatios)

    def getAUC(self):
        labels = []
        posts = []
        for bagNum in range(len(self.bags)):
            for cnum in range(self.n_clusters):
                posts.append(self.posterior(bagNum,cnum))
                labels.append(self.bags[bagNum].hiddenLabels[self.bags[bagNum].unlabeled_cluster_assignment == cnum])
        labels = np.concatenate(labels)
        posts = np.concatenate(posts)
        return roc_auc_score(labels, posts)

    def estimateBagParameters(self):
        N = len(self.bags)
        self.alphaHats = np.zeros(N)
        self.pi = np.zeros((N,self.n_clusters))
        self.rho = np.zeros((N,self.n_clusters))
        for bagNum, b in enumerate(self.bags):
            eta_j = self.eta[bagNum]
            gamma_j = np.unique(b.unlabeled_cluster_assignment,
                                return_counts=True)[1] / b.unlabeled_cluster_assignment.shape[0]
            alpha_j = eta_j.dot(gamma_j)
            pi_j = np.multiply(eta_j, gamma_j) / alpha_j
            rho_j = np.multiply(1 - eta_j, gamma_j) / (1 - alpha_j)
            self.alphaHats[bagNum] = alpha_j
            self.pi[bagNum] = pi_j
            self.rho[bagNum] = rho_j