# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"getBag": "02_SyntheticData.ipynb",
         "buildDatasetDict": "02_SyntheticData.ipynb",
         "Dataset": "11_Construct_Datasets.ipynb",
         "buildDataset": "11_Construct_Datasets.ipynb",
         "estimator": "03_Dataset_Utils.ipynb",
         "addTransformScores": "03_Dataset_Utils.ipynb",
         "splitIntoBags": "03_Dataset_Utils.ipynb",
         "getTransformScores": "03_Dataset_Utils.ipynb",
         "getBootstrapSample": "03_Dataset_Utils.ipynb",
         "estimate": "03_Dataset_Utils.ipynb",
         "getEsts": "03_Dataset_Utils.ipynb",
         "getBagAlphaHats": "03_Dataset_Utils.ipynb",
         "getCliqueAlphaHats": "03_Dataset_Utils.ipynb",
         "getAlphaPrime": "03_Dataset_Utils.ipynb",
         "addGlobalEsts": "03_Dataset_Utils.ipynb",
         "addBagAlphaHats": "03_Dataset_Utils.ipynb",
         "eng": "03_Dataset_Utils.ipynb",
         "path": "03_Dataset_Utils.ipynb",
         "getKSMatrixPMatrix": "03_Dataset_Utils.ipynb",
         "getAllCliques": "03_Dataset_Utils.ipynb",
         "clusterByLeidenAlg": "03_Dataset_Utils.ipynb",
         "getOptimalAdjacency": "04_Distribution_Distance_Approaches.ipynb",
         "fitKDE": "04_Distribution_Distance_Approaches.ipynb",
         "KLD": "04_Distribution_Distance_Approaches.ipynb",
         "JSD": "04_Distribution_Distance_Approaches.ipynb",
         "getJSDDistMat": "04_Distribution_Distance_Approaches.ipynb",
         "getKLDMat": "04_Distribution_Distance_Approaches.ipynb",
         "getWassersteinMat": "04_Distribution_Distance_Approaches.ipynb",
         "AgglomerativeClustering": "06_Agglomerative_Clustering_KS_on_One_D_Scores.ipynb",
         "WardClustering": "07_Ward_Clustering.ipynb",
         "getAlphaHat": "10_Gradient_With_Likelihood.ipynb",
         "getAlphaLoss": "10_Gradient_With_Likelihood.ipynb",
         "getWLoss": "10_Gradient_With_Likelihood.ipynb",
         "gradientMethod": "09_Gradient_Method.ipynb",
         "getAlphaLossWithLL": "10_Gradient_With_Likelihood.ipynb",
         "getParams": "11_Construct_Datasets.ipynb",
         "getBagDict": "11_Construct_Datasets.ipynb",
         "getGlobalAlphaHat": "12_Gradient_With_Size_Penalty.ipynb",
         "initDS": "12_Gradient_With_Size_Penalty.ipynb",
         "addEsts": "12_Gradient_With_Size_Penalty.ipynb",
         "aL0": "12_Gradient_With_Size_Penalty.ipynb",
         "wL0": "12_Gradient_With_Size_Penalty.ipynb",
         "g1": "12_Gradient_With_Size_Penalty.ipynb",
         "plotResults": "12_Gradient_With_Size_Penalty.ipynb",
         "initRealDS": "12_Gradient_With_Size_Penalty.ipynb",
         "logLikelihood": "22_Quarantine_DistCurve.ipynb",
         "getChildren": "22_Quarantine_DistCurve.ipynb",
         "treeNegativeLogLikelihood": "22_Quarantine_DistCurve.ipynb",
         "runAlgorithm": "22_Quarantine_DistCurve.ipynb",
         "plotMAE": "22_Quarantine_DistCurve.ipynb",
         "plotDistrs": "22_Quarantine_DistCurve.ipynb",
         "prepDS": "24_Likelihood_Cleaned.ipynb",
         "LikelihoodMethod": "25_Likelihood_With_Ward.ipynb",
         "getSequence": "27_Read_Varibench.ipynb",
         "KMeansInit": "33_EM.ipynb",
         "EM": "33_EM.ipynb",
         "MultiEM": "33_EM.ipynb",
         "generateBags": "33_EM.ipynb",
         "PULoss": "38_Chainer_NNPU.ipynb",
         "pu_loss": "38_Chainer_NNPU.ipynb",
         "MLP": "38_Chainer_NNPU.ipynb",
         "getPosterior": "43_TF_NNPU.ipynb",
         "DensityRatioEM": "47_DREM_2.ipynb",
         "mixture": "40_Gaussian_DG.ipynb",
         "DataGenerator": "40_Gaussian_DG.ipynb",
         "GaussianDG": "40_Gaussian_DG.ipynb",
         "UniformDG": "40_Gaussian_DG.ipynb",
         "NormalMixDG": "40_Gaussian_DG.ipynb",
         "MVNormalMixDG": "40_Gaussian_DG.ipynb",
         "NormalMixRandomParameters": "40_Gaussian_DG.ipynb",
         "NormalMixParameters": "40_Gaussian_DG.ipynb",
         "min_aucpn": "40_Gaussian_DG.ipynb",
         "GaussianMixtureDataGenerator": "40_Gaussian_DG.ipynb",
         "Basic": "43_TF_NNPU.ipynb",
         "NNPULoss": "43_TF_NNPU.ipynb",
         "NNPUAbsLoss": "43_TF_NNPU.ipynb",
         "gradients": "43_TF_NNPU.ipynb",
         "batch": "43_TF_NNPU.ipynb",
         "batchPos": "43_TF_NNPU.ipynb",
         "batchUL": "43_TF_NNPU.ipynb",
         "batchY": "43_TF_NNPU.ipynb",
         "RankNet": "45_Rank.ipynb",
         "RankerNN": "45_Rank-Copy1.ipynb",
         "RankNetNN": "45_Rank-Copy1.ipynb",
         "LambdaRankNN": "45_Rank-Copy1.ipynb",
         "RankNet2": "45_Rank.ipynb",
         "RankNetTies": "45_Rank.ipynb",
         "getModel": "46_VPU.ipynb",
         "train": "46_VPU.ipynb",
         "generateParams": "47_DREM_2.ipynb"}

modules = ["data/syntheticData.py",
           "utils.py",
           "distanceApproaches.py",
           "agglomerative_clustering.py",
           "ward_clustering.py",
           "gradientMethod.py",
           "data/realData.py",
           "likelihoodMethods.py",
           "likelihood_method.py",
           "tree_likelihood.py",
           "data/gene.py",
           "em.py",
           "nnpu_original.py",
           "density_ratio_em.py",
           "data/gaussian_dg.py",
           "nnpu.py",
           "ranker.py",
           "vpu.py",
           "lambdaRank.py"]

doc_url = "https://Dzeiberg.github.io/multiinstance/"

git_url = "https://github.com/Dzeiberg/multiinstance/tree/master/"

def custom_doc_links(name): return None
