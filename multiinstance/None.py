

# Cell
from .utils import *
from .distanceApproaches import *
from .data.syntheticData import buildDataset,getBag

import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

import scipy.stats as ss

from tqdm.notebook import tqdm



# Cell
from .utils import *
from .distanceApproaches import *
from .data.syntheticData import buildDataset,getBag

import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

import scipy.stats as ss

from tqdm.notebook import tqdm



# Cell
from .utils import *
from .distanceApproaches import *
from .data.syntheticData import buildDataset,getBag

import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

import scipy.stats as ss

from tqdm.notebook import tqdm



# Cell
import autograd
from autograd import grad,jacobian,hessian
from autograd.scipy import stats as agss
import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scipy.stats as ss
import os
from scipy.optimize import minimize
from glob import glob

from .likelihoodMethods import *

import scipy.stats as ss

from .data.syntheticData import buildDataset
from .utils import *
from .agglomerative_clustering import AgglomerativeClustering

os.sched_setaffinity(0,set(range(20,40)))

# Cell
import autograd
from autograd import grad,jacobian,hessian
from autograd.scipy import stats as agss
import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scipy.stats as ss
import os
from scipy.optimize import minimize
from glob import glob

from .likelihoodMethods import *

import scipy.stats as ss

from .data.syntheticData import buildDataset
from .utils import *
from .agglomerative_clustering import AgglomerativeClustering

os.sched_setaffinity(0,set(range(20,40)))

# Cell
import autograd
from autograd import grad,jacobian,hessian
from autograd.scipy import stats as agss
import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scipy.stats as ss
import os
from scipy.optimize import minimize
from glob import glob

from .likelihoodMethods import *

import scipy.stats as ss

from .data.syntheticData import buildDataset
from .utils import *
from .agglomerative_clustering import AgglomerativeClustering

os.sched_setaffinity(0,set(range(20,40)))