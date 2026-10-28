import math
from scipy.stats import t
import numpy as np

def mean_ci(values):
    x=np.asarray(values,dtype=float)
    if len(x)<3:raise ValueError('At least three seeds required')
    return {'mean':float(x.mean()),'ci95_halfwidth':float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))), 'n_seeds':len(x)}
