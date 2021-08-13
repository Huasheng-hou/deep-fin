import numpy as np


def Zscore_norm(f, cols):
    for c in cols:
        mean = np.mean(np.array(f[c]))
        std = np.std(np.array(f[c]))
        if std:
            f[c] = f[c].apply(lambda x: (x - mean) / std)
