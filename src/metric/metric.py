import numpy as np


def acc_perclass(pred, y, n_classes):
    accs = []
    for c in range(n_classes):
        tmp = pred[y == c]
        if tmp.shape[0] == 0:
            accs.append(0)
        else:
            accs.append(100. * tmp[tmp == c].shape[0] / tmp.shape[0])

    return np.asarray(accs)
