import numpy as np

import math


def acc_perclass(pred, y, n_classes):
    accs = []
    for c in range(n_classes):
        tmp = pred[y == c]
        if tmp.shape[0] == 0:
            accs.append(0)
        else:
            accs.append(100. * tmp[tmp == c].shape[0] / tmp.shape[0])

    return np.asarray(accs)


def confusion_matrix(pred, y, n_classes):
    m = np.zeros([n_classes, n_classes])
    pred = np.asarray(pred).flatten()
    y = np.asarray(y).flatten()
    for idx in range(y.shape[0]):
        m[y[idx], pred[idx]] += 1

    return m


def MCC(c_m):
    tp = c_m[1][1]
    fn = c_m[1][0]
    tn = c_m[0][0]
    fp = c_m[0][1]

    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    return mcc
