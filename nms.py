# coding=utf8

import numpy as np

def nms(proposals, threshold, type='Union'):
    '''
    Non-Maximum Suppression
    :param proposals:
    :param threshold:
    :param type:
    :return:
    '''
    h0 = proposals[:, 0]
    w0 = proposals[:, 1]
    h1 = proposals[:, 2]
    w1 = proposals[:, 3]
    volume = (h1 - h0 + 1) * (w1 - w0 + 1)

    scores = proposals[:, 4]
    I = scores.argsort()
    pick = []
    while len(I) > 0:
        inter_h0 = np.maximum(h0[I[-1]], h0[I[0:-1]])
        inter_w0 = np.maximum(w0[I[-1]], w0[I[0:-1]])
        inter_h1 = np.minimum(h1[I[-1]], h1[I[0:-1]])
        inter_w1 = np.minimum(w1[I[-1]], w1[I[0:-1]])
        h = np.maximum(0, inter_h1 - inter_h0 + 1)
        w = np.maximum(0, inter_w1 - inter_w0 + 1)
        inter = h * w
        if type == 'Min':
            iou = inter / np.minimum(volume[I[-1]], volume[I[0:-1]])
        else:
            iou = inter / (volume[I[-1]] + volume[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(iou <= threshold)[0]]
    return pick