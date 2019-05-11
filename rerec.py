# coding=utf8

import numpy as np

def rerec(bbox):
    '''
    Convert to square
    :param bbox:
    :return:
    '''
    h = bbox[:, 2] - bbox[:, 0] + 1
    w = bbox[:, 3] - bbox[:, 1] + 1

    max_l = np.maximum(h, w)
    bbox[:, 0] = np.round(bbox[:, 0] + (h - max_l) * 0.5)
    bbox[:, 1] = np.round(bbox[:, 1] + (w - max_l) * 0.5)
    bbox[:, 2] = bbox[:, 0] + max_l - 1
    bbox[:, 3] = bbox[:, 1] + max_l - 1
    return bbox