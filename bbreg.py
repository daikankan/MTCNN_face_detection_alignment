# coding=utf8

import numpy as np

def bbreg(bbox):
    '''
    Refine bounding box
    :param bbox:
    :return:
    '''
    height = bbox[:, 2:3] - bbox[:, 0:1] + 1
    width = bbox[:, 3:4] - bbox[:, 1:2] + 1
    bbox[:, 0:4] = np.round(
        bbox[:, 0:4] + bbox[:, 5:9] *
        np.concatenate((height, width, height, width), axis=1)
    )
    return bbox