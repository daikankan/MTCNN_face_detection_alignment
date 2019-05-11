# coding=utf8

import cv2
import numpy as np

def cut(img, bbox, h, w, size_input):
    '''
    Cut patches from image, pad if necessary, then resize.
    :param img:
    :param bbox:
    :param h:
    :param w:
    :param size_input:
    :return:
    '''
    input = np.zeros((len(bbox), size_input, size_input, 3),
                     dtype=np.float32)
    for i, box in enumerate(bbox):
        crop_h0 = max(0, box[0])
        crop_w0 = max(0, box[1])
        crop_h1 = min(h - 1, box[2])
        crop_w1 = min(w - 1, box[3])
        cropped_img = img[crop_h0:crop_h1 + 1,
                          crop_w0:crop_w1 + 1, :]
        h0 = - box[0]
        w0 = - box[1]
        h1 = box[2] - h + 1
        w1 = box[3] - w + 1
        if (w0 > 0 or h0 > 0 or w1 > 0 or h1 > 0):
            w0 = max(w0, 0)
            h0 = max(h0, 0)
            w1 = max(w1, 0)
            h1 = max(h1, 0)
            cropped_img = cv2.copyMakeBorder(
                cropped_img, h0, h1, w0, w1,
                cv2.BORDER_CONSTANT, (0, 0, 0)
            )
        input[i] = cv2.resize(
            cropped_img, (size_input, size_input),
            interpolation=cv2.INTER_LINEAR
        )
    return input