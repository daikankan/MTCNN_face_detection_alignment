# coding=utf8

import cv2
import numpy as np
from cut import cut
from nms import nms
from rerec import rerec
from bbreg import bbreg
from loader import PNetPredictor, RNetPredictor, ONetPredictor

class MTCNN(object):

    def __init__(self, pnet_model_path, rnet_model_path, onet_model_path):

        self.pnet = PNetPredictor(pnet_model_path)
        self.rnet = RNetPredictor(rnet_model_path)
        self.onet = ONetPredictor(onet_model_path)

    def detect(self, img, min_size=40, factor=0.709,
               score_threshold=[0.5, 0.5, 0.5],
               iou_threshold=[0.5, 0.7, 0.7, 0.7]):
        '''

        :param img: numpy array (shape [height, width, 3] )
        :return: boxes: numpy array (shape [N, 5])
                                [h0, w0, h1, w1, score]
                 landmarks: numpy array (shape [N, 10]
                            [h_leye, h_reye, h_nose, h_lmouth, h_rmouth,
                             w_leye, w_reye, w_nose, w_lmouth, w_rmouth]
        '''
        # Stage 1
        h, w, _ = img.shape
        min_l = min(h, w)
        m = 12. / min_size
        min_l = min_l * m
        scales = []
        while min_l >= 12:
            scales.append(m)
            min_l *= factor
            m *= factor

        proposals_total = np.zeros((0, 9), dtype=np.float32)
        for scale in scales:
            hs = int(np.round(h * scale))
            ws = int(np.round(w * scale))
            img_resize = cv2.resize(
                img, (ws, hs), interpolation=cv2.INTER_LINEAR
            )
            img_resize = (img_resize - 127.5) * 0.0078125
            img_resize = img_resize[np.newaxis, :, :, :]
            try:
                cls_prob, bbox_reg = self.pnet.predict(img_resize)
            except:
                continue
            prob = cls_prob[..., 1]
            idx_pos = np.where(prob >= score_threshold[0])
            prob_pos = prob[idx_pos].reshape([-1, 1])
            h0 = idx_pos[1].reshape([-1, 1])
            w0 = idx_pos[2].reshape([-1, 1])
            bbox_pos = np.concatenate((h0, w0, h0+5.5, w0+5.5), axis=1)
            bbox_pos = np.round(bbox_pos * 2 / scale)
            bbox_reg_pos = bbox_reg[idx_pos]
            proposals = np.concatenate(
                (bbox_pos, prob_pos, bbox_reg_pos), axis=1
            )
            proposals = proposals[nms(proposals, iou_threshold[0])]
            proposals_total = np.concatenate(
                (proposals_total, proposals), axis=0
            )
        if len(proposals_total) == 0:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 10), dtype=np.float32))

        proposals_total = proposals_total[nms(proposals_total,
                                          threshold=iou_threshold[1])]
        proposals_total = bbreg(proposals_total)

        # Stage 2
        proposals_total = rerec(proposals_total)
        boxes = proposals_total[:, 0:4].astype(np.int32)
        input = cut(img, boxes, h, w, 24)
        input = (input - 127.5) * 0.0078125
        try:
            cls_prob, bbox_reg = self.rnet.predict(input)
        except:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 10), dtype=np.float32))

        prob = cls_prob[..., 1]
        idx_pos = np.where(prob >= score_threshold[1])
        prob_pos = prob[idx_pos].reshape([-1, 1])
        bbox_pos = boxes[idx_pos]
        bbox_reg_pos = bbox_reg[idx_pos]
        proposals_total = np.concatenate(
            (bbox_pos, prob_pos, bbox_reg_pos), axis=1
        )
        if len(proposals_total) == 0:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 10), dtype=np.float32))

        proposals_total = proposals_total[
            nms(proposals_total, threshold=iou_threshold[2])
        ]
        proposals_total = bbreg(proposals_total)

        # Stage 3
        proposals_total = rerec(proposals_total)
        boxes = proposals_total[:, 0:4].astype(np.int32)
        input = cut(img, boxes, h, w, 48)
        input = (input - 127.5) * 0.0078125
        try:
            cls_prob, bbox_reg, landmark_reg = self.onet.predict(input)
        except:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 10), dtype=np.float32))

        prob = cls_prob[..., 1]
        idx_pos = np.where(prob >= score_threshold[2])
        prob_pos = prob[idx_pos].reshape([-1, 1])
        bbox_pos = boxes[idx_pos]
        # refine landmark
        height = bbox_pos[:, 2:3] - bbox_pos[:, 0:1] + 1
        width = bbox_pos[:, 3:4] - bbox_pos[:, 1:2] + 1
        landmark = landmark_reg[idx_pos]
        landmark[:, 0:5] = np.round(
            bbox_pos[:, 0:1] + landmark[:, 0:5] * height
        )
        landmark[:, 5:10] = np.round(
            bbox_pos[:, 1:2] + landmark[:, 5:10] * width
        )
        bbox_reg_pos = bbox_reg[idx_pos]
        proposals_total = np.concatenate(
            (bbox_pos, prob_pos, bbox_reg_pos), axis=1
        )
        if len(proposals_total) == 0:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 10), dtype=np.float32))

        proposals_total = bbreg(proposals_total)
        idx_keep = nms(proposals_total, threshold=iou_threshold[2],
                       type='Min')

        return proposals_total[idx_keep][:, 0:5], landmark[idx_keep]