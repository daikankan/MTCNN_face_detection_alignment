# coding=utf8

import cv2
import time
from face_detect import MTCNN

if __name__ == '__main__':

    pnet_model_path = './models/pnet'
    rnet_model_path = './models/rnet'
    onet_model_path = './models/onet'

    mtcnn = MTCNN(pnet_model_path,
                  rnet_model_path,
                  onet_model_path)

    img_path = './test1.jpg'
    img = cv2.imread(img_path)

    time_start = time.time()

    bounding_boxes, landmarks = mtcnn.detect(
        img=img, min_size=80, factor=0.709,
        score_threshold=[0.6, 0.6, 0.6]
    )

    time_total = (time.time() - time_start)
    print('time: {} (ms)'.format(time_total * 1000))

    # visualize
    h, w, c = img.shape
    for idx, bbox in enumerate(bounding_boxes):
        score = bbox[4]
        # If out of bound
        h0 = max(int(round(bbox[0])), 0)
        w0 = max(int(round(bbox[1])), 0)
        h1 = min(int(round(bbox[2])), h - 1)
        w1 = min(int(round(bbox[3])), w - 1)
        cv2.rectangle(img, (w0, h0), (w1, h1), (0, 255, 0), 2)

        landmark = landmarks[idx]
        for i in range(5):
            pt_h = landmark[i]
            pt_w = landmark[i + 5]
            if 0 <= pt_h and pt_h < h and 0 <= pt_w and pt_w < w:
                cv2.circle(img, (pt_w, pt_h), 1, (0, 255, 0),
                           thickness=3)

    cv2.imshow('test', img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()