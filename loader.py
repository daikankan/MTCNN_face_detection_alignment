# coding=utf8

import numpy as np
import tensorflow as tf

class PNetPredictor(object):

    def __init__(self, model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            tf.saved_model.loader.load(
                self.sess, ["serve"], model_path
            )
            self.input_placeholder = \
                self.graph.get_tensor_by_name('inputs:0')
            self.cls_prob_tensor = \
                self.graph.get_tensor_by_name('pnet/cls_prob:0')
            self.bbox_reg_tensor = \
                self.graph.get_tensor_by_name('pnet/conv_2_1/BiasAdd:0')

    def predict(self, input):
        '''

        :param input: shape(N, Height, width, 3)
        :return: cls_prob: shape(N, H, W, 2)
                 box_reg: shape(N, H, W, 4)
                 [pt0_h, pt0_w, pt1_h, pt1_w]
        '''

        feed_dict = {self.input_placeholder: input.astype(np.float32)}
        return self.sess.run((self.cls_prob_tensor,
                              self.bbox_reg_tensor),
                             feed_dict=feed_dict)

class RNetPredictor(object):

    def __init__(self, model_path):

        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.saved_model.loader.load(
                self.sess, ["serve"], model_path
            )
            self.input_placeholder = \
                self.graph.get_tensor_by_name('rnet/inputs:0')
            self.cls_prob_tensor = \
                self.graph.get_tensor_by_name('rnet/cls_prob:0')
            self.bbox_reg_tensor = \
                self.graph.get_tensor_by_name('rnet/bbox_reg_1:0')

    def predict(self, input):
        '''

        :param input: shape(N, 24, 24, 3)
        :return: cls_prob: shape(N, 2)
                 bbox_reg: shape(N, 4)
                 [pt0_h, pt0_w, pt1_h, pt1_w]
        '''

        feed_dict = {self.input_placeholder: input.astype(np.float32)}
        return self.sess.run((self.cls_prob_tensor,
                              self.bbox_reg_tensor),
                             feed_dict=feed_dict)

class ONetPredictor(object):

    def __init__(self, model_path):

        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.saved_model.loader.load(
                self.sess, ["serve"], model_path
            )
            self.input_placeholder = \
                self.graph.get_tensor_by_name('onet/inputs:0')
            self.cls_prob_tensor = \
                self.graph.get_tensor_by_name('onet/cls_prob:0')
            self.bbox_reg_tensor = \
                self.graph.get_tensor_by_name('onet/bbox_reg_1:0')
            self.pts_reg_tensor = \
                self.graph.get_tensor_by_name('onet/landmark_reg_1:0')

    def predict(self, input):
        '''

        :param input: shape(N, 48, 48, 3)
        :return: cls_prob: shape(N, 2)
                 bbox_reg: shape(N, 4) [pt0_h, pt0_w, pt1_h, pt1_w]
                 landmark_reg: shape(N, 10)
        '''

        feed_dict = {self.input_placeholder: input.astype(np.float32)}
        return self.sess.run((self.cls_prob_tensor,
                              self.bbox_reg_tensor,
                              self.pts_reg_tensor),
                             feed_dict=feed_dict)