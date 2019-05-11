# coding=utf8

import tensorflow as tf

class BaseNet(object):

    def _get_conv_filter(self, name):
        with tf.variable_scope(name):
            shape = self.net_dict[name]['shape']
            initializer = tf.contrib.layers.xavier_initializer()
            var = tf.get_variable(
                'weights', shape=shape, initializer=initializer
            )
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)
            return var

    def _get_bias(self, name):
        with tf.variable_scope(name):
            shape = [self.net_dict[name]['shape'][3]]
            initializer = tf.constant_initializer(
                value=0., dtype=tf.float32
            )
            var = tf.get_variable(
                "biases", shape=shape, initializer=initializer
            )
            return var

    def _prelu(self, input, name='prelu'):
        with tf.variable_scope(name):
            alpha = tf.get_variable(
                'alpha', [input.get_shape()[-1]],
                initializer=tf.constant_initializer(0.25),
                dtype=tf.float32
            )
        pos = tf.nn.relu(input)
        neg = alpha * (input - abs(input)) * 0.5
        return pos + neg

    def _conv_layer(self, input, is_train, relu=True, batch_norm=False,
                    bias=True, name=None):
        with tf.name_scope(name) as scope:
            filter = self._get_conv_filter(name)
            strides = self.net_dict[name]['strides']
            output = tf.nn.conv2d(input, filter, strides, padding='VALID')
            if batch_norm:
                output = self._batch_norm(output, is_train, name=scope)
            else:
                if bias:
                    biases = self._get_bias(name)
                    output = tf.nn.bias_add(output, biases)
            if relu:
                output = self._prelu(output, name=scope)

            return output

    def _max_pool(self, input, name):
        with tf.name_scope(name):
            ksize = self.net_dict[name]['ksize']
            strides = self.net_dict[name]['strides']
            return tf.nn.max_pool(
                input, ksize=ksize, strides=strides, padding='VALID'
            )

    def _fc_layer(self, input, relu=True, bias=True, name=None):
        with tf.variable_scope(name):
            in_channels = self.net_dict[name]['in_channels']
            out_channels = self.net_dict[name]['out_channels']
            weights = tf.get_variable(
                'weights', shape=[in_channels, out_channels],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable(
                "biases", shape=[out_channels],
                initializer=tf.constant_initializer(
                    value=0., dtype=tf.float32
                )
            )
        output = tf.matmul(input, weights)
        if bias:
            output = tf.nn.bias_add(output, biases, name=name)
        if relu:
            output = self._prelu(output, name='prelu')
        return output

class PNet(BaseNet):

    def __init__(self):

        self.net_dict = {
            'conv_0': {'shape': [3, 3, 3, 10], 'strides': [1, 1, 1, 1]},
            'pool_0': {'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1]},
            'conv_1_0': {'shape': [3, 3, 10, 16], 'strides': [1, 1, 1, 1]},
            'conv_1_1': {'shape': [3, 3, 16, 32], 'strides': [1, 1, 1, 1]},
            'conv_2_0': {'shape': [1, 1, 32, 2], 'strides': [1, 1, 1, 1]},
            'conv_2_1': {'shape': [1, 1, 32, 4], 'strides': [1, 1, 1, 1]}
        }

    def inference(self, input, is_train, name='pnet'):
        '''

        :param input: shape(N, H, W, 3)
        :param is_train:
        :return:
        '''

        with tf.variable_scope(name):

            conv_0 = self._conv_layer(input, is_train, name='conv_0')
            pool_0 = self._max_pool(conv_0, name='pool_0')
            conv_1_0 = self._conv_layer(pool_0, is_train, name='conv_1_0')
            conv_1_1 = self._conv_layer(conv_1_0, is_train, name='conv_1_1')
            cls = self._conv_layer(conv_1_1, is_train, relu=False,
                                   name='conv_2_0')
            bbox_reg = self._conv_layer(conv_1_1, is_train, relu=False,
                                        name='conv_2_1')

            cls_prob = tf.nn.softmax(cls, name='cls_prob')

            return cls_prob, bbox_reg

class RNet(BaseNet):

    def __init__(self):

        self.net_dict = {
            'conv_0': {'shape': [3, 3, 3, 28], 'strides': [1, 1, 1, 1]},
            'pool_0': {'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1]},
            'conv_1': {'shape': [3, 3, 28, 48], 'strides': [1, 1, 1, 1]},
            'pool_1': {'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1]},
            'conv_2': {'shape': [2, 2, 48, 64], 'strides': [1, 1, 1, 1]},
            'fc_3': {'in_channels': 3 * 3 * 64, 'out_channels': 128},
            'cls_logits': {'in_channels': 128, 'out_channels': 2},
            'bbox_reg': {'in_channels': 128, 'out_channels': 4}
        }

    def inference(self, input, is_train, name='rnet'):
        '''

        :param input: shape(N, 24, 24, 3)
        :param is_train:
        :return:
        '''
        with tf.variable_scope(name):

            batch_size = tf.shape(input, name='shape_input')[0]

            conv_0 = self._conv_layer(input, is_train, name='conv_0')
            pool_0 = self._max_pool(conv_0, name='pool_0')
            conv_1 = self._conv_layer(pool_0, is_train, name='conv_1')
            pool_1 = self._max_pool(conv_1, name='pool_1')
            conv_2 = self._conv_layer(pool_1, is_train, name='conv_2')
            conv_2_reshape = tf.reshape(conv_2, [batch_size, -1],
                                        name='conv_2_reshape')
            fc_3 = self._fc_layer(conv_2_reshape, name='fc_3')
            cls_logits = self._fc_layer(fc_3, relu=False, name='cls_logits')
            bbox_reg = self._fc_layer(fc_3, relu=False, name='bbox_reg')

            cls_prob = tf.nn.softmax(cls_logits, name='cls_prob')

            return cls_prob, bbox_reg

class ONet(BaseNet):

    def __init__(self):

        self.net_dict = {
            'conv_0': {'shape': [3, 3, 3, 32], 'strides': [1, 1, 1, 1]},
            'pool_0': {'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1]},
            'conv_1': {'shape': [3, 3, 32, 64], 'strides': [1, 1, 1, 1]},
            'pool_1': {'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1]},
            'conv_2': {'shape': [3, 3, 64, 64], 'strides': [1, 1, 1, 1]},
            'pool_2': {'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1]},
            'conv_3': {'shape': [2, 2, 64, 128], 'strides': [1, 1, 1, 1]},
            'fc_4': {'in_channels': 3 * 3 * 128, 'out_channels': 256},
            'cls_logits': {'in_channels': 256, 'out_channels': 2},
            'bbox_reg': {'in_channels': 256, 'out_channels': 4},
            'landmark_reg': {'in_channels': 256, 'out_channels': 10}
        }

    def inference(self, input, is_train, name='onet'):
        '''

        :param input: shape(N, 48, 48, 3)
        :param is_train:
        :return:
        '''

        with tf.variable_scope(name):

            conv_0 = self._conv_layer(input, is_train, name='conv_0')
            pool_0 = self._max_pool(conv_0, name='pool_0')
            conv_1 = self._conv_layer(pool_0, is_train, name='conv_1')
            pool_1 = self._max_pool(conv_1, name='pool_1')
            conv_2 = self._conv_layer(pool_1, is_train, name='conv_2')
            pool_2 = self._max_pool(conv_2, name='pool_2')
            conv_3 = self._conv_layer(pool_2, is_train, name='conv_3')
            conv_3_reshape = tf.reshape(conv_3, [-1, 3 * 3 * 128],
                                        name='conv_3_reshape')
            fc_4 = self._fc_layer(conv_3_reshape, name='fc_4')

            cls_logits = self._fc_layer(fc_4, relu=False, name='cls_logits')
            bbox_reg = self._fc_layer(fc_4, relu=False, name='bbox_reg')
            pts_reg = self._fc_layer(fc_4, relu=False, name='landmark_reg')

            cls_prob = tf.nn.softmax(cls_logits, name='cls_prob')

            return cls_prob, bbox_reg, pts_reg