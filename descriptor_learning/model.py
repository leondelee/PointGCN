#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import *
from tensorpack.tfutils import get_current_tower_context, summary
from dataflow import  MyDataflow
import multiprocessing
import numpy as np
import pickle
import ops
import sys
sys.path.append('./src/build')
import sdv

INPUT_DIM = int(4096)


def conv_block(input_anc, input_pos, channels, dropout_flag, dropout_rate, laxer_idx, stride_input=1, k_size=3,
               padding_type='SAME'):

    # Traditional 3D conv layer followed by batch norm and relu activation

    i_size = input_anc.get_shape().as_list()[-2]/stride_input

    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(laxer_idx+1), reuse=tf.get_variable_scope().reuse)

    bias = ops.bias([i_size, i_size, i_size, channels[1]], layer_name='bcnn' + str(laxer_idx+1),reuse=tf.get_variable_scope().reuse)

    conv_output_anc = tf.add(ops.conv3d(input_anc, weights, stride=[stride_input,stride_input, stride_input], padding=padding_type),bias)
    conv_output_pos = tf.add(ops.conv3d(input_pos, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type),bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = ops.relu(conv_output_anc)
    conv_output_pos = ops.relu(conv_output_pos)

    if dropout_flag:
        conv_output_anc = Dropout(conv_output_anc, keep_prob=dropout_rate)
        conv_output_pos = Dropout(conv_output_pos, keep_prob=dropout_rate)

    return conv_output_anc, conv_output_pos


def out_block(input_anc, input_pos, channels, laxer_idx, stride_input=1, k_size=8, padding_type='VALID'):

    # Last conv layer, flatten the output
    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(laxer_idx+1))

    bias = ops.bias([1, 1, 1, channels[1]], layer_name='bcnn' + str(laxer_idx + 1))

    conv_output_anc = tf.add(ops.conv3d(input_anc, weights, stride=[stride_input,stride_input, stride_input], padding=padding_type), bias)
    conv_output_pos = tf.add(ops.conv3d(input_pos, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type), bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = tf.contrib.layers.flatten(conv_output_anc)
    conv_output_pos = tf.contrib.layers.flatten(conv_output_pos)

    return conv_output_anc, conv_output_pos


class Model(ModelDesc):
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, [None, INPUT_DIM], name='input_x'),
                tf.placeholder(tf.float32, [None, INPUT_DIM], name='input_y')]

    def build_graph(self, x, y):
        edge = int(np.cbrt(INPUT_DIM))
        x = tf.reshape(x, [-1, edge, edge, edge, 1])
        y = tf.reshape(y, [-1, edge, edge, edge, 1])

        # Join the 3DSmoothNet structure with the desired output dimension
        net_structure = [1, 32, 32, 64, 64, 128, 128]
        outputDim = 32
        channels = [*net_structure, outputDim]

        # In the third layer stride is 2
        stride = np.ones(len(channels))
        stride[2] = 2

        # Apply dropout in the 6th layer
        dropout_flag = np.zeros(len(channels))
        dropout_flag[5] = 1

        layer_index = 0

        # Loop over the desired layers
        with tf.variable_scope('3DIM_cnn'):
            for layer in np.arange(0, len(channels)-2):
                scope_name = "3DIM_cnn" + str(layer_index+1)
                with tf.variable_scope(scope_name):
                    x, y = conv_block(x, y, [channels[layer], channels[layer + 1]],
                                                      dropout_flag[layer], 0.7, layer_index,
                                                      stride_input=stride[layer])

                layer_index += 1

            with tf.variable_scope('3DIM_cnn7'):
                x, y = out_block(x, y, [channels[-2], channels[-1]],
                                                 layer_index)

            output_x, output_y = ops.l2_normalize(x), ops.l2_normalize(y)

            dists_xy = tf.norm(tf.expand_dims(output_x, axis=1) - tf.expand_dims(output_y, axis=0), axis=-1)

            positive_mask = tf.eye(tf.shape(dists_xy)[0])
            negative_mask = tf.ones_like(positive_mask) - positive_mask
            positive_dists = tf.reduce_sum(dists_xy * positive_mask, axis=1)
            hardest_negative_dists = tf.reduce_min(dists_xy * negative_mask, axis=1)

        cost = tf.reduce_mean(tf.nn.softplus(positive_dists - hardest_negative_dists), name='cost')

        accuracy = tf.equal(tf.argmin(dists_xy, axis=1, output_type=tf.int32), tf.range(tf.shape(dists_xy)[0]))
        print(accuracy)
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32), name='accuracy')

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        # wd_cost = tf.multiply(1e-3,
        #                       regularize_cost('kernel.*', tf.nn.l2_loss),
        #                       name='regularize_loss')
        # total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, accuracy)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*_W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return cost

    def optimizer(self):
        lr = 1e-3
        # lr = tf.train.exponential_decay(
        #     learning_rate=1e-3,
        #     global_step=get_global_step_var(),
        #     decay_steps=468 * 10,
        #     decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        # tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


if __name__ == '__main__':
    with open('objects.pkl', 'rb') as f:
        objects, labels = pickle.load(f)
    print(sdv.compute(objects[0], interest_point_idxs=np.array([3])))