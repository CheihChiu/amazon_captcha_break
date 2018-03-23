# Builds the cnn network for running the network forward to make predictions.
# Author : CheihChiu
# Date   : 2017-06-06

import math

import tensorflow as tf

import numpy as np

# The capcha dataset has 18 classes, representing the charaters in ['a', 'b', 'c', 'e', 'f', 'g', 'h', 'j', 'k','l', 'm', 'n', 'p', 'r', 't', 'u', 'x', 'y']
NUM_CLASSES = 18

W_ALPHA = 0.01
B_ALPHA = 0.1
KEEP_PROB = 1.0

CONV_STRIDES = [1, 1, 1, 1]
POOLING_STRIDES = [1, 2, 2, 1]
POOLING_K_SIZE = [1, 2, 2, 1]
PADDING = 'SAME'
OUT_CHANNELS = [128, 256, 256, 512, 512]
FULL_CONNECT_CHANNELS = 1024
FILTER_WIDTH = 3


def logits(images, width, height):
    x = tf.reshape(images, shape=[-1, height, width, 1])

    # conv layer
    in_channel = 1
    input = x
    for out_channel in OUT_CHANNELS:
        w = tf.Variable(W_ALPHA * tf.random_normal([FILTER_WIDTH, FILTER_WIDTH, in_channel, out_channel]))
        b = tf.Variable(B_ALPHA * tf.random_normal([out_channel]))
        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=CONV_STRIDES, padding=PADDING), b))
        conv = tf.nn.max_pool(conv, ksize=POOLING_K_SIZE, strides=POOLING_STRIDES, padding='SAME')
        conv = tf.nn.dropout(conv, KEEP_PROB)
        input = conv
        # conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=CONV_STRIDES, padding=PADDING), b))
        # pool = tf.nn.max_pool(conv, ksize=POOLING_K_SIZE, strides=POOLING_STRIDES, padding='SAME')
        # dropout = tf.nn.dropout(pool, KEEP_PROB)
        # input = dropout
        in_channel = out_channel

    # Fully connected layer
    w_f = tf.Variable(W_ALPHA * tf.random_normal([FILTER_WIDTH * (FILTER_WIDTH - 1) * OUT_CHANNELS[len(OUT_CHANNELS) - 1]
                                                   ,FULL_CONNECT_CHANNELS]))
    b_f = tf.Variable(B_ALPHA * tf.random_normal([FULL_CONNECT_CHANNELS]))
    dense = tf.reshape(input, [-1, w_f.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_f), b_f))
    dense = tf.nn.dropout(dense, KEEP_PROB)

    w_out = tf.Variable(W_ALPHA * tf.random_normal([FULL_CONNECT_CHANNELS, NUM_CLASSES]))
    b_out = tf.Variable(B_ALPHA * tf.random_normal([NUM_CLASSES]))
    logits = tf.add(tf.matmul(dense, w_out), b_out)
    return logits