# coding=utf-8
# http://www.jianshu.com/p/3e5ddc44aa56
# tensorflow 1.3.1
# python 3.6
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy
from PIL import Image


def num2label(labelnum):
    lens = len(labelnum)
    label = np.zeros([lens, 42])
    for i in range(lens):
        label[i, int(labelnum[i]) + 40] = 1
    return label


def load_our_data():
    testdata = np.load('testdata.npy')
    testlabel = np.load('testlabel.npy')

    testlabel = num2label(testlabel)

    testdata = np.reshape(testdata / 255., (-1, 57 * 47))

    return testdata.astype('float32'), testlabel



def convolutional_layer(data, kernel_size, bias_size, pooling_size):
    kernel = tf.get_variable("selfconv", kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('selfbias', bias_size, initializer=tf.random_normal_initializer())

    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')
    linear_output = tf.nn.relu(tf.add(conv, bias))
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
    return pooling

def linear_layer(data, weights_size, biases_size):
    weights = tf.get_variable("selfweigths", weights_size, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("selfbiases", biases_size, initializer=tf.random_normal_initializer())
    return tf.add(tf.matmul(data, weights), biases)

def convolutional_neural_network(data):
    # 根据类别个数定义最后输出层的神经元
    n_ouput_layer = 42

    kernel_shape1=[5, 5, 1, 32]
    kernel_shape2=[5, 5, 32, 64]
    full_conn_w_shape = [15 * 12 * 64, 1024]
    out_w_shape = [1024, n_ouput_layer]

    bias_shape1=[32]
    bias_shape2=[64]
    full_conn_b_shape = [1024]
    out_b_shape = [n_ouput_layer]

    data = tf.reshape(data, [-1, 57, 47, 1])

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch, 29, 24, 32]
    with tf.variable_scope("selfconv_layer1", reuse=tf.AUTO_REUSE) as layer1:
        layer1_output = convolutional_layer(
            data=data,
            kernel_size=kernel_shape1,
            bias_size=bias_shape1,
            pooling_size=[1, 2, 2, 1]
        )
    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("selfconv_layer2", reuse=tf.AUTO_REUSE) as layer2:
        layer2_output = convolutional_layer(
            data=layer1_output,
            kernel_size=kernel_shape2,
            bias_size=bias_shape2,
            pooling_size=[1, 2, 2, 1]
        )
    with tf.variable_scope("selffull_connection", reuse=tf.AUTO_REUSE) as full_layer3:
        # 讲卷积层张量数据拉成2-D张量只有有一列的列向量
        layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
        layer3_output = tf.nn.relu(
            linear_layer(
                data=layer2_output_flatten,
                weights_size=full_conn_w_shape,
                biases_size=full_conn_b_shape
            )
        )
        # layer3_output = tf.nn.dropout(layer3_output, 0.8)
    with tf.variable_scope("selfoutput", reuse=tf.AUTO_REUSE) as output_layer4:
        output = linear_layer(
            data=layer3_output,
            weights_size=out_w_shape,
            biases_size=out_b_shape
        )

    return output;


def test_facedata(testdata, model_dir, model_path):
    X = tf.placeholder(tf.float32, [None, 57 * 47])
    # Y = tf.placeholder(tf.float32, [None, 42])

    predict = convolutional_neural_network(X)

    # 用于保存训练的最佳模型
    saver = tf.train.Saver()
    # model_dir = './model_of_recognition'
    # model_path = model_dir + '/best.ckpt'
    with tf.Session() as session:
        # 若不存在模型数据，需要训练模型参数

        # 恢复数据并校验和测试
        saver.restore(session, model_path)

        test_pred = tf.argmax(predict, 1).eval({X: testdata})

    return test_pred


def main():
    testdata, testlabel = load_our_data()
    model_dir = './model_of_recognition'
    model_path = model_dir + '/best.ckpt'

    print(test_facedata([testdata[0]], model_dir, model_path))


if __name__ == "__main__":
    main()
