#coding=utf-8
#http://www.jianshu.com/p/3e5ddc44aa56
#tensorflow 1.3.1
#python 3.6
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy
import pdb
import time
from PIL import Image




def num2label(labelnum):
    lens = len(labelnum)
    label = np.zeros([lens,42])
    for i in range(lens):
        label[i,int(labelnum[i])+40]=1
    return label


def load_our_data():
    traindata = np.load('traindata.npy')
    trainlabel = np.load('trainlabel.npy')
    testdata = np.load('testdata.npy')
    testlabel = np.load('testlabel.npy')
    valdata = testdata[0:15,:,:]
    vallabel = testlabel[0:15,]
    testdata = testdata[15:,:,:]
    testlabel = testlabel[15:]

    trainlabel = num2label(trainlabel)
    vallabel = num2label(vallabel)
    testlabel = num2label(testlabel)

    traindata = np.reshape(traindata/255.,(-1,57*47))
    testdata = np.reshape(testdata/255.,(-1,57*47))
    valdata = np.reshape(valdata/255.,(-1,57*47))
    return traindata.astype('float32'),trainlabel,valdata.astype('float32'),vallabel,testdata.astype('float32'),testlabel
# traindata,trainlabel,valdata,vallabel,testdata,testlabel = load_our_data()

# import pdb; pdb.set_trace()

#获取dataset
def load_data(dataset_path):
    img = Image.open(dataset_path)
    # 定义一个20 × 20的训练样本，一共有40个人，每个人都10张样本照片
    img_ndarray = np.asarray(img, dtype='float64') / 256
    #img_ndarray = np.asarray(img, dtype='float32') / 32

    # 记录脸数据矩阵，57 * 47为每张脸的像素矩阵
    faces = np.empty((400, 57 * 47))

    for row in range(20):
        for column in range(20):
            faces[20 * row + column] = np.ndarray.flatten(
                img_ndarray[row * 57: (row + 1) * 57, column * 47 : (column + 1) * 47]
            )

    label = np.zeros((400, 42))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1

    # 将数据分成训练集，验证集，测试集
    train_data = np.empty((320, 57 * 47))
    train_label = np.zeros((320, 42))
    vaild_data = np.empty((40, 57 * 47))
    vaild_label = np.zeros((40, 42))
    test_data = np.empty((40, 57 * 47))
    test_label = np.zeros((40, 42))

    for i in range(40):
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]

        vaild_data[i] = faces[i * 10 + 8]
        vaild_label[i] = label[i * 10 + 8]

        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    vaild_data = vaild_data.astype('float32')
    test_data = test_data.astype('float32')

    traindata,trainlabel,valdata,vallabel,testdata,testlabel = load_our_data()
    # import pdb; pdb.set_trace()
    train_data = np.concatenate((train_data,traindata))
    vaild_data = np.concatenate((vaild_data,valdata))
    test_data  = np.concatenate((test_data ,testdata))

    train_label = np.concatenate((train_label,trainlabel))
    vaild_label = np.concatenate((vaild_label,vallabel))
    test_label  = np.concatenate((test_label,testlabel))

    # import pdb; pdb.set_trace()
    #shuffle
    idx = np.arange(train_data.shape[0])
    np.random.shuffle(idx)
    train_data = train_data[idx,:]
    train_label = train_label[idx,:]

    idx = np.arange(test_data.shape[0])
    np.random.shuffle(idx)
    test_data = test_data[idx,:]
    test_label = test_label[idx,:]

    idx = np.arange(vaild_data.shape[0])
    np.random.shuffle(idx)
    vaild_data = vaild_data[idx,:]
    vaild_label = vaild_label[idx,:]

    # import pdb; pdb.set_trace()
    return [
        (train_data, train_label),
        (vaild_data, vaild_label),
        (test_data, test_label)
    ]

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
    with tf.variable_scope("selfconv_layer1") as layer1:
        layer1_output = convolutional_layer(
            data=data,
            kernel_size=kernel_shape1,
            bias_size=bias_shape1,
            pooling_size=[1, 2, 2, 1]
        )
    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("selfconv_layer2") as layer2:
        layer2_output = convolutional_layer(
            data=layer1_output,
            kernel_size=kernel_shape2,
            bias_size=bias_shape2,
            pooling_size=[1, 2, 2, 1]
        )
    with tf.variable_scope("selffull_connection") as full_layer3:
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
    with tf.variable_scope("selfoutput") as output_layer4:
        output = linear_layer(
            data=layer3_output,
            weights_size=out_w_shape,
            biases_size=out_b_shape
        )

    return output;

def train_facedata(dataset, model_dir,model_path):
    # train_set_x = data[0][0]
    # train_set_y = data[0][1]
    # valid_set_x = data[1][0]
    # valid_set_y = data[1][1]
    # test_set_x = data[2][0]
    # test_set_y = data[2][1]
    # X = tf.placeholder(tf.float32, shape=(None, None), name="x-input")  # 输入数据
    # Y = tf.placeholder(tf.float32, shape=(None, None), name='y-input')  # 输入标签

    batch_size = 40

    # train_set_x, train_set_y = dataset[0]
    # valid_set_x, valid_set_y = dataset[1]
    # test_set_x, test_set_y = dataset[2]
    train_set_x = dataset[0][0]
    train_set_y = dataset[0][1]
    valid_set_x = dataset[1][0]
    valid_set_y = dataset[1][1]
    test_set_x = dataset[2][0]
    test_set_y = dataset[2][1]

    X = tf.placeholder(tf.float32, [None, 57 * 47])
    Y = tf.placeholder(tf.float32, [None, 42])

    predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_func)

    # 用于保存训练的最佳模型
    saver = tf.train.Saver()
    #model_dir = './model'
    #model_path = model_dir + '/best.ckpt'
    with tf.Session() as session:
        # 若不存在模型数据，需要训练模型参数
        if not os.path.exists(model_path + ".index"):
            session.run(tf.global_variables_initializer())
            best_loss = float('Inf')
            for epoch in range(20):
                epoch_loss = 0
                for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                    x = train_set_x[i * batch_size: (i + 1) * batch_size]
                    y = train_set_y[i * batch_size: (i + 1) * batch_size]
                    _, cost = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                    epoch_loss += cost

                print(epoch, ' : ', epoch_loss)
                if best_loss > epoch_loss:
                    best_loss = epoch_loss
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                        print("create the directory: %s" % model_dir)
                    save_path = saver.save(session, model_path)
                    print("Model saved in file: %s" % save_path)

        # 恢复数据并校验和测试
        saver.restore(session, model_path)
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        a = time.time()
        print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))

        print('test set accuracy: ', valid_accuracy.eval({X: test_set_x, Y: test_set_y}))
        print (time.time()-a )
        test_pred = tf.argmax(predict, 1).eval({X: test_set_x})
        test_true = np.argmax(test_set_y, 1)
        # import pdb; pdb.set_trace()
        test_correct = correct.eval({X: test_set_x, Y: test_set_y})
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i'
                %(test_true[i], test_pred[i]))
        
        _,_,_,_,testdata,testlabel = load_our_data()

        test_pred = tf.argmax(predict, 1).eval({X: testdata})
        test_true = np.argmax(testlabel, 1)
        
        test_correct = correct.eval({X: testdata, Y: testlabel})
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i'
                %(test_true[i], test_pred[i]))

#画出在测试集中错误的数据
def plot_errordata(error_index, dataset_path):
    img = mpimg.imread(dataset_path)
    plt.imshow(img)
    currentAxis = plt.gca()
    for index in error_index:
        row = index // 2
        column = index % 2
        currentAxis.add_patch(
            patches.Rectangle(
                xy=(
                     47 * 9 if column == 0 else 47 * 19,
                     row * 57
                    ),
                width=47,
                height=57,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
    )
    plt.savefig("result.png")
    plt.show()


def main():
    dataset_path = "olivettifaces.gif"
    data = load_data(dataset_path)
    model_dir = './model'
    model_path = model_dir + '/best.ckpt'
    train_facedata(data, model_dir, model_path)

if __name__ == "__main__" :
    main()