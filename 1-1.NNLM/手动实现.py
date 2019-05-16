# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

sentences = [ "i like dog", "i love coffee", "i hate milk"]
word_list = ' '.join(sentences).split()
# print(word_list)
word_list = list(set(word_list)) # 去重功能
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)
# print(word_dict)
# print(num_dict)

# input_i = np.zeros(shape=[n_class])
# input_i[num_dict['i']] = 1
# input_like = np.zeros(shape=[n_class])
# input_like[num_dict['like']] = 1
# input_love = np.zeros(shape=[n_class])
# input_love[num_dict['love']] = 1
# input_hate = np.zeros(shape=[n_class])
# input_hate[num_dict['hate']] = 1
# input_batch = [
#     [input_i, input_like],
#     [input_i, input_love],
#     [input_i, input_hate],
# ]
#
# input_dog = np.zeros(shape=[n_class])
# input_dog[num_dict['dog']] = 1
# input_coffee = np.zeros(shape=[n_class])
# input_coffee[num_dict['coffee']] = 1
# input_milk = np.zeros(shape=[n_class])
# input_milk[num_dict['milk']] = 1
#
# target_batch = [
# input_dog, input_coffee, input_milk
# ]
def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch
# print('输入：', input_batch)
# print('输出：', target_batch)
def layers(x, n_hidden):
    w = tf.random_normal(shape=[n_step, n_hidden])
    b = tf.random_normal(shape=[n_hidden])
    return tf.matmul(x, w) + b

# #############################
# 模型超参数
# #############################
n_step = 2
n_hiddens = 16
epochs = 1000

# #############################
# 神经网络
# #############################
x = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_class], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, n_class], name='true')

input = tf.reshape(x, shape=[-1, n_step * n_class])
n_hiddens_1 = 2
n_hiddens_2 = 7
# tf.Variable(tf.random_normal(shape=[n_step * n_class, ]))
w1 = tf.Variable(tf.random_normal([n_step * n_class, n_hiddens_1]))
b1 = tf.Variable(tf.random_normal(shape=[n_hiddens_1]))
w2 = tf.Variable(tf.random_normal(shape=[n_hiddens_1, n_class]))
b2 = tf.Variable(tf.random_normal(shape=[n_class]))

tanh = tf.nn.tanh(b1 + tf.matmul(input, w1))  # [batch_size, n_hidden]
h2 = tf.matmul(tanh, w2) + b2
# #############################
# 预测
# #############################
y_pred = tf.argmax(h2, 1)
lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h2, labels=y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
opt = tf.train.AdamOptimizer(0.01).minimize(lost)


# #############################
# 开始训练
# #############################
input_batch, target_batch = make_batch(sentences)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        _, lost_ = sess.run([opt, lost], feed_dict={x:input_batch, y:target_batch})

        if (i + 1) % 10 == 0:
            print('epochs:{}, lost:{}'.format(i, lost))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pred = sess.run(y_pred, feed_dict={x:input_batch})

    # for i in range(len(input_batch)):
    #
    # print('{} {} ===> [{}/{}], P={}'.format(input_batch[i]))
    # for i in pred[0]:
    #     for j, k in enumerate(i):
    #         print('{} ===> {}, P={.3f}'.format(sentences[_], number_dict[j], i[j]))
    print(pred)