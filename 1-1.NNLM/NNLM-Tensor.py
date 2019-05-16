# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # number of steps ['i like', 'i love', 'i hate']

n_hiddens = [2, 16, 128, 1024] # number of hidden units

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
epochs = 2000
loss__ = []
for n_hidden in n_hiddens:
    # Model
    X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
    Y = tf.placeholder(tf.float32, [None, n_class])

    input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]
    H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
    d = tf.Variable(tf.random_normal([n_hidden]))
    U = tf.Variable(tf.random_normal([n_hidden, n_class]))
    b = tf.Variable(tf.random_normal([n_class]))

    tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
    model = tf.matmul(tanh, U) + b # [batch_size, n_class]

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    prediction =tf.argmax(model, 1)

    # Training

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        input_batch, target_batch = make_batch(sentences)

        loss_ = []

        for epoch in range(epochs):
            _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
            loss_.append(loss)
            # if (epoch + 1)%10 == 0:
            #     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss__.append(loss_)
# #######################
# 模型展示
# #######################
plt.figure()
t = np.arange(epochs)
colors = [] # 颜色
dm = len(n_hiddens)
labels = [u'神经元数量：{}'.format(i) for i in n_hiddens]
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))
for i in range(len(loss__)):

    plt.plot(t, loss__[i], color=colors[i], label=labels[i])
    plt.legend('upper right')
    plt.grid(False)
    plt.title('不同神经元数量下，损失值随迭代次数的变化关系')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')

plt.show()
# Predict
# predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
input = [sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])