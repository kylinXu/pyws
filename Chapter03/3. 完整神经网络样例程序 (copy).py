## coding=utf-8
## 简单神经网络实现二分类的问题

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# shape的第一个维度置为none可以定制batch内样本的个数
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input") # 二维输入数据
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input') # 一维输出标签

a = tf.matmul(x, w1)
y = tf.matmul(a, w2) # 这里y表示预测值

# tf.clip_by_value()可以将计算的数值限制在一个范围内（1e-10~1.0）
# y_表示真实值，y表示预测值，定义的是交叉熵损失函数
# 对于回归问题，最常用的损失函数是均方误差（MSE）mse = tf.reduce_mean(tf.square(y_-y))
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) # 定义交叉熵
# 多分类问题适合softmax+cross_entrpy
# cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(y,y_)

# 定义优化算法来优化网络中的参数 （需要学习率，和损失函数）
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) # 定义反向传播的优化方法

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 创建一个会话来执行程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print ("w1:", sess.run(w1))
    print ("w2:", sess.run(w2))
    print ("\n")

    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})# 每次输入一个batch的样本
        if i % 1000 == 0: # 每1000步打印一次训练误差
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    print ("\n")
    print ("w1:", sess.run(w1))
    print ("w2:", sess.run(w2))
