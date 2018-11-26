#coding=utf-8
# 神经网络的优化
# 简单神经网络实现自定义损失函数
# 加入学习率的设置（指数衰减）
# 加入L2正则化损失的实现
# 不包含隐层

import tensorflow as tf
from numpy.random import RandomState # 需要用到才会变色

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")   # 两个输入节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')  # 一个输出节点 （单变量回归）

# 前向传播
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1) # 矩阵乘法

# 定义损失函数使得预测少了的损失大
loss_less = 10
loss_more = 1

## 自定义分段损失函数
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
#                               (y - y_) * loss_more, (y_ - y) * loss_less))

# 自定义损失函数+L2正则化损失，0.5是lamada参数
loss = tf.reduce_sum(tf.where(tf.greater(y,y_), # 将tf.select替换为tf.where即可 (版本升级API修改了)
                    (y-y_)*loss_more,(y_-y)*loss_less))+tf.contrib.layers.l2_regularizer(0.5)(w1)

# 自定义损失函数+L1正则化损失，0.5是lamada参数
# loss = tf.reduce_sum(tf.select(tf.greater(y,y_),
#                     (y-y_)*loss_more,(y_-y)*loss_less))+tf.contrib.layers.l1_regularizer(0.5)(w1)

# 原始的优化方式
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 1. 学习率为1的时候，x在5和-5之间震荡。
# TRAINING_STEPS = 10
# LEARNING_RATE = 1

#### 2. 学习率为0.001的时候，下降速度过慢，在901轮时才收敛到0.823355。
# TRAINING_STEPS = 1000
# LEARNING_RATE = 0.001

#### 3. 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练步数下取得不错的收敛程度。
# TRAINING_STEPS = 100 # 较小的训练步数
global_step = tf.Variable(0) # 初始化变量为0 (Creates a new variable with value `initial_value`.)

# 学习率的设置：指数衰减法，参数：初始参数，全局步骤，每训练100轮乘以衰减速度0,96(当staircase=True的时候)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

# minimize函数中传入global_step参数 自动更新学习率
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)


# 3. 生成模拟数据集。
rdm = RandomState(1)
dataset_size=128
X = rdm.rand(dataset_size,2)
#加入了一个噪音值，-0.05～0.05之间
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

# 4. 训练模型。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print (sess.run(w1))

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # print ("After %d training step(s), w1 is: " % (i))
            # print (sess.run(w1), "\n")
            total_loss = sess.run(
                loss, feed_dict={x: X, y_: Y})
            print("After %d training_step(s) ,loss on all data is %g" % (i, total_loss))
    print ("Final w1 is: \n", sess.run(w1))
sess.close()  # 显式关闭会话， 关闭会话使得本次运行中使用到的资源可以被释放。
