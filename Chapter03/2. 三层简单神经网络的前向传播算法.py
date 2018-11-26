
# coding=utf-8
import tensorflow as tf
# 先定义变量和常量
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1)) # 均值为0 标准差为1 随机种子
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]]) # 这是个1×2的向量 （1个2维的样本）

a = tf.matmul(x, w1) # Multiplies matrix `a` by matrix `b`, producing `a` * `b`
y = tf.matmul(a, w2)

sess = tf.Session()
# 逐个变量初始化
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()

# 输入可以使用常量，但是数据量太大的时候，这样会生成大量的计算图
# x = tf.constant([[0.7,0.9]])
# placeholder 机制定义输入数据的位置 数据在程序运行时给定
# 这里维度不一定要定义，但是如果维度是确定的，可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 初始化所有定义的变量
# tf.global_variables_initializer().run() # 默认的初始化？
init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7,0.9]]})) # feed_dict 指定x的取值


x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
## 一起初始化
# 使用tf.global_variables_initializer()来初始化所有的变量
init_op = tf.global_variables_initializer()
sess.run(init_op)
# 分步初始化
# sess.run(w1.initializer)
# sess.run(w2.initializer)
# print (tf.assign(w2,w1,validate_shape=False)) # 将w1的值赋予w2

#print sess.run(y) #当使用placeholder时，因为没有传入输入，因此会报错
print (sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})) # feed_dict是个字典 用来指定x的值
sess.close()
