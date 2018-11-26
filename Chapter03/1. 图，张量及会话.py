# coding=utf-8

## 计算模型---计算图
# tensorflow的程序分为两个阶段
# 1. 定义（图中的）计算
# 2. 执行计算

# 1. 定义计算
import tensorflow as tf
a=tf.constant([1.0, 2.0], name="a")  # define the constant
b=tf.constant([2.0, 3.0], name="b")
result = a+b
print(result)   # 这里输出一个张量 （名字、维度、类型）add：0

# 2. 执行计算 (通过会话session来执行计算)
print (tf.Session().run(result))  # run(result)并打印结果
tf.Session().close() # 显式关闭会话

## 管理图模型
# tf.Graph()生成计算图，不同图上的变量和计算不会共享
g1 = tf.Graph()
with g1.as_default():
    # 在图g1中定义变量v，并设置初始值为0
    v = tf.get_variable("v", [1], initializer = tf.zeros_initializer()) # 第二个参数为shape
g2 = tf.Graph()
with g2.as_default():
    # 在图g2中定义变量v，并设置初始值为1
    v = tf.get_variable("v", [1],initializer = tf.ones_initializer())
# 在计算图g1中读取变量v的取值
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v"))) # 输出[0.]
# 在计算图g2中读取变量v的取值
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v"))) # 输出[1.]

# tf.Graph()不仅可以用来管理图，还可以指定运行计算的GPU设备
g3=tf.Graph()
with g3.device('/gpu:0'): # 指定计算图运行设备
    result= a + b
# 2. 执行计算 (通过session来执行计算)
print (tf.Session().run(result))  # run(result)并打印结果
tf.Session().close() #显式关闭会话


## 数据模型--张量的使用
# 张量有三个属性: name, shape, type
# tf.constant: Creates a constant tensor
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b # still a Tensor class
# 打印张量的信息
print (result) # Tensor("add_3:0", shape=(2,), dtype=float32)
# result.get_shape


## 执行模型--会话
## 会话拥有程序运行时的所以资源，完成后需要关闭会话回收资源，避免资源泄露
## 两种使用会话的模式

## 1.显示调用和关闭会话
sess = tf.Session()
# tf需要显式指定默认的会话，并通过tf.Tensor.eval()计算张量的取值
with sess.as_default():  # 指定默认会话
    # 两个命令功能相同
    print(sess.run(result))
    print(result.eval(session=sess)) # 在会话中可以使用张量的eval函数查看结果
    # 对应元素相乘，矩阵乘法式matmul
    # print (a*b).eval() # 这行代码有错
sess.close() # 显式关闭会话， 关闭会话使得本次运行中使用到的资源可以被释放。

# InteractiveSession()自动将生成的会话注册为默认的会话
sess = tf.InteractiveSession ()     # 更加方便
print(result.eval())
sess.close()                        # 关闭会话


# 通过ConfigProto对会话进行配置、、、、533----
config = tf.ConfigProto(allow_soft_placement=True,  # GPU运算可以放到CPU （默认为0,一般设置为ture）
                        log_device_placement=False) # 记录每个节点被安排在哪个设备上。设置为false，减少日志量
# 两种设置方式
sess1 = tf.InteractiveSession(config=config)        # 默认的会话
sess2 = tf.Session(config=config)
sess1.close()
sess2.close()

# 一个完整的例子
# 声明一个2×3的矩阵变量，均值为0,标准差为2,另外还有random_gamma,truncated_normal,random_uniform
weights = tf.Variable(tf.random_normal([2,3],stddev=2,seed=1))

#除了使用随机数或者常量，tf还支持通过其他变量的值的初始化来初始化新的变量
w1 = tf.Variable(weights.initialized_value()) # 通过变量weights进行初始化
w2 = tf.Variable(weights.initialized_value()*2.0)

# 产生全为0的数组
a  = tf.Variable(tf.zeros([2,3],tf.int32))
# 产生全为1的数组
b = tf.Variable(tf.ones([2,3],tf.int32))
# 产生一个全部为给定数字的数组
c = tf.Variable(tf.fill([2,3],9))
# 产生一个给定值的常量
d = tf.Variable(tf.constant([2,3]))

# greater和select的应用
v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,2.0,2.0,5.0])

sess2 = tf.InteractiveSession()
print (tf.greater(v1,v2).eval())  # 逐元素判断v1是否大于v2 [False False  True False]
# print (tf.select(tf.greater(v1,v2),v1,v2).eval()) # module 'tensorflow' has no attribute 'select'
sess2.close()
