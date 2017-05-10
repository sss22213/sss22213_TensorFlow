import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
#data
xs=tf.placeholder(tf.float32,[1,180])
ys=tf.placeholder(tf.float32,[1,20])
active_function=tf.nn.softmax
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
#network
def MLP_network(x, weights, biases):
    #Layer 1
    layer1=tf.add(tf.matmul(xs,weights['h1']),biases['b1'])
    layer1=active_function(layer1)
    #Layer 2
    layer2=tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2=active_function(layer2)
    #Layer 3
    layer3=tf.add(tf.matmul(layer2,weights['h3']),biases['b3'])
    layer3=active_function(layer3)
    #Layer 4
    layer4=tf.add(tf.matmul(layer3,weights['h4']),biases['b4'])
    layer4=active_function(layer4)
    #Layer 5
    layer5=tf.add(tf.matmul(layer4,weights['h5']),biases['b5'])
    layer5=active_function(layer5)
    #Layer 6
    layer6=tf.add(tf.matmul(layer5,weights['h6']),biases['b6'])
    layer6=active_function(layer6)
    #out layer
    out_layer=tf.add(tf.matmul(layer6,weights['out']),biases['out'])
    out_layer=active_function(out_layer)
    return out_layer
#argument
weights={
    'h1': tf.Variable(tf.random_normal([180,180])),
    'h2': tf.Variable(tf.random_normal([180, 180])),
    'h3': tf.Variable(tf.random_normal([180, 180])),
    'h4': tf.Variable(tf.random_normal([180, 180])),
    'h5': tf.Variable(tf.random_normal([180, 180])),
    'h6': tf.Variable(tf.random_normal([180, 180])),
    'out': tf.Variable(tf.random_normal([180, 20]))
}
biases={
    'b1': tf.Variable(tf.random_normal([180, 180])),
    'b2': tf.Variable(tf.random_normal([180, 180])),
    'b3': tf.Variable(tf.random_normal([180, 180])),
    'b4': tf.Variable(tf.random_normal([180, 180])),
    'b5': tf.Variable(tf.random_normal([180, 180])),
    'b6': tf.Variable(tf.random_normal([180, 180])),
    'out': tf.Variable(tf.random_normal([180, 20]))
}
pred=MLP_network(xs, weights, biases)
'''
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
'''
loss = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(pred,1e-10,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#plot
'''
fig = plt.figure()
plt.axis([0, 15000, 0, 4])
plt.ion()
plt.show()
'''
#
for j in range(499):
    x_data = genfromtxt('D:\\Deep_Learning\\Tensorflow_project\\sss22213_TensorFlow\\csv\\'+str(j+1)+'.csv', delimiter=",")[np.newaxis,:]
    y_data=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])[np.newaxis,:]
    print ("Num:"+str(j+1))
    for i in range(500):
        # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
        # x_data:[300,1]   y_data:[300,1]
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
        # 要取出預測的數值 必須再run 一次才能取出
            prediction_value = sess.run(pred, feed_dict={xs: x_data})
        if i % 100 ==0:
            print("loss",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            '''
            plt.scatter(i,sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            plt.pause(0.1)
            '''
    correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={xs:x_data ,ys: y_data}))
for j in range(499):
    x_data = genfromtxt('D:\\Deep_Learning\\Tensorflow_project\\sss22213_TensorFlow\\csv2\\'+str(j+1)+'.csv', delimiter=",")[np.newaxis,:]
    y_data=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])[np.newaxis,:]
    print ("Num:"+str(j+1))
    for i in range(500):
        # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
        # x_data:[300,1]   y_data:[300,1]
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
        # 要取出預測的數值 必須再run 一次才能取出
            prediction_value = sess.run(pred, feed_dict={xs: x_data})
        if i % 100 ==0:
            print("loss",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            '''
            plt.scatter(i,sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            plt.pause(0.1)
            '''
    correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={xs:x_data ,ys: y_data}))
    print (sess.run(weights))