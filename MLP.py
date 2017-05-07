import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

def add_Layer(input_data,in_size,out_size,activation_function=None):
    weight=tf.Variable(tf.random_normal([in_size,out_size]))
    bias=tf.Variable(tf.zeros([out_size]))
    w_plus_b=tf.matmul(input_data,weight)+bias
    if activation_function==None:
        output=w_plus_b
    else:
        output=activation_function(w_plus_b)
    return output

#input
xs=tf.placeholder(tf.float32,[1,180])
ys=tf.placeholder(tf.float32,[1,20])

#construct netual network
input_layer=add_Layer(xs,180,180,activation_function=tf.sigmoid)
h1=add_Layer(input_layer,180,180,activation_function=tf.sigmoid)
h2=add_Layer(h1,180,180,activation_function=tf.sigmoid)
h3=add_Layer(h2,180,36,activation_function=tf.sigmoid)
output_layer=add_Layer(h3,36,20,activation_function=None)


#gernel data
x_data=np.linspace(-1,1,180)[np.newaxis,:]
y_data=np.linspace(-1,1,20)[np.newaxis,:]

#loss function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys - output_layer),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#plot
fig = plt.figure()
plt.axis([0, 3000, 0, 0.005])
plt.ion()
plt.show()

#
for i in range(3000):
   
     # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
     # x_data:[300,1]   y_data:[300,1]
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
    # 要取出預測的數值 必須再run 一次才能取出
        prediction_value = sess.run(output_layer, feed_dict={xs: x_data})
    if i % 100 ==0:
        print("loss",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        plt.scatter(i,sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        plt.pause(0.1)
    






