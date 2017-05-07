import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#define Layer
def add_layer(inpots,in_sizes,out_sizes,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_sizes,out_sizes]))
    biases=tf.Variable(tf.zeros([1,out_sizes])+0.1)
    wx_plus_b=tf.matmul(inpots,Weights)+biases
    if activation_function is None:
        output=wx_plus_b
    else:
        output=activation_function(wx_plus_b)
    return output
#sample and answer variable
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#construct netual network
#hidden layer
l1=add_layer(xs,1,100,activation_function=tf.nn.relu)
#output layer
prediction=add_layer(l1,100,1,activation_function=None)

#gernel data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
#
y_data=np.square(x_data)-0.05+noise
#loss function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#plot

fig = plt.figure()
plt.axis([0, 30000, 0, 0.005])
plt.ion()
plt.show()

#
for i in range(30000):
   
     # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
     # x_data:[300,1]   y_data:[300,1]
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
    # 要取出預測的數值 必須再run 一次才能取出
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    if i % 100 ==0:
        print("loss",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        plt.scatter(i,sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        plt.pause(0.1)
    
'''
    if i % 50 == 0:
        # 畫出下一條線之前 必須把前一條線刪除掉 不然會看不出學習成果
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        
        # 要取出預測的數值 必須再run 一次才能取出
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # 每隔0.1 秒畫出來
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
'''