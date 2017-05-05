import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np

#data set
x=np.random.rand(100).astype(np.float32)
y_data=x*0.1+0.3

#neural argumnet
Weights=tf.Variable(tf.random_uniform([1],1,100))
biases=tf.Variable(tf.zeros([1]))

#forward neural
y=x*Weights+biases

#loss(loss function)
loss=tf.reduce_mean(tf.square(y-y_data))

#train(Gradient Descent; 0.5 is Learning rate or step size)
optimizer=tf.train.GradientDescentOptimizer(0.5)

#
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

####
sess=tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(Weights),sess.run(biases))


