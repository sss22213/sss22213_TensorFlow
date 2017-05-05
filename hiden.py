import tensorflow as tf
import numpy as np

#define Layer
def add_layer(inpots,in_sizes,out_sizes,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_sizes,out_sizes]))
    biases=tf.Variable(tf.zeros([1,out_sizes])+0.1)
    wx_plus_b=tf.matmul(inpots,Weights)+biases
    if activation_function is None:
        output=wx_plus_b
    else
        output=activation_function(wx_plus_b)
    return output