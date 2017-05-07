import tensorflow as tf
import numpy as np

#init Weight and bias
def weight_variable(shape):
    output=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(output)

def bias_variable(shape):
    output=tf.constant(0.1, shape=shape)
    return tf.Variable(output)
