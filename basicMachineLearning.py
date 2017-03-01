"""
This is a basic machine learning example. The key part is the concept of "gradient descent"/

* define a model : y = Wx + b
* define error function (cost function)
* computing of minimizing the error function (cost function)
* iterately run
"""


import tensorflow as tf
import numpy as np

# input 
x = tf.placeholder("float")
y = tf.placeholder("float")

# w will be computed/learned.
w = tf.Variable([1.0])
b = tf.Variable([2.0])

# model of y = W*x + b
y_model = W*x + b        # or this way: y_model = tf.multiply(x, W) + b
"""
Here "x" and "w" are just numbers (scalar), they can also be n-dimension matrix. make sure w,x and b are dimension-consistent.
y = w_1*x_1 + x_2*x_2 + x_3*x+3 ... + w_n*x_n +b_0 +..+b_n
"""

# error function
error = tf.square(y - y_model)

# The Gradient Descent Optimizer: computing of minimizing of error function
learning_rate = 0.1  # note the parameter "0.1" is customizable, you can change it to see what happens.
train_op = tf.train.GradientDescentOptimizer(learning_rat).minimize(error)
# other gradient optimizers are here:
# https://www.tensorflow.org/api_guides/python/train#Optimizers
"""
tf.train.Optimizer
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.AdagradDAOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradOptimizer
tf.train.RMSPropOptimizer
"""

with tf.Session() as session:
    # initialize variables
    tf.global_variables_initializer().run() 
    for i in range(10):
        # the input dataset has a equation: y=2x+6. so the leared w and b should be 2 and 6, respectively.
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        session.run(train_op, feed_dict={x: x_value, y: y_value})
        print(session.run(w))       

    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

    
"""
THis is actually a linear regression.
The model is : 
y = wx+b
of which, "w" and "b" are learned/computed by the algorithm itself. 

Paramters to the algothrim: 
* learning rate "0.1"
* gradient optimizer
* iteration number

"""
