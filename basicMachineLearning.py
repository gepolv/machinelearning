"""This is a basic machine learning example. The key part is the concept of "gradient descent"/

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

# error function
error = tf.square(y - y_model)

# The Gradient Descent Optimizer: computing of minimizing of error function
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)


with tf.Session() as session:
    # initialize variables
    tf.global_variables_initializer().run() 
    for i in range(10):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        session.run(train_op, feed_dict={x: x_value, y: y_value})
        print(session.run(w))       

    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
