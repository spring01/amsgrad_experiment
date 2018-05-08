import numpy as np
import tensorflow as tf

C = 2.5

ph_c = tf.placeholder(tf.float32, [None])
x = tf.get_variable('x', dtype=tf.float32, shape=[1])
fx = ph_c * x

sgd = tf.train.GradientDescentOptimizer(1e-2)
adam = tf.train.AdamOptimizer(1e-2, beta1=0, beta2=(1 / (1 + C**2)))
rmsprop = tf.train.RMSPropOptimizer(1e-2, decay=(1 / (1 + C**2)))

for opt in sgd, adam, rmsprop:
    op = opt.minimize(fx)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for t in range(1000):
        c = C if t % 3 == 1 else -1.0
        sess.run(op, feed_dict={ph_c: [c]})
    print(sess.run(x))
