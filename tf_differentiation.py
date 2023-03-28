import tensorflow as tf
import numpy as np
# print(tf.__version__)

#一次偏微, 二次偏微
c = tf.constant(1)
x = tf.Variable(3.0)
# x = tf.constant(3.0)
with tf.GradientTape() as g2:
	# g2.watch(x)
	with tf.GradientTape() as g:
		# g.watch(x)
		y = 2 * x * x 
	dx = g.gradient(y,x)
d2x = g2.gradient(dx,x)

# print(dx.numpy())
# print(d2x.numpy())

#多變數偏微

x = tf.Variable(2.0)
y = tf.Variable(2.0)
with tf.GradientTape(persistent = True) as g:
	z = y * y + x * x * x

dz_dx = g.gradient(z, x)
dz_dy = g.gradient(z, y)


print(dz_dx.numpy())
print(dz_dy.numpy())