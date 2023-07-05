import time
import tensorflow as tf

start_time = time.time()
x=tf.random.normal([5,5,500], 0, 1, tf.float32)
x=(x**2)*10
y=tf.math.tan(x)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)