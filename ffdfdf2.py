import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflowtools as TFT
import random as r
import os
import mnist_basics as MNIST

class CaseLoader():
    def __init__(self):
        self.filepath = "C:/Users/agmal_000/Skole/AI prog/Oppgave 2/mnist-zip/"

    def parity(self, bits):
        return TFT.gen_all_parity_cases(bits)

    def wine(self):
        #output: [[features], [1-hot]] for each row. 1-hot list of length 6, representing classes 3-8.
        with open (self.filepath + "winequality_red.txt") as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        wines = [0]*len(content)
        for i in range(len(content)):
            wines[i] = [[float(x) for x in content[i].split(";")]]
            wines[i].append([0]*6)
            c = int(wines[i][0].pop(-1))
            wines[i][1][c-3] = 1
        return wines

    def glass(self):
        #output: [[features], [1-hot] for each row. 1-hot list of length 6, representing classes 1-3 and 5-7.
        with open (self.filepath + "glass.txt") as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        glass = [0]*len(content)
        for i in range(len(content)):
            glass[i] = [[float(x) for x in content[i].split(",")]]
            glass[i].append([0]*6)
            c = int(glass[i][0].pop(-1))
            if (c > 4):
                c -= 1
            glass[i][1][c-1] = 1
        return glass

    def yeast(self):
        #output: [[features], [1-hot]] for each row. 1-hot list of length 10, representing classes 1-10.
        with open (self.filepath + "yeast.txt") as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        yeast = [0]*len(content)
        for i in range(len(content)):
            yeast[i] = [[float(x) for x in content[i].split(",")]]
            yeast[i].append([0]*10)
            c = int(yeast[i][0].pop(-1))
            yeast[i][1][c-1] = 1
        return yeast

    def phishing(self):
        #output: [[features], [1-hot]] for each row. 1-hot list of length 2, representing classes 1-2 where 1 = no phishing and 2 = phishing.
        with open (self.filepath + "phishing.txt") as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        phish = [0]*len(content)
        for i in range(len(content)):
            phish[i] = [[int(x) for x in content[i].split(",")]]
            phish[i].append([0]*2)
            c = phish[i][0].pop(-1)
            if (c == 1):
                phish[i][1][1]=1
            else:
                phish[i][1][0]=1
        return phish

    def mnist(self):
        data_set = MNIST.load_mnist()
        flat_set = MNIST.gen_flat_cases(cases = data_set)
        return_set = []
        for i in range(len(flat_set[0])):
            return_set.append([flat_set[0][i], TFT.int_to_one_hot(flat_set[1][i], 10)])
        return return_set

def compute_accuracy(prediction, s, v_xs, v_ys):
    y_pre = s.run(prediction, feed_dict={xs: v_xs})
##    print (v_xs)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
##    print (y_pre)
##    print (v_ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = s.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

#Set up training and test sets
def generate_test_sets(data):
    np.random.shuffle(data)
    test_set = data[:int(np.ceil(0.2*len(data)))]
    X_test, y_test = [], []
    for i in range(len(test_set)):
        X_test.append(test_set[i][0])
        y_test.append(test_set[i][1])
    return data[int(np.ceil(0.2*len(data))):], np.array(X_test), np.array(y_test)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    
def generate_training_sets(xs, ys):
    x_test = [0]*int(np.ceil(len(xs)*0.2))
    y_test = [0]*len(x_test)
    t = 0
    while (t<len(x_test)):
        ra = r.randint(0, len(x_test)-1)
        y_test[t] = ys[ra]
        x_test[t] = xs[ra]
        t += 1
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return xs, ys, x_test, y_test


def next_batch(data, size):
    batch_x, batch_y = [], []
    while len(batch_x) < size:
        ran = r.randint(0, len(data)-1)
        batch_x.append(data[ran][0])
        batch_y.append(data[ran][1])
    return np.array(batch_x), np.array(batch_y)

def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = weight_variable([in_size, out_size], "W")
            tf.summary.histogram(layer_name+"/weights", Weights)
        with tf.name_scope("biases"):
            biases = bias_variable([out_size], "b")
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if (activation_function is None):
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs", outputs)
        return outputs


c = CaseLoader()
case = c.mnist()
print(len(case))
train_set, x_test, y_test = generate_test_sets(case)

print("Train size: ", len(train_set))
print("Test size: ", len(x_test))

#CLASSIFICATION
#placeholder for input to network
x = tf.placeholder(tf.float32, shape=[None, len(x_test[0])])
y_ = tf.placeholder(tf.float32, shape=[None, len(y_test[0])])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(5000):
    batch_x, batch_y = next_batch(train_set, 50)
    if i % 10 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
      print("Step %04d" %i, " training accuracy = %g" %train_accuracy)
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
  print('test accuracy %g' % accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))

print (sess.run(am, feed_dict={X: X_test, Y: y_test, keep_prob: 1.0}))

print("Accuracy: {0}".format(sess.run(accuracy, feed_dict={X: X_test, Y: y_test, keep_prob: 1.0})))

sess.close()

