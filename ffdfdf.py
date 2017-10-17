import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflowtools as TFT
import random as r
import os

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

def compute_accuracy(prediction, s, v_xs, v_ys):
    y_pre = s.run(prediction, feed_dict={xs: v_xs})
##    print (v_xs)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
##    print (y_pre)
##    print (v_ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = s.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

#x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
#y_data = np.square(x_data) - 0.5 + noise

#Define placeholder for inputs to network
##        with tf.name_scope("inputs"):
##            xs = tf.placeholder(tf.float32, [None, 11], name = "features")
##            ys = tf.placeholder(tf.float32, [None, 6], name = "class")
##
##        #input layer
##        l1 = self.add_layer(xs, 11, 15,n_layer = 1, activation_function = tf.nn.relu)
##
##        #output layer
##        with tf.name_scope("prediction"):
##            prediction = self.add_layer(l1, 15, 6, n_layer = 2, activation_function = None)
##
##        #error
##        with tf.name_scope("loss"):
##            loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.square(ys-prediction),2), reduction_indices=[1]))
##        tf.summary.scalar("loss",loss)
##        
##        with tf.name_scope("train"):
##            train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
##        init = tf.global_variables_initializer()
##        sess = tf.Session()
##        merged = tf.summary.merge_all()
##        #Tensorboard
##        writer = TFT.viewprep(sess)
##                              
##        sess.run(init)
##
##        #fig = plt.figure()
##        #ax = fig.add_subplot(1,1,1)
##        #plt.ion()
##        #plt.show()
##
##        for i in range (5000):
##            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
##            if i % 50 == 0:
##                result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
##                writer.add_summary(result,i)
##                #try:
##                 #   ax.lines.remove(lines[0])
##                #except Exception:
##                 #   pass
##                #prediction_value = sess.run(loss, feed_dict={ys: y_data})
##                #lines = ax.plot(y_data, prediction_value, 'r', lw=3)
##                #plt.pause(0.5)
##                print (sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
##

#Set up training and test sets
def generate_test_sets(data):
    np.random.shuffle(data)
    test_set = data[:int(np.ceil(0.2*len(data)))]
    X_test, y_test = [], []
    for i in range(len(test_set)):
        X_test.append(test_set[i][0])
        y_test.append(test_set[i][1])
    return data[int(np.ceil(0.2*len(data))):], X_test, y_test

def weight_variable(shape, n):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = n)

def bias_variable(shape, n):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = n)
    
def generate_training_sets(data):
    np.random.shuffle(data)
    x = [0]*len(data)
    y = [0]*len(data)
    for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = data[i][1]
    X_train = np.array(x)
    y_train = np.array(y)
    return X_train, y_train

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
case = c.parity(10)
train_set, X_test, y_test = generate_test_sets(case)

input_size = len(X_test[0])
output_size = len(y_test[0])
batch_size = int(np.ceil(len(train_set)/10))
print ("Training set size: ",len(train_set))
print ("Test set size: ",len(X_test))

#CLASSIFICATION
#placeholder for input to network
learning_rate = 0.05
X = tf.placeholder("float", [None, input_size], name='X-input')
Y = tf.placeholder("float", [None, output_size], name='y-input')
keep_prob = tf.placeholder(tf.float32)

#Layer 1
L1 = add_layer(X, input_size, input_size//2, 1, activation_function = tf.nn.sigmoid)
pred = add_layer(L1, input_size//2, output_size, 2, activation_function = tf.nn.softmax)


###Layer 2
##W2 = tf.Variable(tf.zeros([6, 6]), name='Weights3')
##b2 = tf.Variable(tf.zeros([6]), name='Biases3')
##L2 = tf.nn.softmax(tf.matmul(L1, W2) + b2)

error = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=[1]))
tf.summary.scalar('loss', error)
tf.summary.histogram('Layer 1/outputs', pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
am = tf.argmax(pred,1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
merged = tf.summary.merge_all()
os.system('rm ' + "logs" +'/events.out.*')
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    X_train, y_train = generate_training_sets(train_set)
    number_of_batches = int(len(X_train) / batch_size)
    if i % 10 == 0:
        feed = {X: X_test, Y: y_test, keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict=feed)
        print("Accuracy at step %s: %s" % (i, acc))
        err = sess.run(error, feed_dict=feed)
        print("Error at step %s: %s" % (i, err))
    for start, end in zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)):
        feed = {X: X_train[start:end], Y: y_train[start:end], keep_prob: 0.5}
        sess.run(optimizer, feed_dict=feed)
        
    train_result = sess.run(merged, feed_dict={X: X_train, Y:y_train, keep_prob: 1.0})
    test_result = sess.run(merged, feed_dict={X:X_test, Y:y_test, keep_prob: 1.0})
    train_writer.add_summary(train_result, i)
    test_writer.add_summary(test_result, i)
print (sess.run(am, feed_dict={X: X_test, Y: y_test, keep_prob: 1.0}))

print("Accuracy: {0}".format(sess.run(accuracy, feed_dict={X: X_test, Y: y_test, keep_prob: 1.0})))

sess.close()

