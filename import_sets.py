import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflowtools as TFT
import random as r

class CaseLoader():
    def __init__(self):
        self.filepath = "C:/Users/agmal_000/Skole/AI prog/Oppgave 2/mnist-zip/"

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



def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = "W")
            tf.summary.histogram(layer_name+"/weights", Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name = "B")
            tf.summary.histogram(layer_name+"/biases", biases)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if (activation_function is None):
            outputs = Wx_plus_b
            print ("NEENIF")
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs", outputs)
        return outputs

def compute_accuracy(prediction, s, v_xs, v_ys):
##    global prediction
##    y_pre = s.run(prediction, feed_dict={xs: v_xs})
##    print (v_xs)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(v_ys, 1))
##    print (y_pre)
##    print (v_ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = s.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

c = CaseLoader()
wine_case = c.wine()
x = [0]*len(wine_case)
y = [0]*len(wine_case)
for i in range(len(wine_case)):
    x[i] = wine_case[i][0]
    y[i] = wine_case[i][1]
x_data = np.array(x)
y_data = np.array(y)

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
closed = [0]*len(x_data)
x_train, y_train, x_test, y_test = [],[],[],[]
while (sum(closed)<0.8*len(x_data)):
    i = r.randint(0,len(closed)-1)
    if (closed[i] == 0):
        x_train.append(x_data[i])
        y_train.append(y_data[i])
        closed[i] = 1
for i in range(len(closed)):
    if closed[i] == 0:
        x_test.append(x_data[i])
        y_test.append(y_data[i])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
print (len(x_test))

#CLASSIFICATION
#placeholder for input to network
xs = tf.placeholder(tf.float32, [None, 11], name = "features") #11 vurderingsakser
ys = tf.placeholder(tf.float32, [None, 6], name = "class") #faktisk klasse

#output
prediction = add_layer(xs, 11, 6, n_layer=1,
                            activation_function=tf.nn.softmax)

#error
cross_entropy = -tf.reduce_mean(ys*tf.log(prediction))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = TFT.gen_initialized_session()

batch_size = 100
number_of_batches = float(np.ceil((len(x_train)-1)/batch_size))
average_cost = 0
for i in range(int(number_of_batches)):
    batch_xs = x_train[i*batch_size : i*batch_size+batch_size]
    batch_ys = y_train[i*batch_size : i*batch_size+batch_size]
##    TFT.quickrun4([train_step],
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    average_cost +=sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys})/number_of_batches
print (compute_accuracy(prediction, sess, x_test, y_test))
print("Finished optimization")
sess.close()


