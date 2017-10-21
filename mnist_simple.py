import case_loader
import tensorflow as tf
import numpy as np
import random as r

#Nåværende løsning: 5-lag [784, 200, 100, 60, 30, 10], learning rate 0.005, 5000 iterasjoner, batch size 100
#Gir accuracy på ~97%

#Set up training and test sets
def generate_test_sets(data):
    np.random.shuffle(data)
    test_set = data[:int(np.ceil(0.2*len(data)))]
    X_test, y_test = [], []
    for i in range(len(test_set)):
        X_test.append(test_set[i][0])
        y_test.append(test_set[i][1])
    return data[int(np.ceil(0.2*len(data))):], np.array(X_test), np.array(y_test)

#returns a random batch from data set
def next_batch(data, size):
    batch_x, batch_y = [], []
    while len(batch_x) < size:
        ran = r.randint(0, len(data)-1)
        batch_x.append(data[ran][0])
        batch_y.append(data[ran][1])
    return np.array(batch_x), np.array(batch_y)

#adds a new layer to the NN
def add_layer(inputs, input_size, output_size, activation_function = None):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev = 0.1))
    b = tf.Variable(tf.zeros([output_size]))
    output = tf.matmul(inputs, W) + b
    if activation_function is not None:
        output = activation_function(output)
        "With activation function", activation_function
    return output

def mnist(learning_rate = 0.005):
    case = case_loader.CaseLoader().mnist()

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])


    L1 = add_layer(x, 784, 200, activation_function = tf.nn.relu)
    L2 = add_layer(L1, 200, 100, activation_function = tf.nn.relu)
    L3 = add_layer(L2, 100, 60, activation_function = tf.nn.relu)
    L4 = add_layer(L3, 60, 30, activation_function = tf.nn.relu)
    y_logits = add_layer(L4, 30, 10)
    y = tf.nn.softmax(y_logits)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    train_data, x_test, y_test = generate_test_sets(case)
    print(x_test[0])
    print(y_test[0])
    print (train_data[0])
    for i in range(5000):
        batch_xs, batch_ys = next_batch(train_data, 100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if (i % 50 == 0):
            print ("Step %04d" %i, " accuracy = %g" %sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
    print ("Accuracy:",sess.run(accuracy, feed_dict = {x: x_test, y_: y_test}))
    sess.close()

mnist()
