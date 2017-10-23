import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import case_loader
import random as r

# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

class Gann():

    def __init__(self, dims, cman, activation_function, lrate=.1,showint=None,mbs=10,vint=None,softmax=False, keeps = 1.0):
        self.learning_rate = lrate
        self.activation_function = activation_function
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.keeps = keeps
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module): self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.keep_prob = tf.placeholder(tf.float64)
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i,outsize in enumerate(self.layer_sizes[1:len(self.layer_sizes)-1]):
            gmod = Gannmodule(self,i,invar,insize,outsize, self.activation_function, self.keeps)
            invar = gmod.output; insize = gmod.outsize
        gmod = Gannmodule(self,len(self.layer_sizes)-1,invar,insize,self.layer_sizes[-1], None, self.keeps)
        self.out_logits = gmod.out_logits
        self.output = gmod.output # Output of last module is output of whole network
        if self.softmax_outputs: self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.out_logits),name='RM')
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        # Defining the training operator
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    def do_training(self,sess,cases,epochs=100,continued=False):
        if not(continued): self.error_history = []
        val_cases = self.caseman.get_validation_cases()
        x_val = [c[0] for c in val_cases]; y_val = [c[1] for c in val_cases]
        for i in range(epochs):
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets, self.keep_prob: self.keeps}
                _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((step, error/nmb))
##            self.consider_validation_testing(step,sess)
            if (i%5 == 0):
                print ("Step %04d" %i, " accuracy = %g" %sess.run(self.accuracy, feed_dict={self.input: x_val, self.target: y_val, self.keep_prob: 1}))
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))
        
    def do_training_1batch(self, sess, cases, epochs = 100, continued = False):
        if not(continued): self.error_history = []
        val_cases = self.caseman.get_validation_cases()
        x_val = [c[0] for c in val_cases]; y_val = [c[1] for c in val_cases]
        for i in range(epochs+1):
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases)
            inputs, targets = self.next_batch(cases, mbs)
            feeder = {self.input: inputs, self.target: targets}
            _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
            error += grabvals[0]
            if (i%50 == 0):
                self.error_history.append((step, error/50))
                self.validation_testing(step,sess)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))

    def next_batch(self, data, size):
        batch_x, batch_y = [], []
        while len(batch_x) < size:
            ran = r.randint(0, len(data)-1)
            batch_x.append(data[ran][0])
            batch_y.append(data[ran][1])
        return np.array(batch_x), np.array(batch_y)

    def do_testing(self,sess,cases,msg='Testing'):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        error, grabvals, _ = self.run_one_step(self.error, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
##        print('%s Set Error = %f ' % (msg, error))
        if (msg == 'Final Testing'):
            print ("Final test accuracy = %g" %sess.run(self.accuracy, feed_dict={self.input: [c[0] for c in cases], self.target: [c[1] for c in cases], self.keep_prob: 1}))
        return error  # self.error uses MSE, so this is a per-case value


    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training_1batch(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing')

    def validation_testing(self,epoch,sess):
        cases = self.caseman.get_validation_cases()
        if len(cases) > 0:
            error = self.do_testing(sess,cases,msg='Validation Testing')
            self.validation_history.append((epoch,error))
            print ("Step %04d" %epoch, "validation accuracy = %g" %sess.run(self.accuracy, feed_dict={self.input: [c[0] for c in cases], self.target: [c[1] for c in cases], self.keep_prob: 1}))


    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training')

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1):
        names = [x.name for x in grabbed_vars];
##        msg = "Grabbed Variables at Step " + str(step)
##        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
##            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
##                print(v, end="\n\n")

    def run(self,epochs=100,sess=None,continued=False):
        PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session)
        self.testing_session(sess=self.current_session)
        self.close_current_session()
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=True)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize, activation_function, keep_prob):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.keep_prob = keep_prob
        self.name = "Module-"+str(self.index)
        self.build(activation_function)

    def build(self, activation_function):
        mona = self.name; n = self.outsize
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize,n)),
                                   name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        self.out_logits = tf.matmul(self.input,self.weights)+self.biases
        self.out_logits = tf.nn.dropout(self.out_logits, self.keep_prob)
        if activation_function is not None:
            self.output = activation_function(self.out_logits)
        else:
            self.output = self.out_logits
        self.ann.add_module(self)
        print ("Built module. Input size",self.insize," Output size",self.outsize, " Activation function", activation_function)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases


#   ****  MAIN functions ****

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(case, activation_function = tf.nn.relu, epochs=None, nbits=None, num = None, minbit = 0, maxbit = 8, layers = None, lrate=None,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False, keeps = 1.0):
    CL = case_loader.CaseLoader()
    epochs, nbits, num, layers, lrate, mbs = get_specs(CL, case, epochs, nbits, num, layers, lrate, mbs)
    
    switcher = {
        "parity": (lambda : CL.parity(nbits)),
        "wine": (lambda : CL.wine()),
        "glass": (lambda : CL.glass()),
        "yeast": (lambda : CL.yeast()),
        "phishing": (lambda : CL.phishing()),
        "mnist": (lambda : CL.mnist()),
        "bitcount": (lambda : TFT.gen_vector_count_cases(num, nbits)),
        "segmentcount": (lambda : TFT.gen_segmented_vector_cases(nbits, num, minbit, maxbit))
    }
    
    case_generator = switcher.get(case, "nothing")
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    x_len, y_len = len(cman.get_validation_cases()[0][0]), len(cman.get_validation_cases()[0][1])
    layers.insert(0, x_len)
    layers.append(y_len)
    ann = Gann(dims=layers,cman=cman, activation_function = activation_function, lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm, keeps=keeps)
##    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
##    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    #ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs)
    #ann.runmore(epochs*2)
    return ann

def get_specs(CL, case, epochs, nbits, num, layers, lrate, mbs):
    epochs_, nbits_, num_, layers_, lrate_, mbs_ = CL.get_fav_specs(case)
    if (case == "parity"):
        layers = []
        for i in range(4):
            layers.append(nbits)
        epochs = epochs_ + 2000*(nbits-10)
    elif (layers == None):
        layers = layers_
    if (epochs == None):
        epochs = epochs_
    if (nbits == None):
        nbits = nbits_
    if (num == None):
        num = num_
    if (lrate == None):
        lrate = lrate_
    if (mbs == None):
        mbs = mbs_
    return epochs, nbits, num, layers, lrate, mbs
    
