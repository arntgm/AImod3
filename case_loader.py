import mnist_basics as MNIST
import tflowtools as TFT

class CaseLoader():
    
    def __init__(self):
        self.filepath = "C:/Users/Bendik/Documents/GitHub/cases/"

    def parity(self, bits):
        self.cases = TFT.gen_all_parity_cases(bits)
        return self.cases

    def count(self, num, bits):
        self.cases = TFT.gen_vector_count_cases(num, bits)
        return self.cases

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
        self.cases = wines
        return self.cases

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
        self.cases = glass
        return self.cases

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
        self.cases = yeast
        return self.cases

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
        self.cases = phish
        return self.cases

    def mnist(self):
        data_set = MNIST.load_mnist()
        flat_set = MNIST.gen_flat_cases(cases = data_set)
        return_set = []
        for i in range(len(flat_set[0])):
            return_set.append([flat_set[0][i], TFT.int_to_one_hot(flat_set[1][i], 10)])
        self.cases = return_set
        return self.cases

    def get_case_size(self):
        return len(self.cases[0][0]), len(self.cases[0][1])

    def get_fav_specs(self, case):
        #GOOD
        if case == "mnist":
            return 2000, 0, 0, [200,100,60,30], 0.005, 100
        if case == "yeast": #use softplus
            return 15000, 0, 0, [40, 200, 40], 0.05, 300 #200, 100, 20
        if case == "wine": #use softplus
            return 10000, 0, 0, [400,200], 0.005, 100
        #GOOD, bruk tf.nn.tanh
        if case == "parity":
            #return 6000, 0, 0, [10,10,10], 0.005, 500
            return 6000, 0, 0, [40], 0.005, 500
        if case == "autoencoder":
            return 1500, 0, 0, [3], 0.005, 200
        #GOOD
        if case == "bitcount":
            return 3000, 16, 2**12, [32], 0.005, 100
        if case == "glass": #use elu
            return 14000, 0, 0, [20,10,10], 0.005, 100
        #GOOD
        if case == "phishing":
            return 1000, 0, 0, [30, 20,15,10,5], 0.005, 100 #5000 epochs
        #GOOD
        if case == "segmentcount":
            return 5000, 25, 2**12, [150,100,50,25,10], 0.005, 100
        else:
            return []

    def get_specs_from_file(self, filename):
        with open (self.filepath + filename+ ".txt") as f:
            content = f.readlines()
        content = [x.strip().split(":") for x in content]
        specs = []
        for c in content:
            specs.append(c[1].strip())
        return specs
