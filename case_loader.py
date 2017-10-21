import mnist_basics as MNIST
import tflowtools as TFT

class CaseLoader():
    def __init__(self):
        self.filepath = "C:/Users/agmal_000/Skole/AI prog/Oppgave 2/cases/"

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

    def get_fav_specs(self, case):
        if case == "mnist":
            return 2000, 0, [200,100,60,30], 0.005, 100
        if case == "wine":
            return 10000, 0, [22, 16, 11], 0.0005, 500
        if case == "parity":
            return 5000, 0, [16,8,8,4,4,2], 0.05, 20
        else:
            return []
