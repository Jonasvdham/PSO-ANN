# -*- coding: utf-8 -*-
import numpy as np

# Encode UCI data into numpy binary for faster access
data = np.loadtxt("data/uci_data.csv", delimiter=",", skiprows=1)

train_imgs = np.asfarray(data[:4215, :-1]) / 16
test_imgs = np.asfarray(data[4215:, :-1]) / 16
train_labels = np.asfarray(data[:4215, -1:])
test_labels = np.asfarray(data[4215:, -1:])
lr = np.arange(10)
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
for i, d in enumerate([train_imgs, test_imgs, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot]):
    np.save('%i.array' % i, d)

'''
load_data("UCI")  
run_test("UCI", i, 20, hidden_layers = [64, 32], Export=True)
if dataset=="UCI":
    inputs = [64]
elif dataset == "MNIST":
    inputs = [784]
    
shape = inputs + hidden_layers + [10]

ExportToFile = dataset+"blabla"
'''