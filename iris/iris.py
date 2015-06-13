from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import *
import csv


ds = SupervisedDataSet(4, 3)
net = buildNetwork(4, 10, 10, 10, 3, bias=True, hiddenclass=TanhLayer)

flower_names = {
    'Iris-setosa': (1, 0, 0),
    'Iris-versicolor': (0, 1, 0),
    'Iris-virginica': (0, 0, 1)
}

with open('iris.csv', 'rb') as csvfile:
    irisdata = csv.reader(csvfile, delimiter=' ')
    for r in irisdata:
        t = r[0].split(",")
        ds.addSample(tuple(t[0:4]), flower_names[t[-1]])

trainer = BackpropTrainer(net, ds, learningrate=0.008)
e = trainer.train()
while e > 0.01:
    e = trainer.train()
    print e

v = net.activate((7.7,3.8,6.7,2.2,))
print [round(f) for f in v]
