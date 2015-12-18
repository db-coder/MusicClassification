from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import matplotlib.pyplot as plt
import numpy as np
import gzip

row=50

f = open('file1.txt','r')
l=[line.strip() for line in f];
out = eval(l[0]);
tstoutput = eval(l[1]);
trnerr = eval(l[2]);
plt.plot(trnerr)
plt.ylabel("error")
plt.xlabel("epochs")
plt.show()

print(out, tstoutput)
x = np.linspace(1,row/5 ,row+1)
x.reshape(row+1,1)

plt.scatter(x, tstoutput,color='blue')
plt.scatter(x, out,color='red')
plt.legend(['target', 'output'])
plt.show()

