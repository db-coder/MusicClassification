from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import matplotlib.pyplot as plt
import numpy as np
import gzip

input_col=52
output_col=56
row=100
output=output_col-input_col
trnerr=[]
tstoutput=[]

df = ClassificationDataSet(input_col,output,class_labels=['pop','classical','metal','blues'])
file1 = open('inp_pcmb.txt','r')

for line in file1.readlines():
	data = [float(x) for x in line.strip().split(',') if x != '']
	indata = tuple(data[:input_col])
	outdata = tuple(data[input_col:output_col])
	df.addSample(indata,outdata)
trndata, tstdata = df.splitWithProportion( 0.75 )
n = buildNetwork(df.indim,20,df.outdim,outclass=SoftmaxLayer )
t = BackpropTrainer(n,dataset=trndata,momentum=0.1,verbose=True,weightdecay=0.01)
for x in range(500):
	err=t.train()
	trnerr.append(err)
out=t.testOnClassData(dataset=tstdata)
test_inp, test_out = zip(*tstdata)
for i in range(len(test_out)):
	m=0
	if np.ndarray.flatten(test_out[i])[0]==1 :
		m=0
	elif np.ndarray.flatten(test_out[i])[1]==1 :
		m=1
	elif np.ndarray.flatten(test_out[i])[2]==1 :
		m=2
	elif np.ndarray.flatten(test_out[i])[3]==1 :
		m=3
	tstoutput.append(m)

specificity,sensitivity=[],[]
for j in range(0,output):
    fn,fp,tp,tn=0,0,0,0
    for i in range(0,len(out)):
        fn+=(tstoutput[i]!=j and out[i]==j)
        fp+=(tstoutput[i]==j and out[i]!=j)
        tp+=(tstoutput[i]==j and out[i]==j)
        tn+=(tstoutput[i]!=j and out[i]!=j)
    sensitivity.append(tp/(tp+fn))
    specificity.append(tn/(tn+fp))

wrong=0
for y in range(100):
	if(out[y]==0):
		if((np.ndarray.flatten(test_out[y]))[0]!=1):
			wrong=wrong+1
	elif(out[y]==1):
		if((np.ndarray.flatten(test_out[y]))[1]!=1):
			wrong=wrong+1
	elif(out[y]==2):
		if((np.ndarray.flatten(test_out[y]))[2]!=1):
			wrong=wrong+1
	elif(out[y]==3):
		if((np.ndarray.flatten(test_out[y]))[3]!=1):
			wrong=wrong+1
print(wrong)

print("training error ",err)
print("specificity ",specificity)
print("sensitivity ",sensitivity)

plt.plot(trnerr)
plt.ylabel("error")
plt.xlabel("epochs")
plt.show()

x = np.linspace(1,row/5 ,row)
x.reshape(row,1)

plt.scatter(x, tstoutput,color='blue')
plt.scatter(x, out,color='red')
plt.legend(['target', 'output'])
plt.show()