from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import matplotlib.pyplot as plt
import numpy as np
import gzip

input_col=52
output_col=54
row=50
output=output_col-input_col
trnerr=[]
tstoutput=[]

df = ClassificationDataSet(input_col,output,class_labels=['rock','classical'])
file1 = open('inp_rock_classical.txt','r')

for line in file1.readlines():
	data = [float(x) for x in line.strip().split(',') if x != '']
	indata = tuple(data[:input_col])
	outdata = tuple(data[input_col:output_col])
	#print(indata, outdata)
	df.addSample(indata,outdata)
trndata, tstdata = df.splitWithProportion( 0.75 )
n = buildNetwork(df.indim,20,df.outdim,outclass=SoftmaxLayer )
t = BackpropTrainer(n,dataset=trndata,momentum=0.1,verbose=False,weightdecay=0.01)
for y in range(500):
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
	# elif np.ndarray.flatten(test_out[i])[2]==1 :
	# 	m=2
	tstoutput.append(m)


#print(len(out), "what ", len(tstoutput))

sensitivity,specificty=[],[]
for j in range(0,output):
    fn,fp,tp,tn=0,0,0,0
    for i in range(0,len(out)):
        fn+=(tstoutput[i]!=j and out[i]==j)
        fp+=(tstoutput[i]==j and out[i]!=j)
        tp+=(tstoutput[i]==j and out[i]==j)
        tn+=(tstoutput[i]!=j and out[i]!=j)
    sensitivity.append(tp/(tp+fn))
    specificty.append(tn/(tn+fp))

wrong=0
for y in range(50):
	if(out[y]==0):
		if((np.ndarray.flatten(test_out[y]))[0]!=1):
			wrong=wrong+1
	elif(out[y]==1):
		if((np.ndarray.flatten(test_out[y]))[1]!=1):
			wrong=wrong+1
	# elif(out[y]==2):
	# 	if((np.ndarray.flatten(test_out[y]))[2]!=1):
	# 		wrong=wrong+1
print(wrong),

print(' '),
#print(out)
#print(tstoutput)
print(err),
print(' '),
print(sensitivity),
print(' '),
print(specificty),
print(' '),
print(out),
print(' '),
print(tstoutput),
print(' '),
print(trnerr),

#plt.plot(trnerr)
#plt.ylabel("error")
#plt.xlabel("epochs")
#plt.show()

#x = np.linspace(1,row/5 ,row+1)
#x.reshape(row+1,1)

#plt.scatter(x, tstoutput,color='blue')
#plt.scatter(x, out,color='red')
#plt.legend(['target', 'output'])
#plt.show()
