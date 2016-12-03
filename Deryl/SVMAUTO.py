from sklearn import svm
import numpy as numpy
from numpy import *
import os
from numpy import array,delete, s_



def parsedata(file):
    list= []
    with open("/Users/derylrodrigues/PycharmProjects/untitled2/"+file) as f:
        for line in f:
            list.append([int(x) for x in line.strip().strip('').split('\t')[1].split(",")])
    return list


Trdata = array(parsedata('part-m-00000'))
print (type(Trdata))
print (Trdata)
print (Trdata.shape)

print("newdata", Trdata.shape)
trainlabel = Trdata[:,0]
print("train label", trainlabel.shape)

trainlabel = array([trainlabel.tolist()]).transpose()
# print (trainlabel)
# print (trainlabel.shape)

traindata = delete(Trdata, s_[0:1], axis=1)

print(trainlabel.sort())


Tsdata = array(parsedata('part-test-00000'))
testlabel = Tsdata[:, 0]
testlabel = array([testlabel.tolist()]).transpose()

testdata = delete(Tsdata, s_[0:1], axis=1)

# print(testlabel)
def normalize(t):
    for i in range(t.shape[0]):
        if 33 <= t[i] and t[i] < 60:
            t[i]=0
        elif 60 <= t[i] and t[i] < 75:
            t[i]=1
        elif 75 <= t[i] and t[i] < 85:
            t[i]=2
        elif 85 <= t[i] and t[i] < 95:
            t[i]=3
    return t

testlabel = normalize(testlabel)
print(testlabel)


list1 = ['linear']

for i in list1:
    print()
    print ("Kernel  " , str(i))
    print()
    clf_rbf = svm.SVC(kernel=i)
    fitting = clf_rbf.fit(traindata, array(trainlabel.transpose().tolist()[0]))
    prediction =  clf_rbf.predict(testdata)
    print (type(prediction))
    print(prediction.shape)
    print(prediction)
    prediction = normalize(array(prediction))
    totalcount = len(prediction)
    error = 0
    actualpred = testlabel.transpose().tolist()[0]
    for i in range(len(prediction)):
        if prediction[i] != actualpred[i]:
            error += 1

    if error == 0:
        print("Accuracy is 100")
    else:
        print("Accuracy is ", str((1 - (error / totalcount)) * 100))

    print(fitting)
    print(prediction)





