from numpy import array,delete, s_
from sklearn import tree

def parsedata(file):
    list= []
    with open('C:/Users/bhakt/PycharmProjects/ml/' +file) as f:
        for line in f:
            list.append([int(x) for x in line.strip().strip('').split('\t')[1].split(",")])

    list = array(list);
    labels = list[:, 0]
    labels = labels.tolist()
    data = delete(list, s_[0:1], axis=1);
    data = data.tolist()
    # print(len(data))
    return data, labels


#read train data
trainData, trainLabels = parsedata('fullTrainData')

#read test data
testData, testLabels = parsedata('testData')

# create decision tree train data
clf = tree.DecisionTreeClassifier();

clf = clf.fit(trainData, trainLabels);
tree.export_graphviz(clf, 'tree.dot')
#predict
prediction = clf.predict(testData)

print(clf.predict_proba(testData))
print(prediction)