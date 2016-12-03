from numpy import array,delete, s_
from sklearn import tree
import sys
import pydotplus
import time
from IPython.display import Image

def parsedata(file):
    list= []
    with open('C:/Users/bhakt/PycharmProjects/ml/' +file) as f:
        for line in f:
            list.append([int(x) for x in line.strip().strip('').split('\t')[1].split(",")])
    return list


def dataandlabels(list):
    list = array(list);

    # labels is the first column
    last = list.shape[1] - 1
    labels = list[:, last]
    labels = labels.tolist()

    # data is remaining columns of list
    data = delete(list, s_[last:last + 1], axis=1);
    data = data.tolist()

    return data, labels

def split(seq, num):
    c = 0
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        chunk = seq[int(last):int(last + avg)]
        out.append(chunk)
        c += len(chunk)
        last += avg

    return out


def decisiontreeclassifier(X, y):
    # create decision tree train for the train sample
    clf = tree.DecisionTreeClassifier();
    clf = clf.fit(X, y);

    # uncomment code to trees as dot files
    filename = "tree" + str(int(round(time.time() * 1000)))
    tree.export_graphviz(clf, filename + '.dot')
    # (graph, ) = pydot.graph_from_dot_file(filename + '.dot')
    # graph.write_png(filename + '.png')
    #
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # time.sleep(1)

    return clf



def predict(clf, testData):
    # predict for test data given decision tree
    prediction = clf.predict(testData)
    return prediction

def pickbest(trees, predicitons, testLabels):
    minmisclassified = sys.maxsize;
    treeindex = -1;


    for i in range(len(trees)):
        pred = predicitons[i];

        # find mis-classified sample for the particular decision tree prediction
        missclassifications = 0
        for j in range(len(pred)):
            if(pred[j] != testLabels[j]):
                missclassifications += 1

        if missclassifications < minmisclassified:
            minmisclassified = missclassifications
            treeindex = i

    # print("Missclassifications", minmisclassified);
    # print("Index", treeindex)
    correctclassifications = len(testLabels) - minmisclassified
    accuracy =  correctclassifications / len(testLabels) * 100;
    return predicitons[treeindex], accuracy


def randomForest(n, traindatafile, testdatafile):

    # read trainng samples
    trainSamples = parsedata(traindatafile)
    # print(len(trainSamples))

    #split the data
    splits = split(trainSamples, n)

    #read test data
    testSamples = parsedata(testdatafile)
    testData, testLabels = dataandlabels(testSamples)

    trees = []
    predictions = []

    for chunk in splits:

        # train data
        trainData, trainLabels = dataandlabels(chunk);

        #create decision tree
        clf = decisiontreeclassifier(trainData, trainLabels)

        trees.append(clf)
        p = predict(clf, testData)

        # trees.
        predictions.append(p)

    # print(testLabels)
    # print(predictions)

    bestpred, accuracy = pickbest(trees, predictions, testLabels)

    print(bestpred)
    print(accuracy)

# random forest iterations, traindata, testdata
randomForest(1, 'filltrainData', 'testData')
