import csv
import math
import random
from collections import Counter

dataSetFile="./data_set/nursery.dat"
commentLine = 13
featureDimension = 8
'''
dataSetFile="./data_set/tic-tac-toe.dat"
commentLine = 14
featureDimension = 9
'''

labelSet = set()
def stastic(trainDat):
    labelCount = Counter([x[1] for x in trainDat])
    Prob = []
    for x in trainDat:
        labelSet .add(x[1])
    for i in range(0, featureDimension):
        Prob.append({})
    for i in range(0, featureDimension):
        for x in trainDat:
            prob_delta = 1 / labelCount[x[1]]
            if (x[0][i], x[1]) in Prob[i]:
                Prob[i][(x[0][i], x[1])] +=  prob_delta
            else :
                Prob[i][(x[0][i], x[1])] = prob_delta
    return Prob, labelCount 

def formatData(data_vec):
    data = [(x[:featureDimension],x[featureDimension]) for x in data_vec]
    return data

def loadData(fileName):
    lineNumber = 0
    vec = []
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\n')
        for row in spamreader:
            lineNumber += 1
            if (lineNumber > commentLine):
                vec.append(row)
    random.shuffle(vec)
    l = len(vec) * 7 // 10
    trainDat= formatData(vec[0: l])
    testDat= formatData(vec[l: len(vec)])
    return trainDat, testDat

def predict(data, Prob, labelCount):
    bestAns = -1000
    labelPredict = ''
    for label in labelSet:
        probility = labelCount[label]
        for i in range(0, featureDimension):
            if (data[0][i], label) in Prob[i]:
                probility *= Prob[i][(data[0][i], label)]
            else:
                probility *= 0.5/labelCount[label]
        if (probility > bestAns):
            bestAns = probility
            labelPredict = label
    if (labelPredict == data[1]) :
        return 0
    else:
        return 1

if __name__ == '__main__' :
    trainDat, testDat = loadData(dataSetFile)
    Prob, labelCount = stastic(trainDat)

    print('Training Result')
    error, total = 0, len(trainDat)
    for data in trainDat:
        error += predict(data, Prob, labelCount)
    print(error,'/', total)
    print('Error Rate', error/ total)

    print('Testing Result')
    error, total = 0, len(testDat)
    for data in testDat:
        error += predict(data, Prob, labelCount)
    print(error,'/', total)
    print('Error Rate', error/ total)
