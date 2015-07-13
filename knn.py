import csv
import random
import heapq
from collections import Counter
dataSetFile="./data_set/iris.dat"
commentLine = 9
featureDimension = 4
ifNormailze = False
K = 3

'''
dataSetFile="./data_set/phoneme.dat"
commentLine = 10
featureDimension = 5
ifNormailze = False
K = 3

dataSetFile="./data_set/winequality-red.dat"
commentLine = 16
featureDimension = 11
ifNormailze = True 
K = 15
'''
def normailze(data):
    if (not ifNormailze):
        return data
    minVals = [1000000000.0] * featureDimension 
    maxVals = [-1000000000.0] * featureDimension 
    for x in data:
        for i in range(len(x[0])):
            minVals[i] = min(minVals[i], x[0][i])
            maxVals[i] = max(maxVals[i], x[0][i])
    rangeVals = [ x - y for x, y in zip(maxVals, minVals)]
    for x in data:
        for i in range(len(x[0])):
            x[0][i] = (x[0][i] - minVals[i])/rangeVals[i]
    return data
     

def genData(data_vec):
    data = [[list(map(float, x[:featureDimension])),x[featureDimension]] for x in data_vec]
    return normailze(data)

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
    trainDat= genData(vec[0: l])
    testDat= genData(vec[l: len(vec)])
    return trainDat, testDat

def distance(x, y):
    ans = 0
    assert len(x) == len(y)
    for i in range(len(x)):
        ans += (x[i] - y[i]) ** 2
    return ans

def predict(data, trainDat):
    res = heapq.nsmallest(K, trainDat, lambda x: distance(x[0], data[0]))
    res = Counter([x[1] for x in res]).most_common(1)[0][0]
    return(not res==data[1])


if __name__ == '__main__' :
    trainDat, testDat = loadData(dataSetFile)
    error, total = 0, len(testDat)
    for data in testDat:
        error += predict(data, trainDat)
    print(error,'/', total)
    print('Error Rate', error/ total)


