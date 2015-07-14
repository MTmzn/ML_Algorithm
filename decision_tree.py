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

class node:
    def __init__(self, x = -1):
        self.branch = {}
        self.decision = x 
    def setAxis(self,axis):
        self.axis = axis
    def addSon(self, key, childNode):
        self.branch[key] = childNode 

def genData(data_vec):
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
    trainDat= genData(vec[0: l])
    testDat= genData(vec[l: len(vec)])
    return trainDat, testDat

def calcEntropy(featureKey, subData):
    dataSize = len(subData)
    entropy = 0
    data_freq = Counter([x[1] for x in subData])
    for freq in data_freq.most_common():
        prob = freq[1] / dataSize
        entropy -= prob * math.log(prob, 2)
    return entropy

def splitDataSet(subData, axis):
    dictAxis = {}
    entropy = 0
    for x in subData:
        if (x[0][axis] in dictAxis): 
            dictAxis[x[0][axis]].append(x)
        else:
            dictAxis[x[0][axis]] = [x]
    for featureKey, data in dictAxis.items():
        entropy += len(data)/len(subData) * calcEntropy(featureKey, data) 
    return dictAxis, entropy

def chooseBestFeature(data, validAxis):
    bestEntropy = 1000000000
    bestAxis = -1
    for i in validAxis:
        axisDataSplited, axisEntropy = splitDataSet(data, i) 
        if axisEntropy < bestEntropy :
            bestAxis = i
            bestEntropy = axisEntropy
            bestSplit = axisDataSplited
    return bestAxis, bestSplit 

def buildTree(data, validAxis):
    classList = set(x[1] for x in data)
    if len(classList) == 1:
        return node(data[0][1])
    elif len(validAxis) == 0:
        return node(Counter([x[1] for x in data]).most_common(1)[0][0])
    currentNode = node()
    axisChoosen, branches = chooseBestFeature(data, validAxis)
    currentNode.setAxis(axisChoosen)
    validAxisRemain = validAxis[:]
    validAxisRemain.remove(axisChoosen)
    for branchFeature, branchData in branches.items():
        currentNode.addSon( branchFeature, buildTree(branchData, validAxisRemain) )
    return currentNode

def printTree(node, tabs) :
    for branchFeature, branchNode in node.branch.items():
        if (branchNode.decision == -1):
            print(' '*tabs, branchFeature)
            printTree(branchNode , tabs+2)
        else :
            print(' '*tabs, branchFeature,' ',branchNode.decision)

def getLabel(node, sample):
    if not node.decision == -1:
        return node.decision
    if not sample[node.axis] in node.branch:
        return classList[random.randint(0, len(classList)-1)]
    else:
        return getLabel(node.branch[sample[node.axis]], sample)

if __name__ == '__main__' :
    trainDat, testDat = loadData(dataSetFile)
    classList = list(set(x[1] for x in trainDat))
    root = buildTree(trainDat, list(range(0, featureDimension)))
    printTree(root, 0)
    print('Training Result')
    total = len(trainDat)
    error = 0
    for x in trainDat:
        labelPredict = getLabel(root, x[0])
        if not labelPredict == x[1]:
            error += 1
    print(error,'/', total)
    print(error/total)

    print('Testing Result')
    total = len(testDat)
    error = 0
    for x in testDat:
        labelPredict = getLabel(root, x[0])
        if not labelPredict == x[1]:
            error += 1
    print(error/ total)
