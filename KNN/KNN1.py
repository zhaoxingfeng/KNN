#coding=utf-8
"""
作者：zhaoxingfeng	日期：2016.12.17
功能：Python实现KNN分类算法，两个测试样本集：
    1、UCI公共库iris数据集
    2、手写字数据库digits
版本：V1.0
"""
from __future__ import division
import numpy as np
import operator

def KNN(testSet, trainSet, trainLabel, k):
    numSamples = np.shape(trainSet)[0]
    distanceRaw = np.tile(testSet, (numSamples, 1)) - trainSet
    # 每一行求和再平均 -> 欧式距离
    distance = np.sqrt(np.power(distanceRaw, 2).sum(axis=1))
    sortedIndex = distance.argsort()
    classCount = {}
    for i in xrange(k):
        label = trainLabel[sortedIndex[i]]
        classCount[label] = classCount.get(label, 0) + 1
    # 按照每一类数量多少进行排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def loadDataSet(filename):
    dataSet, dataLabel = [], []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataSet.append([float(data) for data in lineArr[:-1]])
            dataLabel.append(float(lineArr[-1]))
    return np.array(dataSet), np.array(dataLabel)

# 32 * 32矩阵数据转化为一维数据
def vector(filename):
    vector1024 = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            vector1024[0, 32 * i + j] = int(linestr[j])
    return vector1024

def loadfiles(dirName):
    from os import listdir
    fileList = listdir(dirName)
    labelSet = []
    dataSet = np.zeros((len(fileList), 1024))
    for i in range(len(fileList)):
        num = int(fileList[i].split('_')[0])
        labelSet.append(num)
        dataSet[i, :] = vector('%s\%s' % (dirName, fileList[i]))
    return np.array(dataSet), np.array(labelSet).T

if __name__ == "__main__":
    def iris():
        trainFile = r'iris.txt'
        trainSet, trainLabel = loadDataSet(trainFile)
        k = 5
        testSet = [[4.8, 3.4, 1.9, 0.2],[5.5, 2.4, 3.8, 1.1],[7.4, 2.8, 6.1, 1.9]]
        classNum = KNN(np.array(testSet)[1], trainSet, trainLabel, k)
        print("The predict label is : %d" % classNum)
    iris()
    def handwriting():
        trainSet, trainLabel = loadfiles(r"trainingDigits")
        testSet, testLabel = loadfiles(r"testDigits")
        numError = 0
        numTest = np.shape(testSet)[0]
        for i in range(numTest):
            classNum = KNN(testSet[i], trainSet, trainLabel, 5)
            print "The predict label i: %d, the real label is: %d" % (classNum, testLabel[i])
            if (classNum != testLabel[i]):
                numError += 1
        print("The total numError = %d" % numError)
        print("Final accuracy = %.3f%%" % ((numTest - numError) / numTest * 100))
    handwriting()