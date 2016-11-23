#coding=utf-8
"""
作者：zhaoxingfeng	日期：2016.11.23
功能：Python实现KNN分类算法，样本来自UCI标准数据库Iris数据集
版本：V2.0
"""
from __future__ import division
import numpy as np
from sklearn import preprocessing


class KNN(object):
    # 读取数据
    def loadDataSet(self,filename):
        dataSet, dataLabel = [], []
        with open(filename) as fr:
            for line in fr.readlines():
                lineArr = line.strip().split(',')
                dataSet.append([float(data) for data in lineArr[:-1]])
                dataLabel.append(float(lineArr[-1]))
        return dataSet, dataLabel

    # 计算测试样本和训练样本的距离
    def evaDist(self,arrA,arrB):
        distance = np.sqrt(np.sum(np.power((arrA - arrB),2)))
        return distance

    # 主程序入口，对数据进行归一化预处理后再计算距离
    def run(self,filename,k,testSet):               # k 所选的最邻近数目
        dataRaw, label = self.loadDataSet(filename)
        numSamples = np.shape(dataRaw)[0]
        predict_label = []
        for i in range(np.shape(testSet)[0]):       # 对测试样本逐个进行分类
            dataRaw.append(testSet[i])
            data = preprocessing.MinMaxScaler().fit_transform(dataRaw)[:-1]   # 归一化
            distant = []                            # 保存测试样本和每一个训练样本的距离
            for j in xrange(numSamples):
                distant.append(self.evaDist(data[j][:],preprocessing.MinMaxScaler().fit_transform(dataRaw)[-1]))
            distantSortIndex = np.argsort(distant)  # 得到按距离排序后的列表索引值
            k_label = []
            for t in xrange(k):
                k_label.append(label[distantSortIndex[t]])
            mountDict = {}                          # 统计每一类包含的样本数量
            for j in xrange(k):
                mountDict[k_label[j]] = mountDict.get(k_label[j], 0) + 1
            mountMax = 0
            for a, b in mountDict.iteritems():
                if b > mountMax:
                    mountMax, mountMax_index = b, a
            predict_label.append(mountMax_index)
        return predict_label

if __name__ == "__main__":
    trainFile = r'iris.txt'   # 训练样本集
    testSet = [[5.1,3.7,1.5,0.4],[6.4,3.2,4.5,1.5],[6.3,2.7,4.9,1.8],[5.6,2.7,4.2,1.3],[6.9,3.2,5.7,2.3]]
    a = KNN()
    predict_label = a.run(trainFile, 20, testSet)
    print("The predict_label is :" + str(predict_label))