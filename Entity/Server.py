import random
import torch
import numpy as np


# 主机
class Server:
    def __init__(self, testData, tesetTarget, cnn, weights, learningRate, momentum):
        self.testData = testData
        self.tesetTarget = tesetTarget

        self.cnn = cnn
        self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=learningRate, momentum=momentum)

        # weights用于记录上轮的权重
        self.weights = weights

        # 准确率，用于记录上一轮的准确率进行对比，对比后确定要不要选择这组权重
        self.acc = 0

    # 改变测试集合标签
    def changeLabels(self):
        for i in range(len(self.tesetTarget)):
            if self.tesetTarget[i] %2 == 1:
                self.tesetTarget[i] -= 1

    #根据夏普利值更新参与方的选择
    def getWeights(self, TMCSVSet, federatedNum, epDegression, minEpsilon):
        # 获取此轮的epsilon贪心值
        epsilon = 1 - epDegression*federatedNum

        if epsilon < minEpsilon:
            epsilon = minEpsilon

        # 生成新的选择权重
        newWeights = np.zeros(len(TMCSVSet), dtype=int)
        for i in range(len(TMCSVSet)):
            sv = TMCSVSet[i]
            if sv > 0:
                newWeights[i] = 1
            else:
                newWeights[i] = 0

        # 生成一个随机数，如果随机数小于epsilon，则进行贪心选择
        greedyNum = random.random()

        # 进行贪心选择
        if greedyNum < epsilon:
            # 搜索权重为0的参与方
            lowDict = []
            for i in range(len(newWeights)):
                weight = newWeights[i]
                if weight == 0:
                    lowDict.append(i)

            # 计算更新几个
            updateNum = int(len(lowDict)*epsilon) + 1

            # 随机选取其中的数量进行改变
            if len(lowDict) > updateNum:
                updateArr = random.sample(lowDict, updateNum)
                for updateClient in updateArr:
                    newWeights[updateClient] = 1

        return newWeights

    # 进行聚合更新并重新分配cnn的参数
    def federated(self, clients):
        # 将主机的cnn参数清空
        for param in self.cnn.state_dict():
            self.cnn.state_dict()[param] *= 0

        # 聚合参数
        num = 0
        for i in range(len(self.weights)):
            cl = clients[i]
            if self.weights[i] == 1:
                num += 1
                for param in self.cnn.state_dict():
                    self.cnn.state_dict()[param] += cl.net.state_dict()[param]

        # 平均参数
        for param in self.cnn.state_dict():
            self.cnn.state_dict()[param] /= num

        # 重新分配
        for cl in clients:
            for param in self.cnn.state_dict():
                cl.net.state_dict()[param] *= 0
                cl.net.state_dict()[param] += self.cnn.state_dict()[param]

    # 生成参与方选择组合
    def getClientSelection(self, weights):
        clientSelection = []
        for i in range(len(weights)):
            weight = weights[i]
            if weight == 1:
                clientSelection.append(i)

        return clientSelection