import random

import torch
import numpy as np
import torch.nn.functional as F


class Client:
    def __init__(self, trainData, trainTarget, net, learningRate, momentum):
        self.trainData = trainData
        self.trainTarget = trainTarget

        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learningRate, momentum=momentum)

        # 本通信回合本选择的情况
        self.roundTrainTarget = []
        self.roundDataDistribution = []

        # 测试集数据分布情况
        trainDataDistribution = np.zeros(len(set(list(self.trainTarget.numpy().tolist()))))
        for a in self.trainTarget:
            trainDataDistribution[int(a)] += 1

        self.trainDataDistribution = trainDataDistribution

    # 将训练集转置
    def dimInverse(self):
        for i in range(len(self.trainData)):
            datas = self.trainData[i][0]
            newDatas = torch.empty(datas.shape)
            dataLen = len(datas)
            for j in range(dataLen):
                finalPos = dataLen - j - 1

                data = datas[j]
                newData = data.numpy()[::-1]
                newDatas[finalPos] = torch.from_numpy(newData.copy())

            self.trainData[i][0] = newDatas

    # 打乱标签
    def randomTarget(self):
       random.shuffle(self.trainTarget)

    # 随机选择部分数据集进行训练
    def getDataRandom(self):
        dataNum = len(self.trainTarget)
        getNum = int(dataNum/10)

        roundTrainData = torch.zeros(getNum, 1, 28, 28)
        roundTrainTarget = torch.zeros(getNum)

        # 确保随机选择得到的数据包含所有标签
        while len(set(roundTrainTarget.numpy().tolist())) != len(self.trainDataDistribution):
            arr = random.sample(range(0, dataNum-1), getNum)

            for i in range(len(arr)):
                roundTrainData[i] = self.trainData[arr[i]]
                roundTrainTarget[i] = self.trainTarget[arr[i]]

        self.roundTrainTarget = roundTrainTarget

        return roundTrainData, roundTrainTarget

    # 获取本轮的数据分布情况
    def getRoundElementNum(self):
        elementDict = {}
        for a in self.roundTrainTarget:
            if int(a) not in elementDict:
                elementDict.update({int(a): 1})
            else:
                num = elementDict.get(int(a)) + 1
                elementDict.update({int(a): num})

        # print(elementDict)
        elementDict = sorted(elementDict.items(), key=lambda d: d[0], reverse=False)

        for i in range(len(elementDict)):
            elementArr = list(elementDict[i])
            elementDict[i] = elementArr

        return elementDict

    # 根据是否选中,从而调整数据的分布
    def changeTrainData(self, weight):
        elementDict = self.getRoundElementNum()

        # 如果被选中,则数据的分布情况不发生变化
        if weight == 1:
            self.roundDataDistribution = elementDict
        else:
            correctSet = self.getClientAcc()

            # 挑选最差的一个标签,减少max(30%, 1),然后随机将这个减少的值分配给另一个
            # 如果本来就只有1个,则随机换一个
            abateIndex = np.argmin(correctSet)

            if elementDict[abateIndex][1] <= 1 :
                randomIndex = -1
                while (randomIndex == abateIndex or randomIndex < 0) or elementDict[randomIndex][1] <= 1:
                    randomIndex = random.randint(0, len(elementDict) - 1)

                abateIndex = randomIndex

            changeNum = max(int(elementDict[abateIndex][1] * 0.3), 1)

            addIndex = np.argmax(correctSet)

            elementDict[abateIndex][1] -= changeNum
            elementDict[addIndex][1] += changeNum

            self.roundDataDistribution = elementDict

    # 根据数据分布情况选择数据
    def getDataRegular(self):
        getNum = int(len(self.trainTarget) / 10)

        dataIndex = []

        # 转为list
        targetList = self.trainTarget.numpy().tolist()

        # 获取所选数据的索引
        for i in range(len(self.roundDataDistribution)):
            key = self.roundDataDistribution[i][0]
            val = self.roundDataDistribution[i][1]

            keyIndexs = [index for (index, value) in enumerate(targetList) if value == key]
            arr = random.sample(keyIndexs, int(val))
            dataIndex.append(arr)

        roundTrainData = torch.empty(getNum, 1, 28, 28)
        roundTrainTarget = torch.empty(getNum)

        # 将上方选择的数据加入本轮中
        num = 0
        for arr in dataIndex:
            for a in arr:
                roundTrainData[num] = self.trainData[a]
                roundTrainTarget[num] = self.trainTarget[a]
                num += 1

        self.roundTrainTarget = roundTrainTarget

        return roundTrainData, roundTrainTarget


    # 计算各个标签的准确率
    def getClientAcc(self):
        elementSet = set(list(self.trainTarget.numpy().tolist()))
        correctSet = np.zeros(len(elementSet))

        with torch.no_grad():
            data = self.trainData.cuda()
            target = self.trainTarget.cuda()

            output = self.net(data)

            pred = output.data.max(1, keepdim=True)[1]
            for i in range(len(pred)):
                tar = target[i]
                if tar == pred[i][0]:
                    correctSet[tar] += 1

        for i in range(len(self.trainDataDistribution)):
            correctSet[i] /= self.trainDataDistribution[i]

        return correctSet