import numpy
import random

import numpy as np
import torch
import torchvision
from Entity import Server, Client, CNN
import torch.nn.functional as F
import math

# 训练集
datasetTrain = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("", train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=64)

# 测试集
datasetTest = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=1000, shuffle=True)

# 确定参与方数量
clientNum = 100
badClientNum = int(clientNum/2)

# 优化器参数
learningRate = 0.01
momentum = 0.5
localTrainTimes = 10

# 生成主机
# 卷积神经网络
serverNet = CNN.Net()
serverNet = serverNet.cuda()

testData = 0
testTarget = 0
for data, target in datasetTest:
    testTarget = target
    testData = data

    if len(testData) != 0:
        break

weights = np.zeros(clientNum)
server = Server.Server(testData, testTarget, serverNet, weights, learningRate, momentum)

# 参与方组
clients = []
num = 0
for data, target in datasetTrain:
    if num < clientNum:
        net = CNN.Net()
        net = net.cuda()

        for param in net.state_dict():
            net.state_dict()[param] *= 0
            net.state_dict()[param] += serverNet.state_dict()[param]

        client = Client.Client(data, target, net, learningRate, momentum)
        clients.append(client)
        num += 1

# 参与方数据攻击处理
for i in range(badClientNum):
    clients[i].randomTarget()


# 本地训练
def localTrain():
    for cl in clients:
        cl.net.train()

        # trainData, trainTarget = cl.getDataRandom()
        trainData = cl.trainData
        trainTarget = cl.trainTarget

        trainData = trainData.cuda()
        trainTarget = trainTarget.cuda()

        for i in range(localTrainTimes):
            cl.optimizer.zero_grad()
            output = cl.net(trainData)
            loss = F.nll_loss(output, trainTarget.long())
            loss = loss.cuda()
            loss.backward()
            cl.optimizer.step()

# 联邦过程
def fedGA(chromosome):
    # 将主机的cnn参数清空
    for param in server.cnn.state_dict():
        server.cnn.state_dict()[param] *= 0

    # 计算加入聚合的总个数
    fedClientNum = np.sum(chromosome)

    # 聚合参数
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            cl = clients[i]
            for param in server.cnn.state_dict():
                server.cnn.state_dict()[param] += cl.net.state_dict()[param]

    for param in server.cnn.state_dict():
        server.cnn.state_dict()[param] /= fedClientNum

    # 重新分配
    for cl in clients:
        for param in server.cnn.state_dict():
            cl.net.state_dict()[param] *= 0
            cl.net.state_dict()[param] += server.cnn.state_dict()[param]


# 计算某跳染色体的准确率
def getChromosomeAcc(Chromosome):
    newNet = server.cnn
    newNet = newNet.cuda()

    # 计算加入聚合的总个数
    fedClientNum = np.sum(Chromosome)

    # 令参数为空
    for param in newNet.state_dict():
        newNet.state_dict()[param] *= 0

    # 聚合参与方的参数
    for i in range(len(Chromosome)):
        if Chromosome[i] == 1:
            cl = clients[i]
            for param in cl.net.state_dict():
                newNet.state_dict()[param] +=  cl.net.state_dict()[param]

    for param in newNet.state_dict():
        newNet.state_dict()[param] /= fedClientNum

    # 计算正确率
    newNet.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        data = server.testData.cuda()
        target = server.tesetTarget.cuda()

        output = newNet(data)

        p = F.nll_loss(output, target, reduction='sum')
        loss += p
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    loss /= len(server.testData)
    return correct / len(server.testData), loss

# 交叉
def Cross(population, ids, crossNum):
    k1 = int(ids[0])
    k2 = int(ids[1])

    crossFragment1 = sorted(random.sample(range(0, clientNum-1), crossNum))
    crossFragment2 = list(set(list(range(0, clientNum))) ^ set(crossFragment1))

    newChromosome1 = population[k1].copy()
    newChromosome2 = population[k2].copy()

    for index in crossFragment1:
        newChromosome1[index] = population[k2][index]

    for index in crossFragment2:
        newChromosome2[index] = population[k1][index]

    return newChromosome1, newChromosome2

# 变异
def Mutation(population, mutationNum, mutationRandom):
    startPos = np.random.randint(0, len(population[0]) - mutationNum, 1, dtype=int)[0]
    endPos = startPos + mutationNum

    mutationFragment = np.random.randint(0, 2, mutationNum, dtype=int)

    for chromosome in population:
        ran = random.random()
        if ran < mutationRandom:
            chromosome[startPos: endPos] = mutationFragment

# 计算验证集上的正确率
def getSeverAcc():
    newNet = server.cnn
    newNet = newNet.cuda()

    # 计算正确率
    newNet.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        data = server.testData.cuda()
        target = server.tesetTarget.cuda()

        output = newNet(data)

        p = F.nll_loss(output, target, reduction='sum')
        loss += p
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    loss /= len(server.testData)
    return 100. * correct / len(server.testData), loss

# 轮盘赌算法，选择两个
def rouletteWheel(fitness):
    # 被选择的索引
    ids = []
    while len(ids) != 2:
        # 生成随机数进行轮盘赌选择
        p = random.random()

        sum = 0
        for i in range(len(fitness)-1):
            sum += fitness[i]

            if sum >= p and i not in ids:
                ids.append(i)
                break
            elif sum <= p < sum + fitness[i+1] and (i+1) not in ids:
                ids.append((i+1))
                break
            elif i == clientNum - 2 and sum <= p:
                ids.append(clientNum-1)
                break

    return ids

# 主函数
if __name__ == '__main__':
    federatedNum = 100

    print("开始GA联邦")
    accFedSet = []
    lossFedSet = []

    # 每一个通信回合遗传算法迭代的次数
    GAIterateNum = 3

    # 种群索引
    chromosomeNum = 40
    populationIndex = list(range(chromosomeNum))

    # 随机生成初始种群
    population = np.random.randint(0, 2, (chromosomeNum, clientNum), dtype=int)

    for i in range(federatedNum):
        localTrain()

        iterateNum = 0

        # 本次通信回合中最好的一条染色体
        greatestFitness = 0
        greatestChromosome = np.zeros(clientNum)

        while iterateNum < GAIterateNum:
            # 计算适应度，此处采用最直观的准确率
            fitness = []
            for chromosome in population:
                chromosomeAcc, chromosomeLoss = getChromosomeAcc(chromosome)
                fitness.append(float(chromosomeAcc))

            for f in range(len(fitness)):
                fitness[f] = round(fitness[f], 4)

            # 选择最佳适应度的染色体
            greatIndex = np.argmax(fitness)

            # 判断是否优于历史最佳适应度，若是，则覆盖
            if fitness[greatIndex] > greatestFitness:
                greatestChromosome = population[greatIndex]
                greatestFitness = fitness[greatIndex]

            # 将最最差个体，并用历史最优个体覆盖
            worstIndex = np.argmin(fitness)
            population[worstIndex] = greatestChromosome
            fitness[worstIndex] = greatestFitness

            # 轮盘赌方法选择两个个体进行交叉，次数和初始种群大小一致
            # 令各个个体的适应度之和为1
            fitnessSum = np.sum(fitness)
            fitness /= fitnessSum

            newPopulation = population.copy()

            newNum = 0
            while newNum < chromosomeNum:
                ids = rouletteWheel(fitness)
                newPopulation[newNum], population[newNum+1] = Cross(population, ids, int(clientNum/2))
                newNum += 2

            population = newPopulation.copy()

            # 变异
            Mutation(population, 5, 0.01)

            iterateNum += 1

        correction = []
        lossSet = []
        for chromosome in population:
            chromosomeAcc, chromosomeLoss = getChromosomeAcc(chromosome)
            correction.append(float(chromosomeAcc))
            lossSet.append(chromosomeLoss)

        bestIndex = np.argmax(correction)
        bestChromosome = population[bestIndex]

        fedGA(bestChromosome)
        fedAcc = correction[bestIndex] * 100
        fedLoss = lossSet[bestIndex]

        print("第{}次的普通联邦正确率为{}，损失值为{}, 选择到的参与方组为{}".format(i+1, fedAcc, '%.4f' % fedLoss, bestChromosome))
        accFedSet.append(fedAcc)
        lossFedSet.append('%.4f' % fedLoss)
    print(accFedSet)
    print(lossFedSet)