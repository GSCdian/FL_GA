import numpy as np
import torch
import torchvision
from Entity import Server, Client, CNN
import torch.nn.functional as F

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

# 普通联邦过程
def fed():
    # 将主机的cnn参数清空
    for param in server.cnn.state_dict():
        server.cnn.state_dict()[param] *= 0

    # 聚合参数
    for cl in clients:
        for param in server.cnn.state_dict():
            server.cnn.state_dict()[param] += cl.net.state_dict()[param]

    for param in server.cnn.state_dict():
        server.cnn.state_dict()[param] /= len(clients)

    # 重新分配
    for cl in clients:
        for param in server.cnn.state_dict():
            cl.net.state_dict()[param] *= 0
            cl.net.state_dict()[param] += server.cnn.state_dict()[param]

# 计算主机的正确率
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


# 主函数
if __name__ == '__main__':
    federatedNum = 100

    print("开始普通联邦")
    accFedSet = []
    lossFedSet = []
    for i in range(federatedNum):
        localTrain()

        # changeWeights = 0
        # index = -1
        #
        # for j in range(len(clients)):
        #     cl = clients[j]
        #     changeWeight = server.cnn
        #
        #     for param in changeWeight.state_dict():
        #         changeWeight.state_dict()[param] *= 0
        #         changeWeight.state_dict()[param] = server.cnn.state_dict()[param] - cl.net.state_dict()[param]
        #
        #     if changeWeights == 0:
        #         changeWeights = changeWeight
        #     else:
        #         if changeWeight > changeWeights:
        #             changeWeights = changeWeight
        #             index = j
        #
        # print(j)

        fed()
        fedAcc, fedLoss = getSeverAcc()
        print("第{}次的普通联邦正确率为{}，损失值为{}".format(i+1, fedAcc, '%.4f' % fedLoss))
        accFedSet.append(fedAcc)
        lossFedSet.append('%.4f' % fedLoss)
    print(accFedSet)
    print(lossFedSet)