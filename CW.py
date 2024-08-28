import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def generateData(numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        centerX, centerY = random.randint(0, 5), random.randint(0, 5)
        for rowNum in range(numberOfClassEl):
            data.append([[random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)], classNum])
    return data

def splitTrainTest(data, testPercent):
    trainData = []
    testData  = []
    for row in data:
        if random.random() < testPercent:
            testData.append(row)
        else:
            trainData.append(row)
    return trainData, testData

def classifyKNN(trainData, testData, k, numberOfClasses):
    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    testLabels = []
    for testPoint in testData:
        testDist = [[dist(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]
        stat = [0] * numberOfClasses
        for d in sorted(testDist)[0:k]:
            stat[d[1]] += 1
        testLabels.append(sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1])
    return testLabels

def generateTestMesh(trainData):
    x_min = min([trainData[i][0][0] for i in range(len(trainData))]) - 1.0
    x_max = max([trainData[i][0][0] for i in range(len(trainData))]) + 1.0
    y_min = min([trainData[i][0][1] for i in range(len(trainData))]) - 1.0
    y_max = max([trainData[i][0][1] for i in range(len(trainData))]) + 1.0
    h = 0.05 # детализация фона
    testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
    return [testX, testY]
def showDataOnMesh(nClasses, nItemsInClass, k, testPercent):
    data = generateData(nItemsInClass, nClasses)
    trainData, testDataWithLabels = splitTrainTest(data, testPercent)
    testData = [testDataWithLabels[i][0] for i in range(len(testDataWithLabels))]
    testDataLabels = classifyKNN(trainData, testData, k, nClasses)
    print("Точность: ", sum([int(testDataLabels[i] == testDataWithLabels[i][1]) for i in range(len(testDataWithLabels))]) / float(len(testDataWithLabels)))

    trainData      = data
    testMesh       = generateTestMesh(trainData)
    testMeshLabels = classifyKNN(trainData, zip(testMesh[0].ravel(), testMesh[1].ravel()), k, nClasses)
    classColormap  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    testColormap   = ListedColormap(['#FF9999', '#99FF99', '#9999FF'])
    plt.pcolormesh(testMesh[0], # фон
                  testMesh[1],
                  np.asarray(testMeshLabels).reshape(testMesh[0].shape),
                  cmap=testColormap)
    plt.scatter([trainData[i][0][0] for i in range(len(trainData))], # точки
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))],
               cmap=classColormap)
    plt.show()

showDataOnMesh(3,10,3,0.5)
