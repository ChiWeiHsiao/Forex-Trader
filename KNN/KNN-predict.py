import requests
import json
from datetime import datetime, timedelta
import numpy as np
from operator import itemgetter
import time 
import Queue
import heapq

#parameters
instr = "EUR_USD"
base_units = 1000
historyX = [] #dimension: 49510x48
historyY = [] #dimension: 49510

testX = []
testY = []

kval = 3

def getHistoryNodes():
    # get history data from file
    # 50 candle prices in one node (25min), 1000 nodes in history(17day)
    with open("history-overlap-49510-nodes-2016dec.txt", "r") as f:
        next(f) #skip first line ( time )
        for line in f: #49510 lines
            data = line.split()
            node = []
            for number in data: #49 numbers
                node.append(float(number))
            historyX.append(node[0:-1])
            historyY.append(node[-1])

def getTestingNodes():
    # get history data from file
    # 50 candle prices in one node (25min), 1000 nodes in history(17day)

    with open("history-overlap-49510-nodes-2017jan.txt", "r") as f:
        print "Use Different Test Data (one month later)"
    #with open("history-overlap-49510-nodes-2016dec.txt", "r") as f:
        #print "Use Same Test Data (one month later)"
        next(f) #skip first line ( time )
        for line in f: #49510 lines
            data = line.split()
            node = []
            for number in data: #49 numbers
                node.append(float(number))
            testX.append(node[0:-1])
            testY.append(node[-1])

def distance( node1, node2 ):
    euclidean = np.linalg.norm( np.array(node1)[0:-1] - np.array(node2)[0:-1] ) #-1: drop the last element of one node
    return euclidean

def predictChange():
    # use KNN-like mmethod to predict if the price will rise/drop
    # use data 2016dec(49510nodes, 49 nums/node), to predict data 2017jan
    # Y = { True(rise), False(drop) }
    correct = 0
    error = 0
    
    testLength = len(testY)
    historyLength = len(historyY)
    
    numTestNodes = 100
    print "number of Testing Nodes: ", numTestNodes
    #for t in range(testLength):
    for t in range(numTestNodes):
        # calculate the distance between history-nodes and current-node
        # choose the k closest nodes to vote
        kneighbors = []
        for h in range(kval):
            heapq.heappush( kneighbors, (-distance(testX[t], historyX[h]), h))
        for h in range(kval, historyLength):
            newNodeDistance = -distance(testX[t], historyX[h])
            if newNodeDistance > kneighbors[0][0]:
                newNode = ( newNodeDistance, h )
                heapq.heappushpop( kneighbors, newNode)
                # node is ( -1*distance, index ) use '-1' to let queue be maxHeap

        rise = 0
        drop = 0
        # Vote for the answer by k neighbors
        totalWeights = 0
        for n in kneighbors:
            weight = (-1.0) / n[0]
            totalWeights = totalWeights + weight
            if( historyY[ n[1] ] > 0 ):
                rise = rise + weight #1
            else:
                drop = drop + weight #1
        predict = (rise > drop) # True=>rise, False=>drop

        #check if predicted y == real testing y?
        if predict == (testY[t] > 0):
            correct = correct + 1
            #print "correct, ans=", testY[t]
        else:
            error = error + 1
            #print "error, ans=", testY[t]
    
    print "Correct Prediction: ", correct
    print "Error Prediction: ", error
    correctness = correct / float(correct+error)
    print "Correctness: ", correctness
    return correctness
    

if __name__ == "__main__":
    getHistoryNodes()
    getTestingNodes()
    print "Prepare for Data...finished"
    print "Use weighted voting"

    results = []

    for i in range(10):
        print "K = ", kval
        results.append( ( kval, predictChange() ))
        kval = kval + 2

    results.sort(key=itemgetter(1))
    print "Results:" ,results
    print "Best kval=", results[-1]
    #getCurNode()
