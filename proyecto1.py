import numpy as np
import random

m1 = np.zeros(shape=(4,8))
m2 = np.zeros(shape=(4,8))

m11 = np.zeros(shape=(4,8))
m22 = np.zeros(shape=(4,8))

m1[0] = [0,1,0,0,0,0,1,1]
m1[1] = [1,0,1,1,0,0,1,0]
m1[2] = [0,0,1,0,1,0,1,1]
m1[3] = [0,1,0,1,0,0,1,1]

m2[0] = [0,0,1,0,0,0,0,1]
m2[1] = [0,0,0,0,1,1,0,1]
m2[2] = [1,1,0,1,0,0,1,0]
m2[3] = [1,1,1,1,0,0,0,0]

def getNeighbors(i,state):
	stateExt = np.concatenate(([0],state,[0]))
	neighbors = stateExt[i:(i+3)]
	return neighbors

def arrayToInt(bitlist):
	out = 0
	for bit in bitlist:
		out = (out << 1) | bit
	return out	

def transition(i,m,state):
	row = i
	col = arrayToInt(getNeighbors(i,state))
	return m[row][col]

def nextState(state,m):
	next = np.zeros(len(state))
	for i in range(0,len(state)):
		next[i] = transition(i,m,state)
	return next

def cellSumatory(state):
	total = 0
	for i in range(0,len(state)):
		total += transition(i,m1,state) - transition(i,m2,state)
	return total

def sgn(x):
	if x > 0:
		return 1
	elif x == 0:
		return 0
	else:
		return -1

def getRuleMatrix(state):
	if sgn(cellSumatory(state)) >= 0:
		return m1
	else:
		return m2

def newState(state):
	return nextState(state,getRuleMatrix(state))

def initChromosome(m,m2):
	for i in range(0,len(m)):
		for j in range(0,len(m[i])):
			if m[i][j] != 0:
				m2[i][j] = m[i][j] + random.random()
			else:
				m2[i][j] = 0
	return np.sort(np.concatenate(m2))

def getGen(Chromosome)
	return random.randint(0,len(Chromosome))

def fitness(tp,fp,g1,g2,ap,an):
	return (1-(tp/tp+fp))+(0.01*((g1+g2)/(len(ap)+len(an))))


def changeM(m,g,a):
	for i in range(0,len(m)):
		for j in range(0,len(m[i])):
			if m[i][j] > a[g]:
				m[i][j] = 1
			else:
				m[i][j] = 0

