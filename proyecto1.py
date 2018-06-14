import numpy as np
import random
import operator
import math
from copy import copy, deepcopy


MAX_POPULATION = 100
MAX_ITERATIONS = 20
MUTATION_RATE = 40

def getNeighbors(i,state):
	stateExt = np.concatenate(([0],state,[0]))
	neighbors = stateExt[i:(i+3)]
	return neighbors

def arrayToInt(bitlist):
	out = 0
	for bit in bitlist:
		out = (out << 1) | int(bit)
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

def sgn(x):
	if x > 0:
		return 1
	elif x == 0:
		return 0
	else:
		return -1

def initGenPool(m):
	m2 = np.zeros(shape=(len(m),8))
	for i in range(0,len(m)):
		for j in range(0,len(m[i])):
			if m[i][j] != 0:
				m2[i][j] = m[i][j] + random.random()
			else:
				m2[i][j] = 0
	return np.sort(np.concatenate(m2))

def getGen(genPool):
	return random.randint(0,len(genPool)-1)

def fitness(tp,fp,g1,g2,ap,an):
	return (1-(tp/(tp+fp)))+(0.01*((g1+g2)/(len(ap)+len(an))))

def getDataset(filename):
	file = open(filename,'r')
	features = []
	for line in file:
		cArray = line.split(",")
		iArray = np.zeros(len(cArray))
		for i in range(0,len(cArray)):
			iArray[i] = int(cArray[i])
		features.append(iArray) 
	return features

def initChromosome(pool1,pool2):
	chromosome = {"gen1":0,"gen2":0,"fitness":0} 
	chromosome["gen1"] = getGen(pool1)
	chromosome["gen2"] = getGen(pool2)
	return chromosome

def generatePopulation(pool1,pool2):
	population = []
	for i in range(0,MAX_POPULATION):
		for j in range (0,1):
			population.append(initChromosome(pool1,pool2))
	return population

def cellSumatory(m1,m2,state):
	total = 0
	for i in range(0,len(state)):
		total += transition(i,m1,state) - transition(i,m2,state)
	return total

def getClassification(m1,m2,state):
	return sgn(cellSumatory(m1,m2,state))

def synthesizeChromosome(m,pool,gen):
	m1 = deepcopy(m)
	for i in range(0,len(m)):
		for j in range(0,len(m[i])):
			#i = int(math.floor(gen/8))
			#j = gen%8
			if m1[i][j] > pool[gen]:
				m1[i][j] = 1
			else:
				m1[i][j] = 0
	return m1

def testPopulation(m1,m2,population,pool1,pool2,training_set):
	maxFP = 0
	maxTP = 0
	for i in range(0,len(population)):
		tp = 0
		fp = 0
		gen1 = population[i]["gen1"]
		gen2 = population[i]["gen2"]
		m1 = synthesizeChromosome(m1,pool1,gen1)
		m2 = synthesizeChromosome(m2,pool2,gen2)
		for t in training_set:
			classification = getClassification(m1,m2,t[0:(len(t)-1)])
			if classification >= 0:
				if t[(len(t)-1)] == 1:
					tp += 1
				else:
					fp += 1
			else:
				if t[(len(t)-1)] == 0:
					tp += 1
				else:
					fp += 1
		maxFP = max(maxFP,fp)
		maxTP = max(maxTP,tp)
		population[i]["fitness"] = fitness(tp,fp,gen1,gen2,pool1,pool2)
	#print(str(maxTP)+" "+str(maxFP))
	population = sorted(population, key = operator.itemgetter("fitness"))
	return population

def selection(population):
	return population[0:48]

def crossbreed(population):
	new = []
	while (len(population) + len(new)) < 100:
		parent1 = population[random.randint(0,len(population)-1)] 
		parent2 = population[random.randint(0,len(population)-1)] 
		child = {"gen1":0,"gen2":0,"fitness":0} 
		if random.random() * 100 < 50:
			child["gen1"] = parent1["gen1"]
			child["gen2"] = parent2["gen2"]
		else:
			child["gen1"] = parent1["gen2"]
			child["gen2"] = parent2["gen1"]
		new.append(child)
	return np.concatenate((population,new))


def mutate(population,max_val):
	for i in range(0,len(population)):
		if random.random() * 100 < MUTATION_RATE:
			wGen = random.random() * 100
			inc = random.random() * 100
			if wGen > 60:
				population[i]["gen1"] += 1 if inc > 50 else -1
			elif wGen > 30:
				population[i]["gen2"] += 1 if inc > 50 else -1
			else:
				inc2 = random.random() * 100
				population[i]["gen1"] += 1 if inc > 50 else -1
				population[i]["gen2"] += 1 if inc2 > 50 else -1 
			population[i]["gen1"] = population[i]["gen1"] % max_val
			population[i]["gen2"] = population[i]["gen2"] % max_val


def GA(m1,m2,training_set):
	pool1 = initGenPool(m1)
	pool2 = initGenPool(m2)
	population = generatePopulation(pool1,pool2)
	i = 0
	while i < MAX_ITERATIONS:
		population = testPopulation(m1,m2,population,pool1,pool2,training_set)
		population = selection(population)
		syntPop(m1,m2,population,pool1,pool2)
		population = crossbreed(population)
		mutate(population,len(pool1))
		pool1 = initGenPool(m1)
		pool2 = initGenPool(m2)
		i += 1


def test(m1,m2):
	test_set = getDataset("/Users/hectormoreno/Documents/CIC/Matematicas/Cellular Automata Classifier/SPECT/SPECT.test.txt")
	tp = 0
	fp = 0
	for t in test_set:
		classification = getClassification(m1,m2,t[0:(len(t)-1)])
		if classification >= 0:
			if t[(len(t)-1)] == 1:
				tp += 1
			else:
				fp += 1
		else:
			if t[(len(t)-1)] == 0:
				tp += 1
			else:
				fp += 1
	print("correctos= "+str(tp))
	print("incorrectos= "+str(fp))

def syntPop(m1,m2,population,pool1,pool2):
	for i in range(0,len(population)):
		gen1 = population[i]["gen1"]
		gen2 = population[i]["gen2"]
		i = int(math.floor(gen1/8))
		j = gen1%8
		if m1[i][j] > pool1[gen1]:
			m1[i][j] = 1
		else:
			m1[i][j] = 0
		i = int(math.floor(gen2/8))
		j = gen2%8
		if m2[i][j] > pool2[gen2]:
			m2[i][j] = 1
		else:
			m2[i][j] = 0
		

def CACTraining():
	training_set = getDataset("/Users/hectormoreno/Documents/CIC/Matematicas/Cellular Automata Classifier/SPECT/SPECT.train.txt")
	n = len(training_set[0]) # numero de atributos de una instancia + 1 de clase
	m1 = np.zeros(shape=(n,8))
	m2 = np.zeros(shape=(n,8))
	for i in range(0,len(training_set)):
		for j in range(0,len(training_set[i])):
			row = j
			col = arrayToInt(getNeighbors(j,training_set[i]))
			if training_set[i][n-1] == 1:
				m1[row][col] += training_set[i][j]
			else:
				m2[row][col] += training_set[i][j]
	print(m1)
	print(m2)		
	GA(m1,m2,training_set)
	test(m1,m2)
	print(m1)
	print(m2)






