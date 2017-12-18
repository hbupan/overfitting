import numpy as np
from sklearn import linear_model
import os

def createData(filename):
	a = np.random.rand(10, 4)
	
	try:
		np.save(filename, a)
	except:
		print("some error occur when creating data")

def loadData(filename):
	try:
		a = np.load(filename + '.npy')
		return a
	except:
		print("cant loading data from file")
		return None

def caculateYthenSave(a, filename):
	b = np.array([3, 4, 6, 1])
	np.save(filename, a.dot(b.T)) 


def modelTrain(X, Y, modelFile):
	regr = linear_model.LinearRegression()
	regr.fit(X, Y)
	print(regr.coef_)

modelTrain(loadData("test"), loadData("testY"), None)

