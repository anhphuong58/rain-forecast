import warnings
import numpy as np
import sys
from math import sqrt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X,Y, arima_order):
	# prepare training dataset
	
	train, test = np.array(X), np.array(Y)
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	history = np.array(history)
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	mape = mean_absolute_error(test, predictions)*100
	return [mape,rmse]


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(datasetX,datasetY, p_values, d_values, q_values):
	datasetX = datasetX.astype('float32')
	datasetY = datasetY.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(datasetX,datasetY, order)
					#if rmse < best_score:
					#	best_score, best_cfg = rmse, order
					#print('ARIMA%s MAPE=%.3f ' % (order,rmse))
					print(rmse)
				except:
					print("Có ngoại lệ ",sys.exc_info()[0]," xảy ra.")
					continue
	#print('Best ARIMA%s MAPE=%.3f' % (best_cfg, best_score))
 
# load dataset

seriesX = read_csv('datatrain.csv', header=0)
seriesX =np.array(seriesX["PRECTOTCORR"])
seriesY = read_csv('datatest.csv', header=0)
seriesY =np.array(seriesY["PRECTOTCORR"])
# evaluate parameters
#p_values = [0, 1, 2, 4, 6, 8, 10]
p_values = [2]
#d_values = range(0, 3)
#q_values = range(0, 3)
d_values = [0]
q_values = [1]

warnings.filterwarnings("ignore")
evaluate_models(seriesX,seriesY, p_values, d_values, q_values)

