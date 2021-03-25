import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from xgb_functions import *
import configuration

# import variables from config file
ticker_file=configuration.ticker_list_file
exchange=configuration.ticker_exchange
n=configuration.ticker_sample_size
col_name=configuration.ticker_col_name
train_start=configuration.training_data_start
train_end=configuration.training_data_end
test_start=configuration.test_data_start
test_end=configuration.test_data_end
save_model=configuration.save_model
model_name=configuration.model_name
index_ticker=configuration.index

# get list of tickers from csv
tickerlist=get_samples(file=ticker_file,n=n,col_name=col_name)

# download data from yfinance
train_data=get_data(tickerlist,exchange=exchange,start=train_start,end=train_end)
test_data=get_data(tickerlist,exchange=exchange,start=test_start,end=test_end)

# add technical indicators and target to training data
traindf=SMA(train_data,10)
traindf=FI(traindf,30)
traindf=RSI(traindf,14)
traindf=MACD(traindf,10,30)
traindf=MACD(traindf,5,10)
traindf=MACDiff(traindf,5)
traindf=index(traindf,index=index_ticker,days=30)
traindf=set_target(traindf,10)

# add technical indicators and target to test data
testdf=SMA(test_data,10)
testdf=FI(testdf,30)
testdf=RSI(testdf,14)
testdf=MACD(testdf,10,30)
testdf=MACD(testdf,5,10)
testdf=MACDiff(testdf,5)
testdf=index(testdf,index=index_ticker,days=30)
testdf=set_target(testdf,10)

# specify features to use and generate train and test sets
feat=['sma','rsi','fi','macd','index','y','target_gains']
trainset=generate_xgb_set(traindf,features=feat)
trainset=trainset.dropna()
testset=generate_xgb_set(testdf,features=feat)
testset=testset.dropna()
print('Trainset columns:', trainset.columns)

# Based on correlation with the target these are observed to be the best features
best_feat=['_sma_10','_rsi_14','_fi_30','_macd_5_10','_macd_10_30_diff5','_index_SPY30']

# dump train and test sets into numpy arrays for training
x_train=trainset[best_feat].to_numpy()
y_train=trainset['_y'].to_numpy()
extra_train=trainset[['Date','_target_gains','STOCK']].to_numpy()

x_test=testset[best_feat].to_numpy()
y_test=testset['_y'].to_numpy()
extra_test=testset[['Date','_target_gains','STOCK']].to_numpy()

print('Train and test set generated, training model...')

model = XGBClassifier(max_depth=10,n_estimators=100, learning_rate=0.1, use_label_encoder=False)
model.fit(x_train,y_train)

print('Training accuracy:',model.score(x_train,y_train))
print('Test accuracy:',model.score(x_test,y_test))

	
if save_model:
	import pickle
	from numpy import save
	file_name=model_name
	pickle.dump(model, open("models/"+file_name+".pickle.dat", "wb"))
	save("models/"+file_name+"_x_test.npy",x_test)
	save("models/"+file_name+"_y_test.npy",y_test)
	save("models/"+file_name+"_extra_test.npy",extra_test)
	print("Successfully saved model in models folder!")





