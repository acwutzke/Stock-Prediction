import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
from eval_functions import *
from xgb_functions import *
import configuration
from datetime import datetime, timedelta

# import variables from config file
ticker_file=configuration.ticker_list_file
exchange=configuration.ticker_exchange
n=configuration.ticker_sample_size
col_name=configuration.ticker_col_name
model_name=configuration.model_name
index_ticker=configuration.index

# get list of tickers from csv
tickerlist=get_samples(file=ticker_file,n=n,col_name=col_name)

# download data from yfinance
ct = datetime.today()+timedelta(days=1)
end = ct.strftime("%Y-%m-%d")
start=ct-timedelta(days=120)
start=start.strftime("%Y-%m-%d")

prediction_data=get_data(tickerlist,exchange=exchange,start=start,end=end)

print("\nData downloaded!")

predictiondf=SMA(prediction_data,10)
predictiondf=FI(predictiondf,30)
predictiondf=RSI(predictiondf,14)
predictiondf=MACD(predictiondf,10,30)
predictiondf=MACD(predictiondf,5,10)
predictiondf=MACDiff(predictiondf,5)
predictiondf=index(predictiondf,index=index_ticker,days=30)
# predictiondf=set_target(traindf,10)

print("\nTechnicals generated!")

feat=['sma','rsi','fi','macd','index']
predictionset=generate_xgb_set(predictiondf,features=feat)
predictionset=predictionset.dropna()

# Based on correlation with the target these are observed to be the best features
best_feat=['_sma_10','_rsi_14','_fi_30','_macd_5_10','_macd_10_30_diff5','_index_SPY30']

# dump train and test sets into numpy arrays for training
x_prediction=predictionset[best_feat].to_numpy()
extra_prediction=predictionset[['Date','STOCK']].to_numpy()

print("\nDataset prepared!")

model_name=configuration.pred_model_name

try:
	model = pickle.load(open("models/"+model_name+".pickle.dat", "rb"))
except:
	print("Model not found, make sure the model exists in the models folder.")
	quit()

print("\nModel loaded! making predictions...\n")

pred=model.predict_proba(x_prediction)
predyes=[]
for p in pred:
  predyes.append(p[1])

resultdf=pd.DataFrame(extra_prediction)
resultdf.columns=['Date','Stock']
resultdf['prediction']=predyes
resultdf=resultdf.sort_values(by=['Date','prediction'], ascending=False).head(configuration.n_picks)
print(resultdf)



