import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import yfinance as yf

def plot_roc_auc(model,x_test,y_test):

	ns_probs = [0 for _ in range(len(y_test))]
	lr_probs = model.predict_proba(x_test)
	lr_probs = lr_probs[:, 1]
	ns_auc = roc_auc_score(y_test, ns_probs)
	lr_auc = roc_auc_score(y_test, lr_probs)
	print('No Skill: ROC AUC=%.3f' % (ns_auc))
	print('XGB: ROC AUC=%.3f' % (lr_auc))
	ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
	lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
	# plot the roc curve for the model
	plt.figure(figsize=(10,6))
	plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
	plt.plot(lr_fpr, lr_tpr, label='XGBoost')
	# axis labels
	plt.xlabel('False Positive Rate', fontsize=20)
	plt.ylabel('True Positive Rate', fontsize=20)
	# show the legend
	plt.legend(fontsize=20)
	# show the plot
	plt.show()

def plot_precision_recall(pred,y_test,extra_test):
	# this script calculates precision and recall for various confidence levels

	avgain=[]
	recall=[]
	precision=[]

	conflist=[.1,.2,.3,.4,.5,.6,.7,.8]

	for conflevel in conflist:

	  conf=conflevel
	  bin=[]
	  for p in pred:
	    if p[1]>conf:
	      bin.append(1)
	    else:
	      bin.append(0)

	  c=0
	  right=0
	  total=0
	  gain=[]

	  for p in bin:
	    if p==1:
	      gain.append(extra_test[c][1])
	      if y_test[c]==1:
	        right+=1
	    c+=1
	  avgain.append(sum(gain)/len(gain))
	  precision.append(right/sum(bin))

	  c=0
	  right=0
	  total=0
	  for y in y_test:
	    if y==1:
	      if bin[c]==1:
	        right+=1
	      total+=1
	    c+=1
	  # print('Recall:',right/total)
	  recall.append(right/total)	

	fig=plt.figure(figsize=(10, 6))
	fig.suptitle('Precision recall plot')
	plt.plot(conflist,avgain, label='Average gain')
	plt.plot(conflist,precision, label='Precision')
	plt.plot(conflist,recall, label='Recall')
	plt.legend(loc="upper right", fontsize=16)
	plt.xlabel('Prediction Probability', fontsize=24)
	plt.ylabel('%', fontsize=24)
	plt.show()


# back test model
def plot_backtest(val_data,pred,confidence=0.2,n_positions=5, start_cash=100000,index='SPY',title='Enter_title'):

  """
  Takes validation data and returns the performance for the period for plotting

  Input:

  val_data : numpy array with shape n * [date, ticker, 10 day gain/loss]
  pred : numpy array with shape n * [model prediction]
  confidence : number between 0 and 1 indicating how high model prediction must be to buy stock
  n_positions : max number of stock positions at any given time, diversification factor
  start_cash : hypothetical starting cash to buy stocks

  Output:

  dates : list of dates
  total_hist : total value (cash + fmv positions) of portfolio on each date
  cash_hist : cash on each date
  fmv_hist : fmv positions on each date
  holdings : holdings on each date


  """

  # organize validation data and predictions into single sorted dataframe
  pred_df=pd.DataFrame(pred)
  extra_df=pd.DataFrame(val_data)
  extra_df['prediction']=pred_df[0]
  df=extra_df.sort_values(by=0)

  df=df.reindex(columns=[0,2,1,'prediction'])
  df.columns=[0,1,2,'prediction']
  # return df




  # initialize variables
  cash=start_cash
  max_positions=n_positions
  d=None
  shape=df.shape

  pos={}
  fmv=0
  fmv_hist=[]
  cash_hist=[]
  total_hist=[]
  dates=[]
  holdings=[]
  x=0

  # start loop through all of the rows in the df
  for r in range(df.shape[0]):
    # if new date make new df 
    if df.iloc[r,0]!=d:
      d=df.iloc[r,0]
      temp=df.loc[(df[0] == d) & (df['prediction'] > confidence)]
      temp=temp.sort_values(by='prediction',ascending=False)
      


      # first remove positions sold after 10 days and add back cash
      # add one more day to each position that is not 10 days
      ls=list(pos.keys())
      for p in ls:
        if pos[p][2]==10:
          cash=cash+(pos[p][0]*(1+pos[p][1]))
          pos.pop(p)
        else:
          pos[p][2]+=1
        
      # add positions
      ls=list(pos.keys())
      for t in range(temp.shape[0]):
        n_pos=len(pos)
        if len(pos)<max_positions:
          if temp.iloc[t,1] not in ls:
            purch_price=cash/(max_positions-n_pos)
            pos[temp.iloc[t,1]]=[cash/(max_positions-n_pos),temp.iloc[t,2],0]
            cash=cash-purch_price

      
      # record fmv and cash to history
      fmv=0
      for p in pos:
        fmv=fmv+pos[p][0]

      fmv_hist.append(fmv)

      cash_hist.append(cash)

      total_hist.append(fmv+cash)
      dates.append(d)

      holdings.append(list(pos.keys()))

  start=dates[0]
  end=dates[-1]
  bench_date,bench_value=benchmark(index=index,start=start,end=end,cash=start_cash)

  fig=plt.figure(figsize=(12, 8))
  fig.suptitle(title+'\n Portfolio Performance \n Confidence : >'+str(confidence)+' \n Max portfolio size : '+str(max_positions)+' stocks', fontsize=12)
  plt.xlabel('Date', fontsize=24)
  plt.ylabel('Portfolio value', fontsize=24)
  plt.plot(dates,total_hist, label='Portfolio')
  plt.plot(bench_date,bench_value, label='Index performance ('+index+')')
  plt.legend(loc="upper left", fontsize=14)
  plt.tick_params(axis='both', which='major', labelsize=16)
  plt.tick_params(axis='both', which='minor', labelsize=8)
  plt.show()


def benchmark(index='SPY',exchange='',start='2020-01-01',end='2021-03-15',cash=100000):

  data=yf.download(tickers=index, start=start, end=end,interval="1d")  
  data=pd.DataFrame(data)
  data=data.reset_index()[['Date','Adj Close']]
  strt=data['Adj Close'].iloc[0]
  data['value']=(data['Adj Close']/strt)*cash
  return data['Date'], data['value']