import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
from eval_functions import *
import configuration

model_name=configuration.eval_model_name

try:
	model = pickle.load(open("models/"+model_name+".pickle.dat", "rb"))
except:
	print("Model not found, make sure the model exists in the models folder.")
	quit()

print("\nModel loaded!")


x_test=np.load("models/"+model_name+"_x_test.npy")
y_test=np.load("models/"+model_name+"_y_test.npy")
extra_test=np.load("models/"+model_name+"_extra_test.npy",allow_pickle=True)


# plot ROC AUC score


# load positive prediction probabilities (probability stock goes up) into a list
pred=model.predict_proba(x_test)
predyes=[]
for p in pred:
  predyes.append(p[1])

# plot ROC AUC score
if configuration.show_auc_roc_curve:
	plot_roc_auc(model,x_test,y_test)

# plot precision, recall, and average gain for range of confidence levels
if configuration.show_prec_recall:
	plot_precision_recall(pred,y_test,extra_test)

if configuration.show_backtest:
	plot_backtest(extra_test,pred,confidence=configuration.backtest_conf,n_positions=configuration.backtest_max_positions,
			 start_cash=configuration.backtest_cash,index=configuration.backtest_index,title=configuration.backtest_title)



