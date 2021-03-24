import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
from eval_functions import *



model_name=input("Input the name of the model you would like to test: ")

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
# plot_roc_auc(model,x_test,y_test)

# plot precision, recall, and average gain for range of confidence levels
# plot_precision_recall(pred,y_test,extra_test)

# 
plot_backtest(extra_test,pred,confidence=0.8,n_positions=5, start_cash=100000,index='XIU.TO',title='TITLE')



