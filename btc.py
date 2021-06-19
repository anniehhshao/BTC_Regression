# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:00:27 2021

@author: user
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import datasets
#a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
 
#for i in a:
    #print(i)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

data=pd.read_csv("C:/Users/user/Downloads/BTC-USD.csv")
data['y']=0
for i in range(0,data.shape[0]-1):
    data.loc[i,'y']=math.log(data.loc[i+1,'Close']/data.loc[i,'Close'],10)
data.dropna()
#y=data['y']
#data['y']=data['y']+data['Yesterday_Close']
#print(data)
#data.set_index("Date" , inplace=True)
data["Date"]=pd.to_datetime(data["Date"])
data["Close_5 MA"]=pd.Series(data['Close']).rolling(5,min_periods=2).mean()
data["Close_30 MA"]=pd.Series(data['Close']).rolling(30,min_periods=2).mean()
data["Close_60 MA"]=pd.Series(data['Close']).rolling(60,min_periods=2).mean()
data["Close_180 MA"]=pd.Series(data['Close']).rolling(180,min_periods=2).mean()
data=data.fillna(0)

train_set = data[:1950]
test_set = data[1950:]
train_y = train_set["y"]
train_x = train_set[["Open","Close","Volume","Close_5 MA","Close_30 MA","Close_60 MA","Close_180 MA"]]
test_x = test_set[["Open","Close","Volume","Close_5 MA","Close_30 MA","Close_60 MA","Close_180 MA"]]
test_y = test_set["y"]


result = []
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)
print("linear_tree_train Coef=",reg.coef_)
y_pred_train= reg.predict(train_x)
y_true_train = train_y.to_numpy()
print("linear_tree_train組均方誤差=",mean_squared_error(y_true_train,y_pred_train))
y_pred_test= reg.predict(test_x)
y_true_test = test_y.to_numpy()
print("Linear_tree_test組均方誤差=",mean_squared_error(y_true_test, y_pred_test))
row=[]
row.append("linear_tree_train組均方誤差")
row.append(round(mean_squared_error(y_true_train, y_pred_train), 10))
result.append(row)
row = []
row.append("Linear_tree_test組均方誤差")
row.append(round(mean_squared_error(y_true_test, y_pred_test), 10))
result.append(row)






reg = linear_model.Lasso(alpha=0.1)
reg.fit(train_x, train_y)
y_true_train = train_y.to_numpy()
y_pred_train= reg.predict(train_x)
y_true_train = train_y.to_numpy()
print("Lasso_train組均方誤差=",mean_squared_error(y_true_train,y_pred_train))
y_pred_test= reg.predict(test_x)
y_true_test = test_y.to_numpy()
print("Lasso_test組均方誤差=",mean_squared_error(y_true_test, y_pred_test))
row=[]
row.append("Lasso_train組均方誤差")
row.append(round(mean_squared_error(y_true_train, y_pred_train), 10))
result.append(row)
row = []
row.append("Lasso_test組均方誤差")
row.append(round(mean_squared_error(y_true_test, y_pred_test), 10))
result.append(row)





reg_decision = DecisionTreeRegressor()
reg_decision.fit(train_x, train_y)
y_pred_train= reg.predict(train_x)
y_true_train = train_y.to_numpy()
print("Decision_tree_train組均方誤差=",mean_squared_error(y_true_train,y_pred_train))
y_pred_test= reg.predict(test_x)
y_true_test = test_y.to_numpy()
print("Decision_tree_test組均方誤差=",mean_squared_error(y_true_test, y_pred_test))
row=[]
row.append("Decision_tree_train組均方誤差")
row.append(round(mean_squared_error(y_true_train, y_pred_train), 10))
result.append(row)
row = []
row.append("Decision_tree_test組均方誤差")
row.append(round(mean_squared_error(y_true_test, y_pred_test), 10))
result.append(row)
print(pd.DataFrame(result))




pd_y_train, pd_y_test= pd.DataFrame(train_y), pd.DataFrame(test_y)
pd_y_train["classify"],pd_y_test["classify"] = None,None

for i in range(0, pd_y_train.shape[0]):
    if(pd_y_train.iloc[i, 0] >= 0):
        pd_y_train.iloc[i, 1] = 1
    
    else:
        pd_y_train.iloc[i, 1] = 0
        
        
        

pd_y_test=pd_y_test.reset_index(drop=True)
for i in range(0, pd_y_test.shape[0]):
    if(pd_y_test.iloc[i, 0] >= 0):
        pd_y_test.iloc[i, 1] = 1
    
    else:
        pd_y_test.iloc[i, 1] = 0

clf = LogisticRegression()
clf = clf.fit(train_x.to_numpy(), np.asarray(list(pd_y_train["classify"])))

print(" LogisticRegression_train_accuracy_score=",accuracy_score(np.asarray(list(pd_y_train["classify"])), clf.predict(train_x.to_numpy())))
print(" LogisticRegression_test_accuracy_score=",accuracy_score(np.asarray(list(pd_y_test["classify"])), clf.predict(test_x.to_numpy())))
print(" LogisticRegression_train_roc_auc_score=",roc_auc_score(np.asarray(list(pd_y_train["classify"])), clf.predict(train_x.to_numpy())))
print(" LogisticRegression_test_roc_auc_score=",roc_auc_score(np.asarray(list(pd_y_test["classify"])), clf.predict(test_x.to_numpy())))

real=pd_y_test["classify"].to_list()+pd_y_train["classify"].to_list()
model= clf.predict(train_x.to_numpy()).tolist()+ clf.predict(test_x.to_numpy()).tolist()
#plt.scatter(data["Date"][-200:] , real[-200:], color="blue", label="真實值", s=0.1)
plt.scatter(data["Date"],model, color="red", label="模擬值", s=0.1)
plt.legend(ncol=3)
plt.show()
#X, y = datasets.load_digits(return_X_y=True)
#clf = clf.fit(X, y)









