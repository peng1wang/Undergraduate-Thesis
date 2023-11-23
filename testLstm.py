# -*- coding = utf-8 -*-
#@Time : 2021/4/6 17:06
#@Author : 123
#@file : testLstm.PY
#@software : PyCharm

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score as CVS
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE,mean_absolute_error as MAE,max_error as ME,r2_score as r2


import math


y_val = pd.read_csv(r'./train_data_111.csv')
y_val['date']=pd.to_datetime(y_val['date'])
# y_val.loc[y_val['LA1251 summary AmereMin charge']>=8000,'LA1251 summary AmereMin charge']=np.nan
# y_val['LA1251 summary AmereMin charge']=y_val['LA1251 summary AmereMin charge'].interpolate()
y_val.loc[y_val['LV1211 S01 PN111_BV25']>=65,'LV1211 S01 PN111_BV25']=np.nan
y_val['LV1211 S01 PN111_BV25']=y_val['LV1211 S01 PN111_BV25'].interpolate()

y_val.loc[y_val['LV1281 AS361 M1']==0,'LV1281 AS361 M1']=np.nan
y_val['LV1281 AS361 M1']=y_val['LV1281 AS361 M1'].interpolate()
y_val.loc[y_val['LV1281 VA307 BB42']==0,'LV1281 VA307 BB42']=np.nan
y_val['LV1281 VA307 BB42']=y_val['LV1281 VA307 BB42'].interpolate()
# y_val.loc[y_val['LV1281 VA307 M3']==0,'LV1281 VA307 M3']=np.nan
# y_val['LV1281 VA307 M3']=y_val['LV1281 VA307 M3'].interpolate()
y_val.loc[y_val['LV1281 VZ366 BB40']==0,'LV1281 VZ366 BB40']=np.nan
y_val['LV1281 VZ366 BB40']=y_val['LV1281 VZ366 BB40'].interpolate()
y_val.loc[y_val['LV1281 WK363 BB40-1']==0,'LV1281 WK363 BB40-1']=np.nan
y_val['LV1281 WK363 BB40-1']=y_val['LV1281 WK363 BB40-1'].interpolate()
y_val.loc[y_val['LV1281 WK363 BB40-2']==0,'LV1281 WK363 BB40-2']=np.nan
y_val['LV1281 WK363 BB40-2']=y_val['LV1281 WK363 BB40-2'].interpolate()



tihuan = ['LV1281 S05 WU301 BB40','LV1281 S05 WU301 BB40-1','LV1281 S06 WU302 BB40','LV1281 S07 WU311 BB40','LV1281 S08 WU312 BB40','LV1281 S09 WU321 BB40']
for item in tihuan:
    y_val.loc[y_val[item] == 0, item] = np.nan
    y_val[item] = y_val[item].interpolate()

tihuan2=['LV1281 S07 WU311 M2','LV1281 S08 WU312 M2','LV1281 S09 WU321 M2','LV1281 S05 WU301 M2','LV1281  S06 WU302 M2']
y_val.loc[73149:73541, tihuan2] = np.nan
for item in tihuan2:
    y_val[item] = y_val[item].interpolate()

y_val.loc[29118:29133,['Oven1_NMHC','Oven1_NOx']] = np.nan
y_val.loc[37679:37698,['Oven1_NMHC','Oven1_NOx']] = np.nan
y_val.loc[57740:57755,['Oven1_NMHC','Oven1_NOx']] = np.nan
y_val.loc[70569:70585,['Oven1_NMHC','Oven1_NOx']] = np.nan
y_val.loc[78940:78954,['Oven1_NMHC','Oven1_NOx']] = np.nan

y_val.loc[24098:24102,['Oven1_NMHC']] = np.nan
y_val.loc[24118:24122,['Oven1_NMHC']] = np.nan
y_val.loc[74000:74004,['Oven1_NMHC']] = np.nan
# y_val.loc[74000:74004,['Oven1_NMHC']] = np.nan
y_val['Oven1_NMHC']=y_val['Oven1_NMHC'].interpolate()
y_val['Oven1_NOx']=y_val['Oven1_NOx'].interpolate()

Y_val =y_val.iloc[56886:81076,:].copy()
Y_val1 = y_val.iloc[20268:44076,:].copy()


# Difference for the linear growth variable
dif_col=['LV1281 S04 WR706 BV25-1','LV1281 S01 Energy consumption','LV1281 S02 Energy consumption','LV1281 S03 Energy consumption']
for item in dif_col:
    ts_diff = np.diff(Y_val[item])
    Y_val[item] = np.append([0], ts_diff)
    ts_diff = np.diff(Y_val1[item])
    Y_val1[item] = np.append([0], ts_diff)
df1 = pd.concat([Y_val1, Y_val], ignore_index=True)
bxg=['LA1251 summary AmereMin charge','LV1211 S01 PN212_BV25','LV1211 S01 PN213_BV25','LV1281 AS361 M1','LV1281 VA307 M3','LV1281 WK363 BV25','LV1281 WK363 M2']
df1=df1.drop(columns=bxg)
IM=[
'date',
'LV1211 S01 PN111_BL50',
'LV1211 S04 VZ502_M1_C',
'LV1281 S04 WR706 M1',
'LV1281 S04 WA506 BB40-2',
'LV1281 S04 WA506 BB40-4',
'LV1281 S04 WA506 BB40-5',
'LV1281 S03 VA306 M1',
'LV1281 S08 WU312 M2',
'LV1281 S07 WU311 M2',
'LV1281 S06 WU302 M1',
'LV1281  S06 WU302 M2',
'LV1281 S05 WU301 M2',
'LV1281 S05 WU301 BB40-1',
'LV1281 VZ366 BB40',
'LV1281 VA307 BB42',
'LF1921 S01 TF270 Number of unit',
'LF1921 S01 TF270 Transport forward',
'LF1921 S01 TF270 Home position',
'LF1921 RB280 Forward',
'LF1921 RB290 Occupied',
'Oven1_NMHC',
'Oven1_NOx']

# df1=df1[IM]
# print(df1)

# print(Y_val.head())
# print(Y_val.shape)
# print(Y_val.isna().sum())
# print([column for column in Y_val] )

#the first 30 minutes of data and forecasts
def feature_engineer(df,name):
    N=30
    look_back=25
    skip = 60
    N2=30
    M=-20
    lag_cols = [column for column in df][1:]
    # lag_cols = [name]
    shift_range = [x + 1 for x in range(N)]

    for col in lag_cols:
        for i in shift_range:
            new_col='{}_lag_{}'.format(col, i)   # format string
            df[new_col]=df[col].shift(i)
    shift_range2 = [x + 61 for x in range(N2)]
    for col in lag_cols:
        for i in shift_range2:
            new_col='{}_lag_{}'.format(col, i)   # format string
            df[new_col]=df[col].shift(i)
    # for i in shift_range:
    #     new_col = '{}_lag_{}'.format(col, i)  # format string
    #     df[new_col] = df[col].shift(i)
    new_col = '{}_pre_{}'.format(name, M)  # format string
    df[new_col] = df[name].shift(M)

    return df[look_back*skip:M-1]
df = feature_engineer(df1,'Oven1_NOx')

# continuous = [i for i in df.loc[:,df.nunique()>=1]][1:]
# categorical = [i for i in df.loc[:,df.nunique()<=2]]
# print(df)
# df2 = pd.DataFrame(columns= continuous)
# for item in continuous:
#     # df1 = df[['Oven1_NOx_pre_-20',item]]
#     df1 = df[['Oven1_NOx_pre_-20', item]]
#     # print(df1.corr('spearman').iloc[0,1])
#     # df2.loc[0,[item]] = df1.corr('spearman').iloc[0, 1]
#     df2.loc[0, [item]] = df1.corr('pearson').iloc[0, 1]
#
#
# print(list(df2[(df2>0.6) | (df2<-0.6)].stack().loc[0].index))
# print(len(list(df2[(df2>0.6) | (df2<-0.6)].stack().loc[0].index)))
#
# df =df[list(df2[(df2>0.6) | (df2<-0.6)].stack().loc[0].index)]
print(df)
from Config import  Config
config = Config()
name = config.dimname

# y_hat = np.load("./Model/"+name+"y_hat.npy")
# df.insert(len(df.columns)-1,'yhat',y_hat.flatten())
# print(df)
# print(df.columns)
#
# allVa = list(df.columns)[1:]
# print(len(allVa))
# # categorical = [i for i in df.loc[:,df.nunique()<=2]]
# df2 = pd.DataFrame(columns= allVa)
# for item in allVa:
#     df1 = df[['Oven1_NOx_pre_-20',item]]
#     # print(df1.corr('spearman').iloc[0,1])
#     # df2.loc[0,[item]] = df1.corr('spearman').iloc[0, 1]
#     df2.loc[0, [item]] = df1.corr('pearson').iloc[0, 1]
#
#
# inpor = list(df2[(df2>0.4) | (df2<-0.4)].stack().loc[0].index)
# inpor.insert(0,'date')
# df = df.loc[:,inpor]
print(df)
continuous = [i for i in df.loc[:,df.nunique()>2]][1:]
categorical = [i for i in df.loc[:,df.nunique()<=2]]


#split train and test dataset
test_size = 0.3
num_test = int(test_size * len(df))
num_train = len(df) - num_test
train = df[:num_train]
test = df[num_train:]
print(train)

continuous = [i for i in df.loc[:,df.nunique()>2]][1:]
categorical = [i for i in df.loc[:,df.nunique()<=2]]
print(continuous)

scaler = StandardScaler().fit(train[continuous])

print(scaler)
train_scaled = scaler.transform(train[continuous])
test_scaled = scaler.transform(test[continuous])

pre=[]
pre.append(train['Oven1_NOx_pre_-20'].mean())
pre.append(train['Oven1_NOx_pre_-20'].std())
print(pre)



# Convert the numpy array back into pandas dataframe
train_scaled = pd.DataFrame(train_scaled, columns=continuous)
test_scaled = pd.DataFrame(test_scaled, columns=continuous)
print(test_scaled['Oven1_NOx_pre_-20']*pre[1]+pre[0])


train.reset_index(drop=True, inplace=True)
train_scaled.reset_index(drop=True, inplace=True)
temp_train = pd.concat([train[categorical],train_scaled],axis=1,join='inner')

test.reset_index(drop=True, inplace=True)
test_scaled.reset_index(drop=True, inplace=True)
temp_test = pd.concat([test[categorical],test_scaled],axis=1,join='inner')

print(temp_train)
# print(temp_test)


X_train_scaled = temp_train.iloc[:,0:-1]
Y_train_scaled = temp_train.iloc[:,-1]
X_test_scaled = temp_test.iloc[:,0:-1]
Y_test_scaled = temp_test.iloc[:,-1]

from sklearn.manifold import TSNE
temp = pd.concat([X_train_scaled,X_test_scaled],axis=0,join='inner')
# tsne = TSNE(n_components=3,init='pca')
# result = tsne.fit_transform(temp)
# temp = pd.DataFrame(result)

import umap
reducer = umap.UMAP(n_components=100)
result = reducer.fit_transform(temp)
temp = pd.DataFrame(result)


X_train_scaled=temp[:num_train]
X_test_scaled = temp[num_train:]
print(X_train_scaled)
print(Y_train_scaled)


from sklearn.decomposition import PCA
from sklearn import manifold

def myPCA(mcomponent, dataAttr, dataTestAttr):
    pca = PCA(n_components=mcomponent, copy=True, whiten=False)  # pca model

    dataAttrAfterPca = pd.DataFrame(pca.fit_transform(dataAttr))  
    dataTestAttrAfterPca = pd.DataFrame(pca.transform(dataTestAttr)) 

    print('降维后新特征的方差百分比: ', pca.explained_variance_ratio_,
          '\n在保持mcomponent达到', mcomponent, '后共有', len(pd.DataFrame(pca.explained_variance_ratio_)), '个新属性')

    dataAttrAfterPca = pd.DataFrame(dataAttrAfterPca.iloc[:, :len(pd.DataFrame(pca.explained_variance_ratio_))])

    # 把降维后的属性名改为str字符型的 便于存储成csv
    for each in dataAttrAfterPca.columns:
        dataAttrAfterPca.rename(columns={each: str(each)}, inplace=True)

    dataTestAttrAfterPca = dataTestAttrAfterPca.iloc[:, :len(pd.DataFrame(pca.explained_variance_ratio_))]
    for each in dataTestAttrAfterPca.columns:
        dataTestAttrAfterPca.rename(columns={each: str(each)}, inplace=True)

    return dataAttrAfterPca, len(pd.DataFrame(pca.explained_variance_ratio_)), dataTestAttrAfterPca


def myLLE(dataAttr, dataTestAttr):
    pca = manifold.LocallyLinearEmbedding(n_neighbors =15, n_components = 100,
                                method='standard')  # pca model

    dataAttrAfterPca = pd.DataFrame(pca.fit_transform(dataAttr))  
    dataTestAttrAfterPca = pd.DataFrame(pca.transform(dataTestAttr))  

    return dataAttrAfterPca, dataTestAttrAfterPca

x_train = X_train_scaled.values
x_test = X_test_scaled.values
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
y_train = Y_train_scaled.values
y_test = Y_test_scaled.values

from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
import tensorflow as tf
from tensorflow import keras as K

# ###################################################################################
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'dodgerblue', label='train loss')
        if loss_type == 'epoch':
            # # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'darkorange', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig('loss{0}'.format(loss_type))

    def acc_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'dodgerblue', label='train acc')
        # loss
        # plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'darkorange', label='val acc')
            # val_loss
            # plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="upper right")
        plt.savefig('acc{0}'.format(loss_type))
# ###############################################################################################################################





start = datetime.datetime.now()
#Build the LSTM model
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#创建室类
history = LossHistory()
#Train the model

model.fit(x_train, y_train, batch_size=200, epochs=100, validation_data=(x_test, y_test),callbacks=[history])

model.summary()

# history.loss_plot('epoch')
# history.acc_plot('epoch')
# Lets predict with the model
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
score = model.evaluate(x_test, y_test)

end = datetime.datetime.now()
print("程序运行时间："+str((end-start).seconds)+"秒")

# invert predictions


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

# Get the root mean squared error (RMSE) and MAE
print(test_predict[:,0]*pre[1]+pre[0])
print(test['Oven1_NOx_pre_-20'])
score_rmse = np.sqrt(MSE(test['Oven1_NOx_pre_-20'].values, test_predict[:,0]*pre[1]+pre[0]))
score_mae = MAE(test['Oven1_NOx_pre_-20'].values, test_predict[:,0]*pre[1]+pre[0])
score_mae1 = MAE(train['Oven1_NOx_pre_-20'].values, train_predict[:,0]*pre[1]+pre[0])
score_me=ME(test['Oven1_NOx_pre_-20'].values, test_predict[:,0]*pre[1]+pre[0])
score_r2=r2(test['Oven1_NOx_pre_-20'].values, test_predict[:,0]*pre[1]+pre[0])
score_smape = smape(test['Oven1_NOx_pre_-20'].values, test_predict[:,0]*pre[1]+pre[0])

print('score_rmse:',score_rmse)
print('score_mae:',score_mae)
print('score_me:',score_me)
print('score_r2:',score_r2)
print('score_smape:',score_smape)
a = np.argmax(test['Oven1_NOx_pre_-20'].values)
print(np.max(test['Oven1_NOx_pre_-20'].values)-np.max(test_predict[a-20:a+5,0]*pre[1]+pre[0]))
print('最大值差：',np.max(test['Oven1_NOx_pre_-20'].values)-np.max(test_predict[:,0]*pre[1]+pre[0]))
x_train_ticks = train['date']
y_train = train['Oven1_NOx_pre_-20']
x_test_ticks = test['date']


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(15)

sns.lineplot(x=x_train_ticks, y=y_train, ax=ax, label='Train Set') #navajowhite
sns.lineplot(x=x_train_ticks, y=train_predict[:,0]*pre[1]+pre[0], ax=ax,color='r', label='Train Prediction') #navajowhite.
sns.lineplot(x=x_test_ticks, y=test['Oven1_NOx_pre_-20'], ax=ax, color='orange', label='Ground truth') #navajowhite
sns.lineplot(x=x_test_ticks, y=test_predict[:,0]*pre[1]+pre[0], ax=ax, color='green', label='Prediction') #navajowhite

ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}, train_MAE:{score_mae1:.2f}', fontsize=14)
ax.set_xlabel(xlabel='Date', fontsize=14)
ax.set_ylabel(ylabel='Oven1_NOx', fontsize=14)
plt.savefig("LSTM1_NOx_60分钟_umap.jpg")