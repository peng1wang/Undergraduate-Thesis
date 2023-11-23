import  pandas as pd
import  numpy as np
from  sklearn import  metrics
from  lstm.Predict_Interface import  PredictWithData
from Config import  Config
from sklearn.metrics import mean_squared_error as MSE,mean_absolute_error as MAE,max_error as ME,r2_score as r2
def GetRMSE(y_hat,y_test):
    sum = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
    return  sum

def GetMAE(y_hat,y_test):
    sum = metrics.mean_absolute_error(y_test, y_hat)
    return  sum

def GetMAPE(y_hat,y_test):
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

def GetMAPE_Order(y_hat,y_test):
    #删除y_test 为0元素
    zero_index = np.where(y_test == 0)
    y_hat = np.delete(y_hat,zero_index[0])
    y_test = np.delete(y_test,zero_index[0])
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum

config = Config()
print(config)



path = config.multpath
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
# Y_val =y_val.iloc[60265:80425,:].copy()
# Y_val1 = y_val.iloc[20268:40428,:].copy()
#线性增长变量做差分
dif_col=['LV1281 S04 WR706 BV25-1','LV1281 S01 Energy consumption','LV1281 S02 Energy consumption','LV1281 S03 Energy consumption']
for item in dif_col:
    ts_diff = np.diff(Y_val[item])
    Y_val[item] = np.append([0], ts_diff)
    ts_diff = np.diff(Y_val1[item])
    Y_val1[item] = np.append([0], ts_diff)
data = pd.concat([Y_val1, Y_val], ignore_index=True)
bxg=['LA1251 summary AmereMin charge','LV1211 S01 PN212_BV25','LV1211 S01 PN213_BV25','LV1281 AS361 M1','LV1281 VA307 M3','LV1281 WK363 BV25','LV1281 WK363 M2']
data=data.drop(columns=bxg)
#选取后20%
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

# continuous = [i for i in data.loc[:,data.nunique()>=1]][1:]
# # categorical = [i for i in df.loc[:,df.nunique()<=2]]
# df2 = pd.DataFrame(columns= continuous)
# for item in continuous:
#     # df1 = df[['Oven1_NOx_pre_-20',item]]
#     df1 = data[['Oven1_NOx', item]]
#     # print(df1.corr('spearman').iloc[0,1])
#     # df2.loc[0,[item]] = df1.corr('spearman').iloc[0, 1]
#     df2.loc[0, [item]] = df1.corr('pearson').iloc[0, 1]


# print(list(df2[(df2>0.6) | (df2<-0.6)].stack().loc[0].index))
# print(len(list(df2[(df2>0.6) | (df2<-0.6)].stack().loc[0].index)))
# df =data[list(df2[(df2>0.6) | (df2<-0.6)].stack().loc[0].index)]
#
# data.reset_index(drop=True, inplace=True)
# df.reset_index(drop=True, inplace=True)
# data = pd.concat([data['date'],df],axis=1,join='inner')




data2=data.iloc[:int(0.7*data.shape[0]),:]
data = data.iloc[int(0.7*data.shape[0]):,:]
print("长度为",data.shape[0])
name = config.dimname

normalize = np.load(config.multpath+name+".npy")
loadmodelname = config.multpath+name+".h5"
y_hat1,y_test1 = PredictWithData(data2,name,loadmodelname,normalize,config)

y_hat,y_test = PredictWithData(data,name,loadmodelname,normalize,config)
y_hat = np.array(y_hat,dtype='float64')
y_test = np.array(y_test,dtype='float64')
# print(y_test)
# y_test = np.insert(y_test,[0,0],y_test[0,0])
# y_test = y_test[0:-2]
# print(y_test)
y_hat1 = np.array(y_hat1,dtype='float64')
y_test1 = np.array(y_test1,dtype='float64')


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
score_rmse = np.sqrt(MSE(y_test, y_hat))
score_mae = MAE(y_test, y_hat)
# score_mae1 = MAE(train['Oven1_NMHC_pre_-20'].values, preds*pre[1]+pre[0])
score_me=ME(y_test, y_hat)
score_r2=r2(y_test, y_hat)
score_smape = smape(y_test, y_hat)
print(y_test)
print(y_hat)
print('score_rmse:',score_rmse)
print('score_mae:',score_mae)
print('score_me:',score_me)
print('score_r2:',score_r2)
print('score_smape:',score_smape)
a = np.argmax(y_test)
print(np.max(y_test)-np.max(y_hat[a-30:a+5,0]))
print('最大值差：',np.max(y_test)-np.max(y_hat[:,0]))

np.save(config.multpath+name+"y_hat.npy",y_hat)
np.save(config.multpath+name+"y_test.npy",y_test)
np.save(config.multpath+name+"y_hat1.npy",y_hat1)
np.save(config.multpath+name+"y_test1.npy",y_test1)
print("结束")
