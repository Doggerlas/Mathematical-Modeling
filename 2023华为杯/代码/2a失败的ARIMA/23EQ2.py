import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="matplotlib")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df1 = pd.read_excel("表1-患者列表及临床信息.xlsx")
df2 = pd.read_excel("表2-患者影像信息血肿及水肿的体积及位置.xlsx")
df3 = pd.read_excel("附表1-检索表格-流水号vs时间New.xlsx")

df1 = pd.concat([df1["入院首次检查流水号"],df1["发病到首次影像检查时间间隔"]],axis=1)
df2 = pd.concat([df2["入院首次检查流水号"],df2["ED_volume0"],df2["ED_volume1"],df2["ED_volume2"],df2["ED_volume3"],df2["ED_volume4"],df2["ED_volume5"],df2["ED_volume6"],df2["ED_volume7"],df2["ED_volume8"]],axis=1)
df3 = pd.concat([df3["入院首次检查流水号"],df3["入院首次检查时间点"],df3["随访1时间点"],df3["随访2时间点"],df3["随访3时间点"],df3["随访4时间点"],df3["随访5时间点"],df3["随访6时间点"],df3["随访7时间点"],df3["随访8时间点"]],axis=1)

df=pd.merge(df1,df2,on=['入院首次检查流水号'])
df=pd.merge(df,df3,on=['入院首次检查流水号'])
df=df.head(100)
df

df['入院首次检查时间点'] = pd.to_datetime(df['入院首次检查时间点'])
df['随访1时间点'] = pd.to_datetime(df['随访1时间点'])
df['发病到随访1时间点时间间隔'] = (df['随访1时间点'] - df['入院首次检查时间点']).dt.total_seconds() / 3600 + df["发病到首次影像检查时间间隔"]

df['随访2时间点'] = pd.to_datetime(df['随访2时间点'])
df['发病到随访2时间点时间间隔'] = (df['随访2时间点'] - df['随访1时间点']).dt.total_seconds() / 3600 + df["发病到随访1时间点时间间隔"]

df['随访3时间点'] = pd.to_datetime(df['随访3时间点'])
df['发病到随访3时间点时间间隔'] = (df['随访3时间点'] - df['随访2时间点']).dt.total_seconds() / 3600 + df["发病到随访2时间点时间间隔"]

df['随访4时间点'] = pd.to_datetime(df['随访4时间点'])
df['发病到随访4时间点时间间隔'] = (df['随访4时间点'] - df['随访3时间点']).dt.total_seconds() / 3600 + df["发病到随访3时间点时间间隔"]

df['随访5时间点'] = pd.to_datetime(df['随访5时间点'])
df['发病到随访5时间点时间间隔'] = (df['随访5时间点'] - df['随访4时间点']).dt.total_seconds() / 3600 + df["发病到随访4时间点时间间隔"]

df['随访6时间点'] = pd.to_datetime(df['随访6时间点'])
df['发病到随访6时间点时间间隔'] = (df['随访6时间点'] - df['随访5时间点']).dt.total_seconds() / 3600 + df["发病到随访5时间点时间间隔"]

df['随访7时间点'] = pd.to_datetime(df['随访7时间点'])
df['发病到随访7时间点时间间隔'] = (df['随访7时间点'] - df['随访6时间点']).dt.total_seconds() / 3600 + df["发病到随访6时间点时间间隔"]

df['随访8时间点'] = pd.to_datetime(df['随访8时间点'])
df['发病到随访8时间点时间间隔'] = (df['随访8时间点'] - df['随访7时间点']).dt.total_seconds() / 3600 + df["发病到随访7时间点时间间隔"]

res = pd.concat([df["入院首次检查流水号"],
          df["发病到首次影像检查时间间隔"],df["ED_volume0"],
          df["发病到随访1时间点时间间隔"],df["ED_volume1"],
          df["发病到随访2时间点时间间隔"],df["ED_volume2"],
          df["发病到随访3时间点时间间隔"],df["ED_volume3"],
          df["发病到随访4时间点时间间隔"],df["ED_volume4"],
          df["发病到随访5时间点时间间隔"],df["ED_volume5"],
          df["发病到随访6时间点时间间隔"],df["ED_volume6"],
          df["发病到随访7时间点时间间隔"],df["ED_volume7"],
          df["发病到随访8时间点时间间隔"],df["ED_volume8"],
          ],axis=1)
res

dfa=pd.concat([df["发病到首次影像检查时间间隔"],df["ED_volume0"]],axis=1)
dfb=pd.concat([df["发病到随访1时间点时间间隔"],df["ED_volume1"]],axis=1)
dfc=pd.concat([df["发病到随访2时间点时间间隔"],df["ED_volume2"]],axis=1)
dfd=pd.concat([df["发病到随访3时间点时间间隔"],df["ED_volume3"]],axis=1)
dfe=pd.concat([df["发病到随访4时间点时间间隔"],df["ED_volume4"]],axis=1)
dff=pd.concat([df["发病到随访5时间点时间间隔"],df["ED_volume5"]],axis=1)
dfg=pd.concat([df["发病到随访6时间点时间间隔"],df["ED_volume6"]],axis=1)
dfh=pd.concat([df["发病到随访7时间点时间间隔"],df["ED_volume7"]],axis=1)
dfi=pd.concat([df["发病到随访8时间点时间间隔"],df["ED_volume8"]],axis=1)
dfa.columns = ['时间间隔(x)','ED_volume(y)']
dfb.columns = ['时间间隔(x)','ED_volume(y)']
dfc.columns = ['时间间隔(x)','ED_volume(y)']
dfd.columns = ['时间间隔(x)','ED_volume(y)']
dfe.columns = ['时间间隔(x)','ED_volume(y)']
dff.columns = ['时间间隔(x)','ED_volume(y)']
dfg.columns = ['时间间隔(x)','ED_volume(y)']
dfh.columns = ['时间间隔(x)','ED_volume(y)']
dfi.columns = ['时间间隔(x)','ED_volume(y)']
df_all = pd.concat([dfa,dfb,dfc,dfd,dfe,dff,dfg,dfh,dfi])
df_not_NaN=df_all.dropna(axis=0,how='any')
df_not_NaN=df_not_NaN.sort_values(by=['时间间隔(x)'])

result_df = df_not_NaN.groupby('时间间隔(x)').agg({'ED_volume(y)': 'mean'})

result_df['ED_volume(y)'] = (result_df['ED_volume(y)'] - result_df['ED_volume(y)'].min()) / (result_df['ED_volume(y)'].max() - result_df['ED_volume(y)'].min())
result_df


! pip install statsmodels scikit-learn pyMetaheuristic

dfa=pd.concat([df['入院首次检查流水号'],df["发病到首次影像检查时间间隔"],df["ED_volume0"]],axis=1)
dfb=pd.concat([df['入院首次检查流水号'],df["发病到随访1时间点时间间隔"],df["ED_volume1"]],axis=1)
dfc=pd.concat([df['入院首次检查流水号'],df["发病到随访2时间点时间间隔"],df["ED_volume2"]],axis=1)
dfd=pd.concat([df['入院首次检查流水号'],df["发病到随访3时间点时间间隔"],df["ED_volume3"]],axis=1)
dfe=pd.concat([df['入院首次检查流水号'],df["发病到随访4时间点时间间隔"],df["ED_volume4"]],axis=1)
dff=pd.concat([df['入院首次检查流水号'],df["发病到随访5时间点时间间隔"],df["ED_volume5"]],axis=1)
dfg=pd.concat([df['入院首次检查流水号'],df["发病到随访6时间点时间间隔"],df["ED_volume6"]],axis=1)
dfh=pd.concat([df['入院首次检查流水号'],df["发病到随访7时间点时间间隔"],df["ED_volume7"]],axis=1)
dfi=pd.concat([df['入院首次检查流水号'],df["发病到随访8时间点时间间隔"],df["ED_volume8"]],axis=1)
dfa.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfb.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfc.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfd.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfe.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dff.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfg.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfh.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
dfi.columns = ['首次流水号','时间间隔(x)','ED_volume(y)']
df_all = pd.concat([dfa,dfb,dfc,dfd,dfe,dff,dfg,dfh,dfi])
dfres=df_all.dropna(axis=0,how='any')
dfres.reset_index(inplace=True)
dfres=dfres.drop(['index'],axis = 1)
dfres['ED_volume(y)'] = (dfres['ED_volume(y)'] - dfres['ED_volume(y)'].min()) / (dfres['ED_volume(y)'].max() - dfres['ED_volume(y)'].min())
print(dfres)

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from pyMetaheuristic.algorithm import whale_optimization_algorithm


n_splits = 5  
max_train_size = 360
tscv = TimeSeriesSplit(max_train_size=None,n_splits=n_splits)
ts=result_df

plt.figure(figsize=(12, 8))

mse_scores = [] 

for i, (train_index, test_index) in enumerate(tscv.split(ts)):
    train_data, test_data = ts.iloc[train_index], ts.iloc[test_index]

    def arima_optimizer(params):
        p, d, q = params
        p=int(p)
        d=int(d)
        q=int(q)
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test_data))
        mse = mean_squared_error(test_data, forecast)
        return mse

    parameters = {
    'hunting_party': 50,
    'min_values': (0, 0, 0),
    'max_values': (2, 2, 2),
    'iterations': 2,
    'spiral_param': 0.5,
    'verbose': True
    }

    woa = whale_optimization_algorithm(target_function = arima_optimizer,**parameters)
    best_params = woa[0][:-1]
    p, d, q = best_params
    print("best_params",int(p), int(d), int(q))

    model = ARIMA(train_data, order=(int(p), int(d), int(q)))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test_data))

    mse = mean_squared_error(test_data, forecast)
    mse_scores.append(mse)

    plt.subplot(2, 3, i + 1)
    plt.plot(train_data, label='train data')
    plt.plot(test_data, label='test data')
    plt.plot(test_data.index, forecast, label='Fit Data', linestyle='--')
    plt.title(f'Fold {i + 1}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

rmse_scores = [math.sqrt(mse) for mse in mse_scores]

for i, rmse in enumerate(rmse_scores):
    print(f'Fold {i+1}: RMSE = {rmse:.2f}')

average_rmse = np.mean(rmse_scores)
print(f'Average RMSE = {average_rmse:.2f}')

plt.tight_layout()
plt.show()