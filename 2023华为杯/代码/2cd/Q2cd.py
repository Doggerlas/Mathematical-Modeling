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

res["speed1"]=(res["ED_volume1"]-res["ED_volume0"])/(res["发病到随访1时间点时间间隔"]-res["发病到首次影像检查时间间隔"])
res["speed2"]=(res["ED_volume2"]-res["ED_volume1"])/(res["发病到随访2时间点时间间隔"]-res["发病到随访1时间点时间间隔"])
res["speed3"]=(res["ED_volume3"]-res["ED_volume2"])/(res["发病到随访3时间点时间间隔"]-res["发病到随访2时间点时间间隔"])
res["speed4"]=(res["ED_volume4"]-res["ED_volume3"])/(res["发病到随访4时间点时间间隔"]-res["发病到随访3时间点时间间隔"])
res["speed5"]=(res["ED_volume5"]-res["ED_volume4"])/(res["发病到随访5时间点时间间隔"]-res["发病到随访4时间点时间间隔"])
res["speed6"]=(res["ED_volume6"]-res["ED_volume5"])/(res["发病到随访6时间点时间间隔"]-res["发病到随访5时间点时间间隔"])
res["speed7"]=(res["ED_volume7"]-res["ED_volume6"])/(res["发病到随访7时间点时间间隔"]-res["发病到随访6时间点时间间隔"])
res["speed8"]=(res["ED_volume8"]-res["ED_volume7"])/(res["发病到随访8时间点时间间隔"]-res["发病到随访7时间点时间间隔"])
res

table1 = pd.read_excel("表1-患者列表及临床信息.xlsx")
table1

table1 = pd.read_excel("表1-患者列表及临床信息.xlsx")



treatment_columns = table1.columns[16:23].tolist()  


merged_data_corrected = pd.merge(res[[ '入院首次检查流水号', 'speed1','speed2','speed3','speed4','speed5','speed6','speed7','speed8']],
                  table1[['入院首次检查流水号'] + treatment_columns],
                  left_on='入院首次检查流水号', right_on='入院首次检查流水号', how='left')


filtered_data_corrected=merged_data_corrected.head(100)
filtered_data_corrected

filtered_data_corrected.to_excel('治疗方法与水肿体积增长速度关系.xlsx')


! pip install mplfonts
import matplotlib.pyplot as plt
from mplfonts.bin.cli import init

from mplfonts import use_font
use_font('Noto Serif CJK SC')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

spearman_corr = filtered_data_corrected[['speed1',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="gist_rainbow" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed2',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="rainbow" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed3',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="Purples" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed4',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="hot_r" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed5',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="viridis" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed6',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu_r" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed7',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="autumn" ,annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['speed8',
                     '脑室引流','止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="OrRd" ,annot=True, fmt=".2f", linewidths=0.5)

import pandas as pd
table1 = pd.read_excel("表1-患者列表及临床信息.xlsx")
table1.head()

table2 = pd.read_excel("表2-患者影像信息血肿及水肿的体积及位置.xlsx")
table2.head()


treatment_columns = table1.columns[16:23].tolist()  

merged_data_corrected = pd.merge(table2[['ID', '入院首次检查流水号','HM_volume1','HM_volume2','HM_volume3','HM_volume4','HM_volume5','HM_volume6','HM_volume7','HM_volume8',
                      'ED_volume1','ED_volume2','ED_volume3','ED_volume4','ED_volume5','ED_volume6','ED_volume7','ED_volume8']],
                  table1[['入院首次检查流水号'] + treatment_columns],
                  left_on='入院首次检查流水号', right_on='入院首次检查流水号', how='left')


filtered_data_corrected = merged_data_corrected.loc[merged_data_corrected['ID'].str[-3:].astype(int) < 100]

filtered_data_corrected.head()

spearman_corr = filtered_data_corrected[['ED_volume1', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume1', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume1', 'ED_volume1']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume2', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume2', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume2', 'ED_volume2']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume3', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume3', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume3', 'ED_volume3']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume4', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume4', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume4', 'ED_volume4']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume5', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume5', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume5', 'ED_volume5']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume6', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume6', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume6', 'ED_volume6']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume7', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume7', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume7', 'ED_volume7']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)

spearman_corr = filtered_data_corrected[['ED_volume8', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume8', '脑室引流',
       '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr,cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)


spearman_corr = filtered_data_corrected[['HM_volume8', 'ED_volume8']].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap="hot", annot=True, fmt=".2f", linewidths=0.5)