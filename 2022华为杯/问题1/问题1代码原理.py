#所需环境
'''
! pip install xlrd --upgrade
！pip install pandas==1.2.0
!{sys.executable} -m pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple
!{sys.executable} -m pip install lightgbm -i https://pypi.tuna.tsinghua.edu.cn/simple
!{sys.executable} -m pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
'''
import pandas as pd
import warnings 
import chardet
import sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# 输出文件原格式
def  GetEncodingSheme(_filename):
    with open(_filename, 'rb') as file:
        buf = file.read()
    result = chardet.detect(buf)
    return result['encoding']

# 原格式转换为utf-8
def ChangeEncoding(_infilename, _outfilname, _encodingsheme='utf-8'):
    ifEncodeSheme = GetEncodingSheme(_infilename)
    with open(_infilename, 'r', encoding=ifEncodeSheme) as fr:
        tempContent = fr.read()
    with open(_outfilname, 'w', encoding=_encodingsheme) as fw:
        fw.write(tempContent)

# MAD:绝对中位差去极值
def filter_extreme_MAD(dataframe,n=3):
    for i in dataframe.columns:
        median = dataframe[i].quantile(0.5)
        new_median = ((dataframe[i] - median).abs()).quantile(0.50)
        max_range = median + n * new_median
        min_range = median - n * new_median
        dataframe[i]=pd.DataFrame(np.clip(dataframe[i].values, min_range, max_range),columns=None)
    return dataframe

# 3sigma 去极值
def filter_extreme_3sigma(dataframe,n=3):
    for i in dataframe.columns:
        mean=dataframe[i].mean()
        std=dataframe[i].std()
        max_range=mean+n*std
        min_range=mean-n*std
        dataframe[i] = pd.DataFrame(np.clip(dataframe[i].values, min_range, max_range), columns=None)
    return dataframe

# 百分位法 去极值
def filter_extreme_percentile(dataframe,min=0.025,max=0.975):  
    for i in dataframe.columns:
        Temp=dataframe[i].sort_values()
        q=Temp.quantile([min,max])
        dataframe[i] = pd.DataFrame(np.clip(dataframe[i].values, q.iloc[0], q.iloc[1]), columns=None)
    return dataframe

'''
**********读取数据**********
'''
filename = '附件15群落结构监测数据集.csv'
print('原格式为：', GetEncodingSheme(filename)) 
ChangeEncoding('附件15群落结构监测数据集.csv', '附件15群落结构监测数据集_new.csv', 'utf-8') #附件15群落结构监测数据集.csv 格式转化为utf-8
print('附件15群落结构监测数据集.csv已转换为utf-8编码')
f = open('附件15群落结构监测数据集_new.csv')

data  = pd.read_excel('附件14不同放牧强度土壤碳氮监测数据.xlsx')# 附件14
temp1 = pd.read_csv(f,encoding='utf-8')                       # 附件15
temp2 = pd.read_excel('附件3、土壤湿度2022—2012年.xls')        # 附件3
temp3 = pd.read_excel('附件4、土壤蒸发量2012—2022年.xls')      # 附件4
temp4 = pd.read_excel('附件6、植被指数-NDVI2012-2022年.xls')   # 附件6
temp5 = pd.read_excel('附件9、径流量2012-2022年.xlsx')         # 附件9

'''
**********数据预处理**********
'''
# 附件3 4 6 9按年份 月份合并
tempdf=pd.merge(temp2,temp3)
tempdf=pd.merge(tempdf,temp4)
tempdf=pd.merge(tempdf,temp5)
tempdf=tempdf[['月份', '年份', '经度(lon)','纬度(lat)', '10cm湿度(kg/m2)', '40cm湿度(kg/m2)',
       '100cm湿度(kg/m2)', '200cm湿度(kg/m2)', '土壤蒸发量(W/m2)',  '土壤蒸发量(mm)',
       '植被指数(NDVI)', '径流量(m3/s)', '径流量(m3)']]

# data生成新列inner 格式为：放牧小区（plot）-year 如G17-2012
data['inner']=data.apply(lambda x:x['放牧小区（plot）']+'-'+str(x['year']),axis=1)
# temp1生成新列inner 格式为：放牧小区Block-年份 如G6-2019
temp1['inner']=temp1.apply(lambda x:x['放牧小区Block']+'-'+str(x['年份']),axis=1)
# 以inner列为索引 合并data与temp1
data2=pd.merge(data,temp1,on='inner') #14297 rows × 24 columns
# 不同列缺失值数量 生殖苗过多 认为是无关项
data2.isnull().sum()
# 人为去除无关项：生殖苗 丛幅1 丛幅2 year(与年份重复) inner(与year plot重复) 日期 放牧小区block(与plot重复) 清洗后为(14297, 17)
data2=data2[[ '年份', '放牧小区（plot）', '放牧强度（intensity）', 'SOC土壤有机碳', 'SIC土壤无机碳',
       'STC土壤全碳', '全氮N', '土壤C/N比',  '轮次', '处理',  '植物种名',
       '植物群落功能群',  '重复', '营养苗',  '株/丛数',
       '鲜重(g)', '干重(g)']] 


# 将小区号进行one-hot编码 1代表data2本行是该小区号
pd.get_dummies(data2['放牧小区（plot）'],prefix='plot_')
# one-hot编码放在data2后
data2=pd.concat([data2,pd.get_dummies(data2['放牧小区（plot）'],prefix='plot_')],axis=1)
# 强度量化
# 对照（NG， 0羊/天/公顷 ）、轻度放牧强度（LGI， 2羊/天/公顷 ）、中度放牧强度（MGI，4羊/天/公顷 ）和重度放牧强度（HGI，8羊/天/公顷 ）
data2['放牧强度（intensity）']=data2['放牧强度（intensity）'].map(
{
    'NG':0,
    'LGI':1, 
    'MGI':2, 
    'HGI':3
})
# 种名变量太多 剔除 将种类化为列 存在则以1填充
data2=pd.concat([data2,pd.get_dummies(data2['轮次'],prefix='轮次_')],axis=1)#4种
data2=pd.concat([data2,pd.get_dummies(data2['处理'],prefix='处理_')],axis=1)#4种
data2=pd.concat([data2,pd.get_dummies(data2['植物群落功能群'],prefix='功能群_')],axis=1)#4种
print('挑选完并量化完的数据：',data2.columns)
#挑选
data2_process=data2[['年份', '放牧强度（intensity）', 'SOC土壤有机碳', 'SIC土壤无机碳',
       'STC土壤全碳', '全氮N', '土壤C/N比',  '营养苗',
       '株/丛数', '鲜重(g)', '干重(g)', 'plot__G11', 'plot__G12', 'plot__G13',
       'plot__G16', 'plot__G17', 'plot__G18', 'plot__G19', 'plot__G20',
       'plot__G21', 'plot__G6', 'plot__G8', 'plot__G9', '轮次__牧前', '轮次__第一轮牧后',
       '轮次__第三轮牧后', '轮次__第二轮牧后', '轮次__第四轮牧后', '处理__中牧（6天）', '处理__无牧（0天）',
       '处理__轻牧（3天）', '处理__重牧（12天）', '功能群__AB', '功能群__PB', '功能群__PF',
       '功能群__PR']]
#按年份合并 
df=pd.merge(tempdf,data2_process,on=['年份'])#(171564, 48)
#数据清洗
df.isnull().sum()/df.shape[0]#计算缺失比例
for i in ['营养苗','株/丛数', '鲜重(g)']:#用均值填充缺失值
    df[i]=df[i].fillna(df[i].mean())
df=filter_extreme_3sigma(df)#去极值

'''
**********模型预测**********
'''
# 'SOC土壤有机碳', 'SIC土壤无机碳','STC土壤全碳', '全氮N', '土壤C/N比',——化学性质
# '营养苗','株/丛数', '鲜重(g)', '干重(g)',——植物生物量
# '10cm湿度(kg/m2)', '40cm湿度(kg/m2)','100cm湿度(kg/m2)', '200cm湿度(kg/m2)'——土壤湿度
# 'plot__G11', 'plot__G12', 'plot__G13','plot__G16', 'plot__G17', 'plot__G18', 'plot__G19', 'plot__G20','plot__G21', 'plot__G6', 'plot__G8', 'plot__G9',——放牧小区(plot)(策略)
# '轮次__牧前', '轮次__第一轮牧后', '轮次__第三轮牧后', '轮次__第二轮牧后', '轮次__第四轮牧后',——第几轮(策略)
#  '处理__中牧（6天）', '处理__无牧（0天）','处理__轻牧（3天）', '处理__重牧（12天）'——放牧强度 (应该和那个合并)

X=df[['SOC土壤有机碳', 'SIC土壤无机碳','STC土壤全碳', '全氮N', '土壤C/N比',  
   '营养苗','株/丛数', '鲜重(g)', '干重(g)',
   'plot__G11', 'plot__G12', 'plot__G13','plot__G16', 'plot__G17', 'plot__G18', 'plot__G19', 'plot__G20','plot__G21', 'plot__G6', 'plot__G8', 'plot__G9', '轮次__牧前', '轮次__第一轮牧后',
   '轮次__第三轮牧后', '轮次__第二轮牧后', '轮次__第四轮牧后', 
   '处理__中牧（6天）', '处理__无牧（0天）','处理__轻牧（3天）', '处理__重牧（12天）', 
   '功能群__AB', '功能群__PB', '功能群__PF','功能群__PR',
   '10cm湿度(kg/m2)', '40cm湿度(kg/m2)','100cm湿度(kg/m2)', '200cm湿度(kg/m2)']]

y=df['放牧强度（intensity）']#对照0 轻1 中2 重3
# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    
#逻辑 回归
model = LogisticRegression()

model.fit(x_train, y_train)
print('逻辑 回归')
print(classification_report(model.predict(x_test),y_test))


# Randomforest分类

model = RandomForestClassifier()

model.fit(x_train, y_train)
print('Randomforest分类')
print(classification_report(model.predict(x_test),y_test))

# lgbm分类
model = lgb.LGBMClassifier()
model.fit(x_train, y_train)
print('lgbm分类')
print(classification_report(model.predict(x_test),y_test))


# XGboost分类
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
print('XGboost分类')
print(classification_report(model.predict(x_test),y_test))

# 结果存储
df.to_csv('Q1.data.csv',index=None)

#逻辑 回归
model = LogisticRegression()

model.fit(X, y)
print('逻辑 回归')
print(classification_report(model.predict(X),y))
print(model.coef_)
print(model.intercept_)