import xlrd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


f_name = '问题2数据.xlsx'
data = xlrd.open_workbook(f_name)
table = data.sheets()[0]
df = pd.DataFrame(columns=['年份', '月份', '10cm湿度', '40cm湿度', '100cm湿度', '200cm湿度'])  # 存储最终结果

for mon in [0, 1, 2]:  # 1~3月
    # 蒸发量 气温 降水量 气压 风速 
    zf = [table.cell_value(i, 2) for i in range(mon * 11 + 1, mon * 11 + 12)]
    qw = [table.cell_value(i, 3) for i in range(mon * 11 + 1, mon * 11 + 12)]
    js = [table.cell_value(i, 4) for i in range(mon * 11 + 1, mon * 11 + 12)]
    qy = [table.cell_value(i, 5) for i in range(mon * 11 + 1, mon * 11 + 12)]
    fs = [table.cell_value(i, 6) for i in range(mon * 11 + 1, mon * 11 + 12)]
    Factor = np.array([zf, qw, js, qy, fs]).transpose()# 统称为气候因素
    # 不同深度的湿度
    sd10 = np.array([table.cell_value(i, 7) for i in range(mon * 11 + 1, mon * 11 + 12)]).flatten()
    sd40 = np.array([table.cell_value(i, 8) for i in range(mon * 11 + 1, mon * 11 + 12)]).flatten()
    sd100 = np.array([table.cell_value(i, 9) for i in range(mon * 11 + 1, mon * 11 + 12)]).flatten()
    sd200 = np.array([table.cell_value(i, 10) for i in range(mon * 11 + 1, mon * 11 + 12)]).flatten()

    # 该月份第几次出现
    time = np.array(range(len(zf))).reshape(-1, 1)
    #t_test = np.array(range(12)).reshape(-1, 1)
    time_test = np.array(range(len(zf)+1)).reshape(-1, 1)

    # 随机森林构造月份出现的次数与气候因素之间的关系模型
    zf_model = RandomForestRegressor()
    zf_model.fit(time, zf)  # 月份出现的次数与蒸发量之间关系模型
    qw_model = RandomForestRegressor()
    qw_model.fit(time, qw)  # 月份出现的次数与气温之间关系模型
    js_model = RandomForestRegressor()
    js_model.fit(time, js)  # 月份出现的次数与降水量之间关系模型
    qy_model = RandomForestRegressor()
    qy_model.fit(time, qy)  # 月份出现的次数与气压之间关系模型
    fs_model = RandomForestRegressor()
    fs_model.fit(time, fs)  # 月份出现的次数与风速之间关系模型

    # 随机森林构造气候因素与不同深度土壤湿度之间的关系模型
    sd10_model = RandomForestRegressor()
    sd10_model.fit(Factor, sd10)  # 10cm湿度预测模型
    sd40_model = RandomForestRegressor()
    sd40_model.fit(Factor, sd40)  # 40cm湿度预测模型
    sd100_model = RandomForestRegressor()
    sd100_model.fit(Factor, sd100)  # 100cm湿度预测模型
    sd200_model = RandomForestRegressor()
    sd200_model.fit(Factor, sd200)  # 200cm湿度预测模型

    # 预测未来蒸发量 气温 降水量 气压 风速 time_test是0~11 指该月份在2012-2023共12年的时间序列 预测结果只取最后一年的 即2023年的 其他年的肯定和已知数据有偏差 很正常 
    zf_test = zf_model.predict(time_test)
    qw_test = qw_model.predict(time_test)
    js_test = js_model.predict(time_test)
    qy_test = qy_model.predict(time_test)
    fs_test = fs_model.predict(time_test)
    Factor_test = np.array([zf_test, qw_test, js_test, qy_test, fs_test]).transpose()

    # 预测未来的湿度
    sd10_test = sd10_model.predict(Factor_test)
    sd40_test = sd40_model.predict(Factor_test)
    sd100_test = sd100_model.predict(Factor_test)
    sd200_test = sd200_model.predict(Factor_test)

    # 输出结果
    s='当前是2023年第'+repr(mon+1)+'月'
    print(s)
    print('输出不同深度的湿度值为', [2023,mon+1,sd10_test[-1], sd40_test[-1], sd100_test[-1], sd200_test[-1]] )
    df.loc[len(df)] =[2023,mon+1,sd10_test[-1], sd40_test[-1], sd100_test[-1], sd200_test[-1]]

Results = {}  
for mon in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # 4~12月
    # 蒸发量 气温 降水量 气压 风速 
    zf = [table.cell_value(i, 2) for i in range(mon * 10 + 4, mon * 10 + 14)]
    qw = [table.cell_value(i, 3) for i in range(mon * 10 + 4, mon * 10 + 14)]
    js = [table.cell_value(i, 4) for i in range(mon * 10 + 4, mon * 10 + 14)]
    qy = [table.cell_value(i, 5) for i in range(mon * 10 + 4, mon * 10 + 14)]
    fs = [table.cell_value(i, 6) for i in range(mon * 10 + 4, mon * 10 + 14)]
    Factor = np.array([zf, qw, js, qy, fs]).transpose()# 统称为气候因素
    # 不同深度的湿度
    sd10 = np.array([table.cell_value(i, 7) for i in range(mon * 10 + 4, mon * 10 + 14)]).flatten()
    sd40 = np.array([table.cell_value(i, 8) for i in range(mon * 10 + 4, mon * 10 + 14)]).flatten()
    sd100 = np.array([table.cell_value(i, 9) for i in range(mon * 10 + 4, mon * 10 + 14)]).flatten()
    sd200 = np.array([table.cell_value(i, 10) for i in range(mon * 10 + 4, mon * 10 + 14)]).flatten()

    # 该月份第几次出现
    time = np.array(range(len(zf))).reshape(-1, 1)
    #t_test = np.array(range(11)).reshape(-1, 1)
    time_test = np.array(range(len(zf)+2)).reshape(-1, 1)#+2是因为4-12月份只到2021年 2022 2023的没有 所以需要输出12年的预测值 取最后两年

    # 随机森林构造月份出现的次数与气候因素之间的关系模型
    zf_model = RandomForestRegressor()
    zf_model.fit(time, zf)  # 月份出现的次数与蒸发量之间关系模型
    qw_model = RandomForestRegressor()
    qw_model.fit(time, qw)  # 月份出现的次数与气温之间关系模型
    js_model = RandomForestRegressor()
    js_model.fit(time, js)  # 月份出现的次数与降水量之间关系模型
    qy_model = RandomForestRegressor()
    qy_model.fit(time, qy)  # 月份出现的次数与气压之间关系模型
    fs_model = RandomForestRegressor()
    fs_model.fit(time, fs)  # 月份出现的次数与风速之间关系模型

    # 随机森林构造气候因素与不同深度土壤湿度之间的关系模型
    sd10_model = RandomForestRegressor()
    sd10_model.fit(Factor, sd10)  # 10cm湿度预测模型
    sd40_model = RandomForestRegressor()
    sd40_model.fit(Factor, sd40)  # 40cm湿度预测模型
    sd100_model = RandomForestRegressor()
    sd100_model.fit(Factor, sd100)  # 100cm湿度预测模型
    sd200_model = RandomForestRegressor()
    sd200_model.fit(Factor, sd200)  # 200cm湿度预测模型

     # 预测未来蒸发量 气温 降水量 气压 风速 time_test是0~11 指该月份在2012-2023共12年的时间序列 预测结果只取最后一年的 即2023年的 其他年的肯定和已知数据有偏差 很正常 
    zf_test = zf_model.predict(time_test)
    qw_test = qw_model.predict(time_test)
    js_test = js_model.predict(time_test)
    qy_test = qy_model.predict(time_test)
    fs_test = fs_model.predict(time_test)
    Factor_test = np.array([zf_test, qw_test, js_test, qy_test, fs_test]).transpose()

    # 预测未来的湿度
    sd10_test = sd10_model.predict(Factor_test)
    sd40_test = sd40_model.predict(Factor_test)
    sd100_test = sd100_model.predict(Factor_test)
    sd200_test = sd200_model.predict(Factor_test)

    # 输出结果
    Results[str(mon)] = [[], []]
    for i in [-2, -1]:#倒数前两个结果是预测的2022 2023年的4-12月份数据
        if i == -2:
            Results[str(mon)][i + 2] =[2022,mon+1,sd10_test[i], sd40_test[i], sd100_test[i], sd200_test[i]]
        else:
            Results[str(mon)][i + 2] =[2023,mon+1,sd10_test[i], sd40_test[i], sd100_test[i], sd200_test[i]]

for mon in range(3, 12):
    s='当前是2022年第'+repr(mon+1)+'月'
    print(s)
    df.loc[len(df)] = Results[str(mon)][0]
    print('输出不同深度的湿度值为', Results[str(mon)][0] )
for mon in range(3, 12):
    s='当前是2023年第'+repr(mon+1)+'月'
    print(s)
    df.loc[len(df)] = Results[str(mon)][1]
    print('输出不同深度的湿度值为', Results[str(mon)][1] )

df.to_csv('第2题结果.csv', sep=',')
