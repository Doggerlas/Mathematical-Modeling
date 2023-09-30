import pandas as pd
table1 = pd.read_excel("表1-患者列表及临床信息.xlsx")
table1.head()

table2 = pd.read_excel("表2-患者影像信息血肿及水肿的体积及位置.xlsx")
table2.head()


merged_data = pd.merge(table2, table1[['入院首次影像检查流水号', '发病到首次影像检查时间间隔']], left_on='入院首次影像检查流水号', right_on='入院首次影像检查流水号', how='left')

merged_data.drop(3, inplace=True)
merged_data.drop(5, inplace=True)
merged_data.reset_index(inplace=True)
merged_data=merged_data.drop(['index'],axis = 1)

filtered_data = merged_data.loc[merged_data['ID'].str[-3:].astype(int) < 101, ['ID','入院首次影像检查流水号', 'ED_volume', '发病到首次影像检查时间间隔']]
filtered_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

! pip install mplfonts
import matplotlib.pyplot as plt
from mplfonts.bin.cli import init
init() 
from mplfonts import use_font
use_font('Noto Serif CJK SC')


X = filtered_data['发病到首次影像检查时间间隔'].values.reshape(-1, 1)
y = filtered_data['ED_volume'].values


X_log = np.log(X)
y_log_transformed = np.log(y)


degrees = list(range(1, 11))
models = []
r2_scores = []

for degree in degrees:
    
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    
    r2 = r2_score(y, y_pred)
    print(f"{degree}次多项式拟合的R2为:{r2}")

    models.append(model)
    r2_scores.append(r2)

    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='实际数据')
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X_range)), color='purple', label=f'拟合（{degree}次多项式）')
    plt.xlabel('发病到首次影像检查时间间隔')
    plt.ylabel('ED_volume')
    plt.legend()
    plt.title(f'拟合（{degree}次多项式） (R^2 = {max(r2_scores):.4f})')
    plt.grid(True)
    plt.show()


model_exp = LinearRegression()
model_exp.fit(X.reshape(-1, 1), np.log(y))
a_exp, b_exp = model_exp.coef_[0], model_exp.intercept_


model_log = LinearRegression()
model_log.fit(X_log.reshape(-1, 1), y_log_transformed)
a_log, b_log = model_log.coef_[0], model_log.intercept_


x_fit = np.linspace(min(X), max(X), 100)
y_fit_exp = np.exp(a_exp * x_fit + b_exp)
y_fit_log = np.exp(a_log * np.log(x_fit) + b_log)


plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.scatter(X, y, label='原始数据')
plt.plot(x_fit, y_fit_exp, 'r', label='指数拟合')
plt.legend()
plt.title('指数拟合')


plt.subplot(1, 2, 2)
plt.scatter(X, y, label='原始数据')
plt.plot(x_fit, y_fit_log, 'g', label='对数拟合')
plt.legend()
plt.title('对数拟合')

plt.tight_layout()
plt.show()

y_pred = model_exp.predict(X.reshape(-1, 1))
r2 = r2_score(np.log(y), y_pred)
models.append(model_exp)
r2_scores.append(r2)
print(f"对数拟合的R2为:{r2}")

y_pred = model_log.predict(X_log.reshape(-1, 1))
r2 = r2_score(y_log_transformed, y_pred)
models.append(model_log)
r2_scores.append(r2)
print(f"指数拟合的R2为:{r2}")

best_degree = degrees[np.argmax(r2_scores)]
best_model = models[np.argmax(r2_scores)]


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='实际数据')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(X_range, best_model.predict(PolynomialFeatures(degree=best_degree, include_bias=False).fit_transform(X_range)), color='red', label=f'最佳拟合（{best_degree}次多项式）')
plt.xlabel('发病到首次影像检查时间间隔')
plt.ylabel('ED_volume')
plt.legend()
plt.title(f'最佳多项式拟合（{best_degree}次多项式） (R^2 = {max(r2_scores):.4f})')
plt.grid(True)
plt.show()


best_degree, max(r2_scores)

y


y_pred = best_model.predict(PolynomialFeatures(degree=best_degree, include_bias=False).fit_transform(X_range))
residuals = y - y_pred


residuals_df = filtered_data.copy()
residuals_df['残差（全体）'] = residuals
residuals_df[['ID','入院首次影像检查流水号', 'ED_volume', '发病到首次影像检查时间间隔', '残差（全体）']]
residuals_df

residuals_df.to_excel('残差（全体）.xlsx')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn import metrics


clustering_data = filtered_data[['发病到首次影像检查时间间隔', 'ED_volume']]



scaler = StandardScaler()


clustering_data_scaled = scaler.fit_transform(clustering_data)


inertia = []  
K = range(1, 10)  


for k in K:
    
    kmeans = KMeans(n_clusters=k, random_state=42).fit(clustering_data_scaled)

    
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))  
plt.plot(K, inertia, 'bx-')  
plt.xlabel('聚类数量 (k)')  
plt.ylabel('惯性值')  
plt.title('肘部法则确定最佳k值')  
plt.grid(True)  
plt.show()  


LK = []
CH = []
DB = []


kmeans = KMeans(n_clusters=3, random_state=42).fit(clustering_data_scaled)  
labels = kmeans.labels_  


filtered_data['Cluster'] = labels  

print("Kmeans")
LK.append(metrics.silhouette_score(clustering_data_scaled,labels,metric='euclidean'))
CH.append(metrics.calinski_harabasz_score(clustering_data_scaled,labels))
DB.append(metrics.davies_bouldin_score(clustering_data_scaled,labels))
print(LK)
print(CH)
print(DB)


plt.figure(figsize=(10, 6))  
for i in range(3):
    cluster_data = filtered_data[filtered_data['Cluster'] == i]  
    plt.scatter(cluster_data['发病到首次影像检查时间间隔'], cluster_data['ED_volume'], label=f'簇 {i+1}')  

plt.xlabel('发病到首次影像检查时间间隔')  
plt.ylabel('ED_volume')  
plt.legend()  
plt.title('Kmeans聚类结果')  
plt.grid(True)  
plt.show()  

from sklearn.cluster import AffinityPropagation,MiniBatchKMeans,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN,Birch

minikmeans = MiniBatchKMeans(n_clusters=3, random_state=42).fit(clustering_data_scaled)  
labels = minikmeans.labels_  


filtered_data['Cluster'] = labels  

print("MiniBatchKMeans")
LK.append(metrics.silhouette_score(clustering_data_scaled,labels,metric='euclidean'))
CH.append(metrics.calinski_harabasz_score(clustering_data_scaled,labels))
DB.append(metrics.davies_bouldin_score(clustering_data_scaled,labels))
print(LK)
print(CH)
print(DB)


plt.figure(figsize=(10, 6))  
for i in range(3):
    cluster_data = filtered_data[filtered_data['Cluster'] == i]  
    plt.scatter(cluster_data['发病到首次影像检查时间间隔'], cluster_data['ED_volume'], label=f'簇 {i+1}')  

plt.xlabel('发病到首次影像检查时间间隔')  
plt.ylabel('ED_volume')  
plt.legend()  
plt.title('MiniBatchKMeans聚类结果')  
plt.grid(True)  
plt.show()  

from sklearn.cluster import AffinityPropagation,MiniBatchKMeans,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN,Birch

sc = SpectralClustering(n_clusters=3, random_state=42).fit(clustering_data_scaled)  
labels = sc.labels_  


filtered_data['Cluster'] = labels  

print("SpectralClustering")
LK.append(metrics.silhouette_score(clustering_data_scaled,labels,metric='euclidean'))
CH.append(metrics.calinski_harabasz_score(clustering_data_scaled,labels))
DB.append(metrics.davies_bouldin_score(clustering_data_scaled,labels))
print(LK)
print(CH)
print(DB)


plt.figure(figsize=(10, 6))  
for i in range(3):
    cluster_data = filtered_data[filtered_data['Cluster'] == i]  
    plt.scatter(cluster_data['发病到首次影像检查时间间隔'], cluster_data['ED_volume'], label=f'簇 {i+1}')  

plt.xlabel('发病到首次影像检查时间间隔')  
plt.ylabel('ED_volume')  
plt.legend()  
plt.title('SpectralClustering聚类结果')  
plt.grid(True)  
plt.show()  

from sklearn.cluster import AffinityPropagation,MiniBatchKMeans,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN,Birch

b = Birch(n_clusters=3).fit(clustering_data_scaled)  
labels = b.labels_  


filtered_data['Cluster'] = labels  

print("Birch")
LK.append(metrics.silhouette_score(clustering_data_scaled,labels,metric='euclidean'))
CH.append(metrics.calinski_harabasz_score(clustering_data_scaled,labels))
DB.append(metrics.davies_bouldin_score(clustering_data_scaled,labels))
print(LK)
print(CH)
print(DB)


plt.figure(figsize=(10, 6))  
for i in range(3):
    cluster_data = filtered_data[filtered_data['Cluster'] == i]  
    plt.scatter(cluster_data['发病到首次影像检查时间间隔'], cluster_data['ED_volume'], label=f'簇 {i+1}')  

plt.xlabel('发病到首次影像检查时间间隔')  
plt.ylabel('ED_volume')  
plt.legend()  
plt.title('Birch的聚类结果')  
plt.grid(True)  
plt.show()  

from sklearn.cluster import AffinityPropagation,MiniBatchKMeans,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN,Birch

ac = AgglomerativeClustering(n_clusters=3).fit(clustering_data_scaled)  
labels = ac.labels_  


filtered_data['Cluster'] = labels  

print("AgglomerativeClustering")
LK.append(metrics.silhouette_score(clustering_data_scaled,labels,metric='euclidean'))
CH.append(metrics.calinski_harabasz_score(clustering_data_scaled,labels))
DB.append(metrics.davies_bouldin_score(clustering_data_scaled,labels))
print(LK)
print(CH)
print(DB)


plt.figure(figsize=(10, 6))  
for i in range(3):
    cluster_data = filtered_data[filtered_data['Cluster'] == i]  
    plt.scatter(cluster_data['发病到首次影像检查时间间隔'], cluster_data['ED_volume'], label=f'簇 {i+1}')  

plt.xlabel('发病到首次影像检查时间间隔')  
plt.ylabel('ED_volume')  
plt.legend()  
plt.title('AgglomerativeClustering的聚类结果')  
plt.grid(True)  
plt.show()  


cluster_models = []  
cluster_r2_scores = []  
dd=pd.DataFrame()  

for i in range(3):
    cluster_data = filtered_data[filtered_data['Cluster'] == i]  

    
    X = cluster_data['发病到首次影像检查时间间隔'].values.reshape(-1, 1)  
    y = cluster_data['ED_volume'].values  

    
    degrees = list(range(1, 6))  
    models = []  
    r2_scores = []  

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)

        r2 = r2_score(y, y_pred)

        models.append(model)
        r2_scores.append(r2)

    best_degree = degrees[np.argmax(r2_scores)]
    best_model = models[np.argmax(r2_scores)]

    
    coefficients = best_model.coef_
    intercept = best_model.intercept_

    
    formula_terms = [f"{coefficients[i]:.4f}x^{i+1}" for i in range(len(coefficients))]
    formula = "y = " + " + ".join(formula_terms) + f" + {intercept:.4f}"

    print(formula)

    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='实际数据')
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_range, best_model.predict(PolynomialFeatures(degree=best_degree, include_bias=False).fit_transform(X_range)), color='red', label=f'最佳拟合（{best_degree}次多项式）')
    plt.xlabel('发病到首次影像检查时间间隔')
    plt.ylabel('ED_volume')
    plt.legend()
    plt.title(f'最佳多项式拟合（{best_degree}次多项式） (R^2 = {max(r2_scores):.4f})')
    plt.grid(True)
    plt.show()

    cluster_data['predic']=best_model.predict(PolynomialFeatures(degree=best_degree, include_bias=False).fit_transform(X))  
    try:
        dd=pd.concat([dd,cluster_data])  
    except:
        pass

dd=dd.sort_values('ID')
dd


residuals = dd['ED_volume'] - dd['predic']


residuals_df = filtered_data.copy()  
residuals_df['残差（亚组）'] = residuals  


residuals_df[['ID', 'ED_volume', '发病到首次影像检查时间间隔', '残差（亚组）']]

residuals_df.to_excel('残差（亚组）.xlsx')

r2_scores=r2_score(dd['ED_volume'],dd['predic'])
r2_scores

plt.figure(figsize=(10, 6))
plt.scatter(dd['发病到首次影像检查时间间隔'], dd['ED_volume'], color='blue', label='Actual data')
plt.scatter(dd['发病到首次影像检查时间间隔'], dd['predic'], color='red')
plt.xlabel('发病到首次影像检查时间间隔')
plt.ylabel('ED_volume')
plt.legend()

plt.grid(True)
plt.show()