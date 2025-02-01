import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd



df = pd.read_csv("income_clustering.csv")
df = df [[ "Age", "Income($)"]]
scaler = StandardScaler()
sc_df = scaler.fit_transform(df)

plt.scatter(df['Age'],df['Income($)'], color="r",marker ="*")
plt.title ("Age vs Income")
plt.xlabel ("Age")
plt.ylabel ("Income")

k_range = range(1,11)
sse = []
for k in k_range:
    kmn = KMeans(n_clusters=k)
    kmn.fit(sc_df)
    sse.append(kmn.inertia_)

plt.plot(k_range,sse, color="r",marker=".")
    kmn = KMeans (n_clusters =3)
clusters = kmn.fit_predict(sc_df)

df ['clusters'] = clusters
df.head()

cl1=df[df['clusters']==0]
cl2=df[df['clusters']==1]
cl3=df[df['clusters']==2]
centroids = scaler.inverse_transform(kmn.cluster_centers_)


plt.title("k-means income clustering vs age")
plt.xlabel("Age")
plt.ylabel("Income($)")
plt.scatter(cl1['Age'],cl1['Income($)'],color="r",marker="*",label="Cluster 1")
plt.scatter(cl2['Age'],cl2['Income($)'],color="b",marker="+",label="Cluster 2")
plt.scatter(cl3['Age'],cl3['Income($)'],color="g",marker="v",label="Cluster 3")
plt.scatter(centroids[:,0],centroids[:,1],s=200,color="Black",label="Centroids",marker="+")
plt.legend()
