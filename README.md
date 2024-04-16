# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: POOJA.S
RegisterNumber:  212223040146
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8 (1).csv')
data
X=data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=5
kmeans =KMeans(n_clusters=k)
kmeans.fit(X)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
                             color=colors[i],label=f'Cluster {i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```


## Output:
![Screenshot 2024-04-16 162433](https://github.com/Shubhavi17/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150005085/194597ab-a734-4435-ae4d-19e6e2817c29)
![Screenshot 2024-04-16 162443](https://github.com/Shubhavi17/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150005085/b9eeaf53-6f79-49b6-97ae-e295418e1b1a)
![Screenshot 2024-04-16 162454](https://github.com/Shubhavi17/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150005085/53959458-8d9d-496a-98bf-8f245a8788f3)
![Screenshot 2024-04-16 184328](https://github.com/Shubhavi17/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150005085/ea94940d-3c3f-436d-8f87-68ecd6b70e24)
![Screenshot 2024-04-16 162506](https://github.com/Shubhavi17/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150005085/74bc2bb9-e5ad-4621-831c-df2bfdb2f7db)
![Screenshot 2024-04-16 162506]![Screenshot 2024-04-16 162524](https://github.com/Shubhavi17/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/150005085/b3cd077f-a34b-49eb-bb5a-446bfaadd43f)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
