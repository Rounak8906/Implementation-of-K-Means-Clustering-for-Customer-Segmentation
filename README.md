# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Import data, inspect it, and handle missing values if any.
2. Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.
3. Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features.
4. Assign cluster labels to each data point.
5. Plot data points in a scatter plot, color-coded by cluster assignments for interpretation
## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ROUNAK SINGH
RegisterNumber:  212225240125

```


```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/Mall_Customers.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()
display(df.head())
```

```py
X = df.iloc[:, [3, 4]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```py
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()
```

```py
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

df['Cluster'] = y_kmeans
display(df.head())
```

```py
plt.figure(figsize=(10, 8))

colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = [
    'Sensible (Low Income, Low Spend)',
    'Standard (Average Income, Average Spend)',
    'Target (High Income, High Spend)',
    'Careless (Low Income, High Spend)',
    'Careful (High Income, Low Spend)'
]

for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=labels[i])

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', label='Centroids', edgecolor='black', marker='*')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
```

## Output:

<img width="802" height="580" alt="image" src="https://github.com/user-attachments/assets/b5984247-02c2-45c1-b0d5-3de95595068d" />

<img width="751" height="505" alt="image" src="https://github.com/user-attachments/assets/85102fbb-1602-48fd-b05c-ff680d04a34e" />

<img width="805" height="286" alt="image" src="https://github.com/user-attachments/assets/b86a43af-983a-47ae-8a1b-87528d2f7ca7" />


<img width="752" height="607" alt="image" src="https://github.com/user-attachments/assets/8b794891-ec94-4b7e-aea0-945518aece04" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
