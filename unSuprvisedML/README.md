# Unsupervised Machine Learning

This section contains implementations of unsupervised learning algorithms that discover hidden patterns in unlabeled data.

## 📂 Modules

### 1. K-means Clustering
Partition data into K clusters based on feature similarity.

**Contents:**
- `Mall_customers.ipynb` - Real-world customer segmentation example
- `app/` - Practical applications

**What you'll learn:**
- Initialization strategies
- Centroid updates
- Convergence criteria
- Choosing optimal K (Elbow method, Silhouette score)
- Customer segmentation
- Limitations and when to use alternatives

**Key Concepts:**
- Distance metrics (Euclidean)
- Iterative clustering process
- Inertia and silhouette coefficient
- Application in customer analytics

### 2. Hierarchical Clustering
Build a hierarchy of clusters using agglomerative or divisive methods.

**What you'll learn:**
- Agglomerative (bottom-up) clustering
- Linkage criteria (Single, Complete, Average, Ward)
- Dendrogram interpretation
- Distance matrix computation
- Cutting dendrograms to get clusters
- Comparison with K-means

**Key Concepts:**
- Cluster linkage methods
- Dendrogram visualization
- Hierarchical structure
- No need to pre-specify cluster count

### 3. DBSCAN
Density-Based Spatial Clustering of Applications with Noise.

**What you'll learn:**
- Density-based clustering concepts
- Epsilon and min_samples parameters
- Core points, border points, and noise
- Handling arbitrary cluster shapes
- Advantages over K-means
- Finding optimal parameters

**Key Concepts:**
- Reachability and density connectivity
- Automatic noise detection
- Arbitrary shaped clusters
- Scalability considerations

### 4. Feature Extraction
Transform raw data into meaningful features using dimensionality reduction.

**Contents Include:**
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Factor Analysis
- Manifold learning techniques

**What you'll learn:**
- Reducing dimensionality while preserving information
- Variance explained
- Loadings and components
- Visualization of high-dimensional data
- Computational efficiency improvements

**Key Techniques:**
- PCA - Linear dimension reduction
- t-SNE - Non-linear visualization
- UMAP - Scalable dimension reduction
- Autoencoders - Neural network-based extraction

### 5. Feature Selection
Select the most relevant features for your model.

**What you'll learn:**
- Filter methods (correlation, variance)
- Wrapper methods (recursive elimination)
- Embedded methods (feature importance)
- Univariate selection
- Multivariate selection
- Handling high-dimensional data

**Methods Covered:**
- SelectKBest
- Mutual information
- Chi-square
- Recursive Feature Elimination
- L1-based selection

### 6. Anomaly Detection
Identify unusual or abnormal data points.

**What you'll learn:**
- Statistical methods
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoders for anomaly detection
- Evaluation metrics for anomaly detection

**Applications:**
- Fraud detection
- Network intrusion detection
- Quality control
- Health monitoring
- Sensor data analysis

## 🎯 Learning Path

1. **Start here:** K-means-clustering - Basic clustering concept
2. **Then:** Hierarchical-clustering - Alternative approach
3. **Next:** DBSCAN - Density-based perspective
4. **Advanced:** Feature_Extraction - Dimensionality reduction
5. **Optimization:** Feature_Selection - Improve efficiency
6. **Special Cases:** anomaly-detection - Find outliers

## 💻 Requirements

```python
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
jupyter
```

## 📊 Algorithms Comparison

| Algorithm | Type | Best For | Complexity |
|-----------|------|----------|-----------|
| K-means | Partitioning | Spherical clusters, speed | O(n*k*i) |
| Hierarchical | Agglomerative | Dendrogram, flexibility | O(n²) to O(n³) |
| DBSCAN | Density-based | Arbitrary shapes, noise | O(n²) |
| PCA | Projection | Linear reduction | O(n*p²) |
| t-SNE | Projection | Visualization | O(n²) |
| Isolation Forest | Anomaly | Fast anomaly detection | O(n log n) |
| LOF | Anomaly | Local density anomalies | O(n²) |

## 📈 Evaluation Metrics

### For Clustering
- Silhouette Score (-1 to 1, higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Index (higher is better)
- Inertia (within-cluster sum of squares)

### For Anomaly Detection
- Precision & Recall
- F1 Score
- ROC-AUC
- Isolation score

## 🚀 Quick Start Examples

### K-means Clustering
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Evaluate
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.4f}")
```

### PCA for Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Check explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

### Anomaly Detection with Isolation Forest
```python
from sklearn.ensemble import IsolationForest

# Detect anomalies
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)

# -1 indicates anomaly, 1 indicates normal
n_anomalies = (anomalies == -1).sum()
print(f"Detected {n_anomalies} anomalies")
```

## 🔍 Finding Optimal Parameters

### K-means: Elbow Method
```python
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot to find elbow point
```

### DBSCAN: Parameter Search
```python
from sklearn.neighbors import NearestNeighbors

# Find optimal epsilon
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances = np.sort(neighbors_fit.kneighbors(X_scaled)[0][:, 4], axis=0)
# Plot distances to find "knee"
```

## 📚 Resources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Stanford CS229 - Unsupervised Learning](http://cs229.stanford.edu/)
- [Introduction to Statistical Learning](https://www.statlearning.com/)
- [Clustering Analysis](https://www.coursera.org/learn/cluster-analysis)

## ✨ Tips

- Always scale/normalize your features before clustering
- Try multiple algorithms and compare results
- Use silhouette score to evaluate cluster quality
- Visualize clusters when possible (2D/3D projection)
- Document your parameter choices
- Consider computational complexity for large datasets
- Interpret results in business context

## 🎓 Common Challenges

1. **Choosing K** - Use elbow method or silhouette analysis
2. **Feature Scaling** - Essential for distance-based algorithms
3. **Interpretability** - Understand why clusters were formed
4. **Scalability** - Some algorithms slow down with large data
5. **Noise Handling** - Different algorithms handle noise differently

---

**Happy Clustering!** 🎓
