# ML Session05: Unsupervised Learning; PCA & K-Means

## Session Overview

In this session, we will explore key concepts in **Unsupervised Learning**, focusing on:
1. Understanding what unsupervised learning is and how it differs from supervised learning
2. Real-life applications of unsupervised learning
3. Common algorithms in unsupervised learning, **Principal Component Analysis (PCA)** and **K-Means Clustering**
4. Hands-on implementation and visualization using Python and Scikit-learn

---

## 1. What is Unsupervised Learning?

Unsupervised learning is a type of machine learning where the model is given **unlabeled data**, and the goal is to discover **underlying patterns**, **groupings**, or **structures** without explicit supervision.

There is **no ground truth output** provided, just input data. 
The algorithm tries to:
- Find clusters or groups (clustering)
- Reduce dimensionality (feature extraction)
- Detect anomalies or unusual patterns

---

## Supervised vs. Unsupervised Learning

![supervised](images/image2.png)

![unsupervised](images/image3.png)

| Aspect                  | Supervised Learning                    | Unsupervised Learning                      |
|-------------------------|----------------------------------------|--------------------------------------------|
| Input                   | Labeled data (features + target)       | Unlabeled data (features only)             |
| Goal                    | Learn a mapping from input to output   | Discover hidden patterns or structure      |
| Example Algorithms      | Linear Regression, Decision Trees, SVM | K-Means, PCA, DBSCAN                       |
| Applications            | Spam detection, fraud prediction       | Customer segmentation, anomaly detection   |

---

## Common Unsupervised Learning Methods

1. **Clustering Algorithms**

Clustering is a technique for exploring raw, unlabeled data and breaking it down into groups (or clusters) based on similarities or differences. 
   - Exclusive, Overlapping, Hierarchical, Probabilistic 
   - **K-Means Clustering** – Groups data into *k* distinct clusters
   
   - **DBSCAN** – Density-based clustering for arbitrary shapes
   - **Hierarchical Clustering** – Builds a tree of clusters


2. **Dimensionality Reduction**

Dimensionality reduction is an unsupervised learning technique that reduces the number of features, or dimensions, in a dataset. More data is generally better for machine learning, but it can also make it more challenging to visualize the data.
Dimensionality reduction extracts important features from the dataset, reducing the number of irrelevant or random features present. 
   - **Principal Component Analysis (PCA)** – Projects data into lower-dimensional space while preserving variance
   - **t-SNE, UMAP** – Used for visualization of high-dimensional data

<!-- .

3. **Association**

Association rule mining is a rule-based approach to reveal interesting relationships between data points in large datasets. Unsupervised learning algorithms search for frequent if-then associations—also called rules—to discover correlations and co-occurrences within the data and the different connections between data objects.  -->

---

## Real-Life Applications of Unsupervised Learning

- **Customer Segmentation** – Grouping customers based on purchasing behavior
- **Image Compression** – Using PCA to reduce pixel dimensions
- **Anomaly Detection** – Identifying fraudulent transactions or system failures
- **Topic Modeling** – Grouping similar documents or news articles
- **Music Recommendation** – Clustering songs/users by listening patterns

---

# **2. Clustering**

![alt text](images/clustering_intro.png)
 - Soft Clustering: 
 
  Each data point is strictly assigned to one cluster, meaning that a point can only belong to one 
cluster and not to multiple clusters.
 - Hard Clustering: 
 
  Each data point can belong to multiple clusters with a probability, indicating how likely a data 
point is to belong to different clusters. This allows for overlapping between clusters, which may 
share common features

![clustering](images/clustering.png)

## 2.1. **K-means**
K-Means is a centroid-based clustering algorithm where data points are assigned to clusters based on the 
closest centroid. The algorithm operates iteratively, improving the position of the centroids over time until a 
local optimal clustering is reached

### Intuition
- K-Means aims to partition a set of n observations into k clusters.
- Each observation belongs to the cluster with the nearest mean (also called the cluster centroid).
- The goal is to minimize the within-cluster sum of squares (WCSS), also known as inertia.

### Algorithm Steps
1. Choose the number of clusters k: Decide how many groups (clusters) you want to divide your data into.

2. Initialize k centroids randomly: Randomly pick k data points as the initial centroids (center points of clusters).

3. Assignment Step: 

   For each data point in the dataset:
      - Calculate the distance between the point and each centroid (Euclidean Distance)
      - Assign the point to the nearest centroid (i.e., the closest cluster)

The Euclidean distance between two points in an n-dimensional space is given by:
$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$



4. Update Step:

   After all points are assigned to clusters:
      - Recalculate the centroid of each cluster as the mean of all points assigned to that cluster

$$
C_k = \frac{1}{|S_k|} \left( x_1 + x_2 + \dots + x_n \right)

 = \frac{1}{|S_k|} \sum_{x_i \in S_k} x_i
$$




5. Repeat steps 3 and 4: 

   Keep reassigning points and updating centroids until:

      - The centroids do not change significantly (convergence), or
      - A maximum number of iterations is reached

### Stopping Criteria
The K-Means algorithm stops iterating when one of the following conditions is met:

1. **Data points stay in the same cluster** – When the assignment of data points to clusters no longer changes.

2. **Centroids remain the same** – When the centroids no longer move, indicating convergence.
3. **Distance threshold** – The algorithm stops if the distance between data points and their centroids falls below a specified threshold.
4. **Maximum iterations** – The algorithm stops after a set number of iterations to prevent excessive computation if convergence is slow.

### Advantages of K-Means Clustering
 - Simple to understand and implement:
 K-Means is easy to apply as it iteratively refines the 
centroids until it converges.
 - Computationally efficient: 
 It is faster and works well with large datasets compared to 
other clustering techniques like hierarchical clustering.
 - Suitable for large datasets: 
 Due to its efficiency, K-Means is ideal for large datasets with 
millions of data points.

### Problems that K-Means Faces:

1. **Choice of K (Number of Clusters)**:
   - K-Means requires the user to specify the number of clusters (K) beforehand. Choosing the wrong value for K can lead to poor results, and there's no inherent method in K-Means to automatically determine the optimal number of clusters.

2. **Sensitivity to Initialization**/ **Convergence to Local Minima**:
   - The algorithm is sensitive to the initial placement of centroids. If the initial centroids are poorly chosen, the algorithm may converge to a suboptimal solution, often stuck in a local minimum. This is why techniques like K-Means++ are used to improve initialization.

3. **Assumption of Spherical Clusters**:
   - K-Means assumes that clusters are spherical (circular in 2D) and of similar size. It struggles with clusters that are non-spherical, uneven in size, or have varying densities.

4. **Sensitivity to Outliers**:
   - K-Means is highly sensitive to outliers because the centroid is the mean of all points in the cluster, and outliers can significantly skew the mean, distorting the results.



### Elbow Method:
A technique used to determine the optimal number of clusters (K) for K-Means clustering

![elbow-method](images/elbow_method.png)


**Steps:**
1. **Run K-Means for Different K Values**:
   - You run the K-Means algorithm for a range of values of K (e.g., from 1 to 10).

2. **Calculate WSS (Inertia)**:
   - For each K, calculate the **within-cluster sum of squared errors (WSS)**, which measures the compactness of the clusters (i.e., how close the data points are to their centroids).

3. **Plot the Elbow Curve**:
   - Plot K (number of clusters) on the x-axis and the WSS on the y-axis. You’ll typically see a steep drop in WSS as K increases, but after a certain point, the rate of decrease slows down.

4. **Identify the Elbow**:
   - The "elbow" point is where the curve starts to level off. This is the point where adding another cluster doesn't significantly improve the WSS. The value of K at the elbow is considered the optimal number of clusters.

---
---
## 3. **Dimensionality Reduction**


### 3.1. The Curse of Dimensionality

The **Curse of Dimensionality** refers to the collection of challenges that arise when analyzing and organizing data in high-dimensional spaces. As the number of dimensions (features) increases, several issues begin to affect data analysis and machine learning models.

---

**Key Problems**

1. **Data Sparsity**  
   - In high dimensions, data points become increasingly sparse.  
   - This makes it difficult to find reliable patterns or clusters.

2. **Loss of Distance Meaningfulness**  
   - In low dimensions, "closeness" between points makes intuitive sense.  
   - In high dimensions, all points tend to become similarly distant from one another.  
   - This impacts algorithms that rely on distance metrics, such as k-NN or k-Means.

3. **Exponential Data Requirement**  
   - The volume of space grows exponentially with dimensions.  
   - To maintain a dense sampling of the space, exponentially more data is needed.

4. **Overfitting Risk**  
   - More features increase model complexity.  
   - Models can fit the training data too well, failing to generalize to new data.

---

### Example: Distance Breakdown

As dimensions increase, the difference between the nearest and farthest neighbor distances shrinks:

| Dimensions | Nearest Distance | Farthest Distance | Ratio |
|------------|------------------|-------------------|-------|
| 1          | 0.1              | 1.0               | 0.1   |
| 10         | 0.75             | 0.9               | 0.83  |
| 100        | 0.95             | 0.99              | 0.96  |

> In high dimensions, the distances are almost the same,  reducing their usefulness.

> In summary, The Curse of Dimensionality reminds us that **more features ≠ better models**. Careful feature engineering and dimensionality reduction are essential for effective learning in high-dimensional spaces.

![...](images/curse_of_dimensionality.png)


## 3.2. How to Address It? Dimensionality Reduction

![dim_redn](images/dimensionality_Reduction.png)

- **Feature Selection**: chooses a subset of the existing features (columns) in your dataset that are most relevant to the task.
- **Feature Extraction**: Feature extraction creates new features by transforming or combining the original ones, often reducing dimensionality in the process.

### Feature Extraction VS Feature Selection
![feature](images/feature_selec_extrac.png)

--------------------






.

.

.

.

.

.

.

.

.



---
---
## Resources
1. [What is unsupervised learning?](https://cloud.google.com/discover/what-is-unsupervised-learning)
2. [K-Means Clustering Algorithm](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)
3. [The Curse of Dimensionality](https://www.youtube.com/watch?v=9Tf-_mJhOkU)
4. [Feature Selection vs Feature Extraction](https://vitalflux.com/machine-learning-feature-selection-feature-extraction/#google_vignette)