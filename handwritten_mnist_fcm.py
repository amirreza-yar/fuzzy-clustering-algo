import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import Parallel, delayed
import time

# Load and preprocess the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
data = mnist['data']
labels = mnist['target'].astype(int)

print("Normalizing data...")
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Optional: Reduce dimensionality for faster computation
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=50)
data_pca = pca.fit_transform(data)

# FCM Parameters
k = 10  # Number of clusters (digits 0-9)
MAX_ITER = 1000
m = 2.00
n = len(data_pca)

# Initialize Membership Matrix
def initializeMembershipMatrix():
    print("Initializing membership matrix...")
    return np.random.dirichlet(np.ones(k), size=n)

# Calculate Cluster Centers
def calculateClusterCenter(membership_mat):
    print("Calculating cluster centers...")
    membership_mat_raised = membership_mat ** m
    cluster_centers = membership_mat_raised.T @ data_pca / membership_mat_raised.sum(axis=0)[:, None]
    return cluster_centers

# Update Membership Values
def updateMembershipValue(i, cluster_centers):
    x = data_pca[i]
    distances = np.linalg.norm(x - cluster_centers, axis=1)
    inv_distances = np.reciprocal(distances, where=distances != 0)
    weights = inv_distances ** (2 / (m - 1))
    new_membership = weights / np.sum(weights)
    return new_membership

# Get Clusters
def getClusters(membership_mat):
    print("Assigning clusters...")
    return np.argmax(membership_mat, axis=1)

def update_membership(i, cluster_centers):
    return updateMembershipValue(i, cluster_centers)

def fuzzyCMeansClustering():
    print("Starting Fuzzy C-Means clustering...")
    membership_mat = initializeMembershipMatrix()
    curr = 0

    while curr <= MAX_ITER:
        print(f"Iteration {curr}...")
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = np.array(Parallel(n_jobs=-1)(delayed(update_membership)(i, cluster_centers) for i in range(n)))
        curr += 1

    print("Clustering complete.")
    return getClusters(membership_mat), cluster_centers

# Evaluate Clustering
def evaluate_clustering(cluster_labels, true_labels):
    print("Evaluating clustering performance...")
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return ari, nmi

# Assign Cluster Labels
def assign_cluster_labels(cluster_labels, true_labels):
    print("Assigning labels to clusters...")
    label_map = {}
    true_labels = np.array(true_labels, dtype=int)  # Convert true labels to a NumPy array of integers
    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]  # Extract the indices
        mode_label = np.bincount(true_labels[indices]).argmax()  # Use numpy's bincount to find the mode
        label_map[cluster] = mode_label
    return label_map

# Predict Labels
def predict(cluster_labels, label_map):
    print("Predicting labels...")
    return np.vectorize(label_map.get)(cluster_labels)

# Plot Clusters
def plot_clusters_pca(data, labels, centers):
    print("Visualizing clusters with PCA...")
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    plt.title('Fuzzy C-Means Clustering of Handwritten Digits')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Plot Clusters with t-SNE
def plot_clusters_tsne(data, labels, centers):
    print("Visualizing clusters with t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30)
    reduced_data = tsne.fit_transform(data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    reduced_centers = tsne.fit_transform(centers)
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', marker='x', s=100)

    plt.title('Fuzzy C-Means Clustering of Handwritten Digits')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# Plot Silhouette Score
def plot_silhouette_score(data, labels):
    print("Calculating Silhouette Score...")
    silhouette_avg = silhouette_score(data, labels)
    print("Silhouette Score:", silhouette_avg)

# Plot K-Means Centers
def plot_kmeans_centers(data, labels):
    print("Visualizing K-Means centers...")
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    plt.title('K-Means Clustering of Handwritten Digits')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot K-Means Mini Batch Centers
def plot_mini_batch_kmeans_centers(data, labels):
    mbkmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000)
    mbkmeans.fit(data)
    centers = mbkmeans.cluster_centers_

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
    plt.title('Mini Batch K-Means Clustering of Handwritten Digits')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Choose Visualization Method
def choose_visualization_method(data, labels, centers):
    print("Choose a visualization method:")
    print("1. PCA")
    print("2. t-SNE")
    print("3. Silhouette Score")
    print("4. K-Means Centers")
    print("5. Mini Batch K-Means Centers")
    choice = input("Enter your choice (1/2/3/4/5): ")

    if choice == "1":
        plot_clusters_pca(data, labels, centers)
    elif choice == "2":
        plot_clusters_tsne(data, labels, centers)
    elif choice == "3":
        plot_silhouette_score(data, labels)
    elif choice == "4":
        plot_kmeans_centers(data, labels)
    elif choice == "5":
        plot_mini_batch_kmeans_centers(data, labels)
    else:
        print("Invalid choice. Please select a valid option.")

# Run FCM
start_time = time.time()
labels, centers = fuzzyCMeansClustering()
end_time = time.time()
print("FCM Execution Time:", end_time - start_time, "seconds")

# Evaluate Clustering
ari, nmi = evaluate_clustering(labels, mnist['target'])
print(f"ARI: {ari}, NMI: {nmi}")

# Assign Cluster Labels and Predict
label_map = assign_cluster_labels(labels, mnist['target'])
predicted_labels = predict(labels, label_map)

# Visualize Clusters
choose_visualization_method(data_pca, labels, centers)
