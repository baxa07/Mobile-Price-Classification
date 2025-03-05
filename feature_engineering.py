import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

def feature_engineering(X_train_scaled, X_test_scaled, feature_names, y_train, clustering_method="GMM"):
    """
    Performs feature selection using Random Forest, applies PCA,
    and clusters the PCA-reduced features (using either GMM or DBSCAN).
    Also generates plots for feature importances, PCA variance, and cluster assignments.
    
    Parameters:
        X_train_scaled (np.ndarray): Standardized training features.
        X_test_scaled (np.ndarray): Standardized testing features.
        feature_names (list): List of feature names.
        y_train (pd.Series): Training target values.
        clustering_method (str): Clustering method ('GMM' or 'DBSCAN').
    
    Returns:
        X_train_combined (np.ndarray): Combined PCA and cluster features for training.
        X_test_combined (np.ndarray): Combined PCA and cluster features for testing.
    """
    print("Starting feature engineering...")
    
    # -------------------------------
    # Feature Selection using Random Forest
    # -------------------------------
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    print("Feature importances:")
    print(importance_df)
    
    # Plot and save feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=importance_df, palette="viridis")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.show()
    
    # Select the top 4 features (adjustable as needed)
    top_features = importance_df.head(4)["feature"].tolist()
    print("Selected top features:", top_features)
    
    # Reduce datasets to selected features
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    X_train_selected = X_train_df[top_features]
    X_test_selected = X_test_df[top_features]
    
    # -------------------------------
    # Apply PCA on Selected Features
    # -------------------------------
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)
    
    print(f"PCA reduced dimensions from {X_train_selected.shape[1]} to {X_train_pca.shape[1]}")
    
    # Plot PCA explained variance ratio
    plt.figure(figsize=(8, 6))
    components = range(1, len(pca.explained_variance_ratio_) + 1)
    plt.bar(components, pca.explained_variance_ratio_, color='skyblue')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance Ratio")
    plt.tight_layout()
    plt.savefig("pca_explained_variance.png")
    plt.show()
    
    # -------------------------------
    # Clustering on PCA Data (GMM or DBSCAN)
    # -------------------------------
    if clustering_method == "GMM":
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm.fit(X_train_pca)
        train_clusters = gmm.predict(X_train_pca)
        test_clusters = gmm.predict(X_test_pca)
        print("GMM clustering completed.")
    elif clustering_method == "DBSCAN":
        dbscan = DBSCAN()
        train_clusters = dbscan.fit_predict(X_train_pca)
        test_clusters = dbscan.fit_predict(X_test_pca)
        print("DBSCAN clustering completed.")
    else:
        raise ValueError("Unsupported clustering method. Choose 'GMM' or 'DBSCAN'.")
    
    # Plot cluster assignments if PCA has at least 2 components
    if X_train_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters, cmap='viridis', s=50)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Cluster Assignments on PCA Data")
        plt.colorbar(label="Cluster Label")
        plt.tight_layout()
        plt.savefig("clusters.png")
        plt.show()
    
    # One-hot encode the cluster labels
    X_train_cluster = pd.get_dummies(train_clusters, prefix="cluster")
    X_test_cluster = pd.get_dummies(test_clusters, prefix="cluster")
    X_train_cluster, X_test_cluster = X_train_cluster.align(X_test_cluster, join="outer", axis=1, fill_value=0)
    
    # Combine PCA features with cluster features
    X_train_combined = np.hstack((X_train_pca, X_train_cluster.values))
    X_test_combined = np.hstack((X_test_pca, X_test_cluster.values))
    
    print("Feature engineering completed.")
    return X_train_combined, X_test_combined
