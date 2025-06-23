import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import os

# Define a directory for saving plots (inside static folder)
PLOTS_DIR = os.path.join(os.getcwd(), 'static', 'images')
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_kmeans_analysis(data_file_path, n_clusters=3, n_nearest_points=6):
    """
    Performs KMeans clustering with PCA and returns visualization and nearest points.
    This version follows the Jupyter notebook logic closely, without scaling and using all features.
    """
    try:
        # === Load and prepare data ===
        df = pd.read_csv(data_file_path)
        df.rename(columns={df.columns[0]: 'ID_ZONE'}, inplace=True)
        X = df.drop(columns=['ID_ZONE'])

        # === Apply KMeans ===
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(X)
        df['Cluster'] = clusters

        # === Compute distances to centroids ===
        X_values = X.values
        centroids = kmeans.cluster_centers_
        distances = []

        for i in range(len(X_values)):
            cluster_id = df.loc[i, 'Cluster']
            centroid = centroids[cluster_id]
            dist = np.linalg.norm(X_values[i] - centroid)
            distances.append(dist)

        df['DistanceToCentroid'] = distances

        # === PCA for 2D visualization ===
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        df['X_PCA'] = components[:, 0]
        df['Y_PCA'] = components[:, 1]

        # Project centroids into PCA space
        centroids_df = pd.DataFrame(centroids, columns=X.columns)
        centroid_components = pca.transform(centroids_df)

        # === Save nearest points to CSV ===
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(output_dir, exist_ok=True)
        nearest_points_path = os.path.join(output_dir, 'nearest_points_to_centroids.csv')

        nearest_points_dfs = []
        for i in range(n_clusters):
            cluster_df = df[df['Cluster'] == i].sort_values(by='DistanceToCentroid').head(n_nearest_points)
            cluster_df = cluster_df.drop(columns=['X_PCA', 'Y_PCA'])  # Remove PCA columns
            nearest_points_dfs.append(cluster_df) 

        if nearest_points_dfs:
            pd.concat(nearest_points_dfs).to_csv(nearest_points_path, index=False)
            print(f"Nearest points to centroids saved to {nearest_points_path}")

        # === Format cluster info ===
        cluster_details_output = []
        for i in range(n_clusters):
            cluster_zones = df[df['Cluster'] == i]['ID_ZONE'].tolist()
            cluster_details_output.append(
                f"🟢 Cluster {i} - Total: {len(cluster_zones)}\n[{', '.join(map(str, cluster_zones))}]"
            )

        # === Format nearest points ===
        nearest_points_output = []
        for i in range(n_clusters):
            cluster_df = df[df['Cluster'] == i].nsmallest(n_nearest_points, 'DistanceToCentroid')
            nearest_points_output.append(f"📌 Nearest {n_nearest_points} points to Centroid of Cluster {i}:")
            nearest_points_output.append(cluster_df[['ID_ZONE', 'DistanceToCentroid']].to_string(index=False))

        # === Plot clustering result ===
        plt.figure(figsize=(10, 6))
        plt.scatter(df['X_PCA'], df['Y_PCA'], c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
        plt.scatter(centroid_components[:, 0], centroid_components[:, 1],
                    c='red', marker='X', s=200, label='Centroids')

        plt.title('K-means Clustering with PCA and Centroids')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.colorbar(label='Cluster')
        plt.legend()
        plt.grid(True)

        # Save plot to static/images/
        plot_filename = 'kmeans_pca_plot.png'
        plot_path = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        # === Return result ===
        return {
            'cluster_details': cluster_details_output,
            'nearest_points': nearest_points_output,
            'pca_components': pca.components_.tolist(),
            'kmeans_plot_url': f"/static/images/{plot_filename}"
        }

    except Exception as e:
        print(f"Error in run_kmeans_analysis: {e}")
        raise e
