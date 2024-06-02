import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import altair as alt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
import matplotlib.pyplot as plt
from loguru import logger
from joblib import dump, load
from config.paths import interim_path, output_path, models_path
from src.ml.models.analyze_clusters import interpret_pca_loadings

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())


def clustering(data: pd.DataFrame, train_model: bool = False):
    kmeans_path = Path(models_path / 'kmeans_model.joblib')

    # Standardize the features
    scaler = MaxAbsScaler()
    features_scaled = scaler.fit_transform(data)

    # Applying PCA
    pca = PCA(n_components=features_scaled.shape[1])
    pca.fit(features_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Determine the number of components to explain 80% of the variance
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    logger.info(f'Number of components explaining 80% variance: {n_components_80}')

    if train_model:
        # Create a DataFrame for the explained variance
        df_variance = pd.DataFrame({
            'Number of Components': np.arange(1, len(explained_variance) + 1),
            'Explained Variance': explained_variance,
            'Cumulative Variance': cumulative_variance
        })

        # Elbow plot using Altair
        elbow_plot = alt.Chart(df_variance).mark_line(point=True).encode(
            x=alt.X('Number of Components:Q', title='Number of Components'),
            y=alt.Y('Explained Variance:Q', title='Explained Variance')
        ).properties(
            title='Elbow Plot of Explained Variance'
        )

        # Cumulative explained variance plot using Altair
        cumulative_plot = alt.Chart(df_variance).mark_line(point=True, color='red').encode(
            x=alt.X('Number of Components:Q', title='Number of Components'),
            y=alt.Y('Cumulative Variance:Q', title='Cumulative Variance')
        ).properties(
            title='Cumulative Explained Variance'
        )

        # Display both plots
        combined_plot = elbow_plot | cumulative_plot
        combined_plot.save(output_path / 'elbow_cum_variance.html')

        pipeline = Pipeline([
            ('scaler', MaxAbsScaler()),
            ('pca', PCA(n_components=n_components_80))
        ])

        # Transform the features
        features_pca = pipeline.fit_transform(data)

        # Zurückgeben welche Spalten in PCA einfließen
        numerical_features = data.columns.tolist()
        loadings = pd.DataFrame(pipeline.steps[1][1].components_.T, index=numerical_features)

        loading_cols = []
        _ = [loading_cols.append(f'PC{x}') for x in loadings.columns]

        loadings.columns = loading_cols

        interpret_pca_loadings(loadings=loadings)

        # Finding the optimal number of clusters using the Silhouette Score
        silhouette_scores = []

        k_s = range(2, 8)
        # Finding the optimal number of clusters using the Silhouette Score and creating Silhouette Plots
        for n_clusters in k_s:
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)

            # The silhouette plot
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(features_pca) + (n_clusters + 1) * 10])

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_pca)

            silhouette_avg = silhouette_score(features_pca, cluster_labels)
            print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
            silhouette_scores.append(silhouette_avg)

            sample_silhouette_values = silhouette_samples(features_pca, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10

            ax1.set_title(f"Silhouette plot for the various clusters with n_clusters = {n_clusters}")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])
            ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

            plt.savefig(output_path / f'silhouette_plot_{n_clusters}.png')
            plt.close(fig)

        # Choose the optimal number of clusters (based on the silhouette score)
        optimal_k = np.argmax(silhouette_scores) + 2
        print(f'Optimal number of clusters: {optimal_k}')

        plt.figure(figsize=(10, 6))
        plt.plot(k_s, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores For Optimal k')
        plt.savefig(output_path / 'silhouette_scores.png')

        # Define and fit the pipeline with the optimal number of clusters
        pipeline = Pipeline([
            ('scaler', MaxAbsScaler()),
            ('pca', PCA(n_components=n_components_80)),
            ('kmeans', KMeans(n_clusters=optimal_k, random_state=42))
        ])

        # Fit the pipeline
        pipeline.fit(data)

        # Save the model
        dump(pipeline, kmeans_path)

    else:
        # Load the model
        if kmeans_path.exists():
            pipeline = load(kmeans_path)
        else:
            raise FileNotFoundError(f"Model file not found at {kmeans_path}")

    # Predict clusters
    data['Cluster'] = pipeline.predict(data)
    data = data.reset_index()

    data.to_csv(interim_path / 'clustered.csv', index=False)
