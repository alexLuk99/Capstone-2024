import pandas as pd
import altair as alt
from pathlib import Path

from sklearn.decomposition import PCA

from config.paths import input_path, output_path, interim_path


def analyze_clusters():
    df_clustered = pd.read_csv(interim_path / 'clustered.csv')

    # PCA für die Visualisierung
    pca = PCA(n_components=2)
    df_clustered['PCA1'] = pca.fit_transform(df_clustered.drop(columns=['VIN', 'Cluster']))[:, 0]
    df_clustered['PCA2'] = pca.fit_transform(df_clustered.drop(columns=['VIN', 'Cluster']))[:, 1]

    # Erstellen der Altair-Plots
    tmp_clustered = df_clustered[['VIN', 'PCA1', 'PCA2', 'Cluster']].copy()
    scatter_plot = alt.Chart(tmp_clustered).mark_circle(size=60).encode(
        x='PCA1',
        y='PCA2',
        color='Cluster:N',
        tooltip=['VIN', 'Cluster']
    ).properties(
        title='Cluster Visualization using PCA'
    ).properties(
        width=600,
        height=600
    )

    # Speichern des Plots
    scatter_plot.save(output_path / 'cluster_visualization.html')

    # Boxplot für jede Clustergröße
    tmp_anrufe = df_clustered[['VIN', 'Cluster', 'Anzahl_Anrufe']]
    boxplot = alt.Chart(tmp_anrufe).mark_boxplot().encode(
        x='Cluster:N',
        y='Anzahl_Anrufe:Q',
        color='Cluster:N'
    ).properties(
        title='Boxplot of Anzahl Anrufe by Cluster'
    ).properties(
        width=400,
        height=600
    )

    # Speichern des Boxplots
    boxplot.save(output_path / 'boxplot_cluster.html')

    # Visualisierungen um Cluster zu analysieren
