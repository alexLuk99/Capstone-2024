import os
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
from config.paths import input_path, interim_path, output_path

os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())


def clustering(df_assistance: pd.DataFrame, df_workshop: pd.DataFrame) -> None:
    # Create dataframe with most important information

    cols = ['Telephone Help', 'Service Paid By Customer']

    for col in cols:
        df_assistance[col] = df_assistance[col].fillna('NO')
        df_assistance[col] = df_assistance[col].str.upper()
        df_assistance[col] = df_assistance[col].map({'YES': True, 'NO': False}).astype(bool)
        df_assistance[col] = df_assistance[col].apply(lambda x: x if isinstance(x, bool) else False)

    df_assistance['Rental Car Days'] = df_assistance['Rental Car Days'].fillna(0)
    df_assistance['days_since_last_towing'] = df_assistance['days_since_last_towing'].fillna(0)

    df_assistance_grouped = df_assistance.groupby(by='VIN', as_index=False).agg(
        Anzahl_Anrufe=pd.NamedAgg(column='Incident Date', aggfunc='count'),
        Durschnittliche_Zeit_zwischen_Towings=pd.NamedAgg(column='days_since_last_towing', aggfunc='mean'),
        Telephone_Help=pd.NamedAgg(column='Telephone Help', aggfunc='sum'),
        Service_Paid_By_Customer=pd.NamedAgg(column='Service Paid By Customer', aggfunc='sum'),
        Rental_Car=pd.NamedAgg(column='Rental Car', aggfunc='sum'),
        Rental_Car_Days=pd.NamedAgg(column='Rental Car Days', aggfunc='sum'),
        Hotel_Service=pd.NamedAgg(column='Hotel Service', aggfunc='sum'),
        Alternative_Transport=pd.NamedAgg(column='Alternative Transport', aggfunc='sum'),
        Taxi_Service=pd.NamedAgg(column='Taxi Service', aggfunc='sum'),
        Vehicle_Transport=pd.NamedAgg(column='Vehicle Transport', aggfunc='sum'),
        Car_Key_Service=pd.NamedAgg(column='Car Key Service', aggfunc='sum'),
        Parts_Service=pd.NamedAgg(column='Parts Service', aggfunc='sum'),
        Additional_Services_Not_Covered=pd.NamedAgg(column='Additional Services Not Covered', aggfunc='sum')
    )

    cols = ['Rental_Car', 'Hotel_Service', 'Alternative_Transport', 'Taxi_Service', 'Vehicle_Transport',
            'Car_Key_Service', 'Parts_Service', 'Additional_Services_Not_Covered']

    df_assistance_grouped['Total Services Offered'] = df_assistance_grouped[cols].sum(axis=1)
    df_assistance_grouped = df_assistance_grouped.drop(columns=cols)

    df_assistance = df_assistance.sort_values(by=['Registration Date', 'VIN'])
    df_assistance_first_registration_date = df_assistance[['VIN', 'Registration Date']].drop_duplicates(
        subset=['VIN'])
    df_assistance_first_registration_date['Registration Date'] = pd.to_datetime(
        df_assistance_first_registration_date['Registration Date'])
    df_assistance_first_registration_date['Registration Date Jahr'] = df_assistance_first_registration_date[
        'Registration Date'].dt.year
    df_assistance_first_registration_date = df_assistance_first_registration_date.drop(columns='Registration Date')

    df_assistance_grouped = df_assistance_grouped.merge(df_assistance_first_registration_date, on='VIN', how='left')

    # Label Encoding der Spalte Modellreihe
    label_encoder = LabelEncoder()
    df_assistance['Modellreihe'] = df_assistance['Modellreihe'].fillna('Nicht definiert')
    df_assistance['Modellreihe_Encoded'] = label_encoder.fit_transform(df_assistance['Modellreihe'])

    tmp_modellreihe = df_assistance[['VIN']].copy().drop_duplicates()
    df_assistance_grouped = df_assistance_grouped.merge(tmp_modellreihe, on='VIN')

    # Label Encodign der Spalte Fahrzeuggruppe
    df_assistance['Fahrzeuggruppe'] = df_assistance['Fahrzeuggruppe'].fillna('Nicht definiert')
    df_assistance['Fahrzeuggruppe_Encoded'] = label_encoder.fit_transform(df_assistance['Fahrzeuggruppe'])

    tmp_fahrzeuggruppe = df_assistance[['VIN']].copy().drop_duplicates()
    df_assistance_grouped = df_assistance_grouped.merge(tmp_fahrzeuggruppe, on='VIN')

    # Spalten als features hinzufügen
    cols_to_pivot = ['Outcome Description', 'Component']

    for col in cols_to_pivot:
        tmp = df_assistance.pivot_table(index='VIN', columns=col, aggfunc='size', fill_value=0).reset_index()
        df_assistance_grouped = df_assistance_grouped.merge(tmp, on='VIN', how='left')
        df_assistance_grouped = df_assistance_grouped.fillna(0)

    # Anzahl Reperaturen pro VIN
    df_repairs = df_workshop.groupby(by='VIN', as_index=False).agg(
        Repairs=pd.NamedAgg(column='Q-Line', aggfunc='count'),
    )

    df_aufenthalte = df_workshop.groupby(by='VIN', as_index=False).agg(
        Aufenthalte=pd.NamedAgg(column='Werkstattaufenthalt', aggfunc='nunique')
    )

    df_assistance_grouped = df_assistance_grouped.merge(df_repairs, on='VIN', how='left')
    df_assistance_grouped = df_assistance_grouped.merge(df_aufenthalte, on='VIN', how='left')

    # Set VIN as Index
    data = df_assistance_grouped.copy()
    data = data.set_index('VIN')

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

    # Predict clusters
    data['Cluster'] = pipeline.predict(data)

    data.to_csv(interim_path / 'clustered.csv', index=False)
