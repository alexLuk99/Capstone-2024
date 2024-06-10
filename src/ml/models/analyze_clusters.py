from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
from joblib import load
from loguru import logger

from sklearn.decomposition import PCA

from config.paths import output_path, interim_path, models_path


def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


def analyze_clusters():
    df_clustered = pd.read_csv(interim_path / 'clustered.csv')
    label_encoder_path = Path(models_path / 'label_encoders.joblib')

    path_cluster_analysis = output_path / 'cluster_analysis'
    path_cluster_analysis.mkdir(exist_ok=True, parents=True)

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

    df_clustered = df_clustered.drop(columns=['PCA1', 'PCA2'])

    # Speichern des Plots
    scatter_plot.save(path_cluster_analysis / 'cluster_visualization.html')

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
    boxplot.save(path_cluster_analysis / 'boxplot_cluster.html')

    # Histogramm für 'Durschnittliche_Zeit_zwischen_Towings'
    tmp_towings = df_clustered[['VIN', 'Cluster', 'Durschnittliche_Zeit_zwischen_Towings']]
    tmp_towings = tmp_towings[tmp_towings['Durschnittliche_Zeit_zwischen_Towings'] > 0].copy()
    hist_towings = alt.Chart(tmp_towings).mark_bar().encode(
        x=alt.X('Durschnittliche_Zeit_zwischen_Towings:Q', bin=alt.Bin(maxbins=10, step=30)),
        y='count()',
        color='Cluster:N'
    ).properties(
        title='Histogram of Durschnittliche Tage zwischen Towings by Cluster'
    ).properties(
        width=400,
        height=600
    )

    # Speichern des Histogramms
    hist_towings.save(path_cluster_analysis / 'histogram_towings.html')

    # Scatter plot für 'Repairs' vs. 'Aufenthalte'
    tmp_repairs_vs_aufenthalte = df_clustered[['VIN', 'Cluster', 'Repairs', 'Aufenthalte']]
    scatter_repairs_vs_aufenthalte = alt.Chart(tmp_repairs_vs_aufenthalte).mark_circle(size=60).encode(
        x='Repairs:Q',
        y='Aufenthalte:Q',
        column='Cluster:N',
        tooltip=['VIN', 'Cluster']
    ).properties(
        title='Scatter Plot of Repairs vs. Aufenthalte by Cluster'
    ).properties(
        width=600,
        height=600
    )

    # Speichern des Scatterplots
    scatter_repairs_vs_aufenthalte.save(path_cluster_analysis / 'scatter_repairs_vs_aufenthalte.html')

    # Histogramm für 'Rental Car Days'
    tmp_rental_car_days = df_clustered[['VIN', 'Cluster', 'Rental_Car_Days']]
    hist_rental_car_days = alt.Chart(tmp_rental_car_days).mark_bar().encode(
        x=alt.X('Rental_Car_Days:Q', bin=alt.Bin(maxbins=10, step=5)),
        y='count()',
        color='Cluster:N'
    ).properties(
        title='Histogram of Rental Car Days'
    ).properties(
        width=400,
        height=600
    )

    # Speichern des Histogramms
    hist_rental_car_days.save(path_cluster_analysis / 'histogram_rental_car_days.html')

    # Load the label encoders
    label_encoder_modellreihe = load(label_encoder_path)

    # Decode the encoded columns
    df_clustered['Modellreihe'] = label_encoder_modellreihe.inverse_transform(df_clustered['Modellreihe_Encoded'])

    # Verteilung von 'Modellreihe'
    modellreihe_distribution = alt.Chart(df_clustered).mark_bar().encode(
        x=alt.X('Modellreihe:N', sort='-y'),
        y='count()',
        color='Cluster:N',
        tooltip=['Modellreihe', 'count()']
    ).properties(
        title='Distribution of Modellreihe by Cluster',
        width=800,
        height=400
    )

    # Speichern des Verteilungsdiagramms für 'Modellreihe'
    modellreihe_distribution.save(path_cluster_analysis / 'distribution_modellreihe.html')

    # Liste der Spalten, die ge-melted werden sollen
    outcome_desc_columns = ['Cancelled', 'Change of Tyre', 'Jump Start',
                            'Rental Car without primary services (i.e. Towing and Roadsaide Assistance)',
                            'Roadside Repair Others', 'Scheduled Towing',
                            'Scheduled roadside repair', 'Towing']
    component_columns = ['Air conditioning ', 'Anti Blocking System (ABS) ', 'Anti Theft Protection', 'Battery',
                         'Body-Equipment inside ', 'Brakes - Brake mechanics ',
                         'Brakes - Hydraulic brake system, regulator ', 'Clutch',
                         'Convertible top, hardtop ', 'Door, central locking system ',
                         'Engine - Cooling', 'Engine - General', 'Engine - Lubrication',
                         'Exhaust system ', 'Final drive - Differential, differential lock ',
                         'Flat Tyre', 'Fuel supply', 'Fuel system / Electronic ignition ',
                         'Generator', 'Glazing, window control ',
                         'Ignition and preheating system ', 'Instruments ',
                         'Insufficient Fuel / Empty Fuel Tank', 'Key/Lock/Remote Control',
                         'Level control, air suspension ', 'Lids, flaps ',
                         'Lights, lamps, switches', 'Not determinable', 'Passenger protection ',
                         'Radio, stereo, telephone, on-board computer ',
                         'Shift / Selector lever', 'Sliding roof, tilting roof', 'Starter',
                         'Steering ', 'Suspension, drive shafts ', 'Transmission',
                         'Windshield wiper and washer system '
                         ]

    # Verteilung Outcome Description
    df_melted_outcome_desc = pd.melt(df_clustered, id_vars=['VIN', 'Cluster'], value_vars=outcome_desc_columns,
                                     var_name='Outcome Description', value_name='Count')

    df_melted_outcome_desc = df_melted_outcome_desc[df_melted_outcome_desc['Count'] > 0]
    df_melted_outcome_desc = df_melted_outcome_desc.groupby(by=['Cluster', 'Outcome Description'], as_index=False)[
        'Count'].sum()

    dist_outcome_desc = alt.Chart(df_melted_outcome_desc).mark_bar().encode(
        x=alt.X('Outcome Description', sort='-y'),
        y='Count',
        color='Cluster:N'
    ).properties(
        title='Verteilung der Outcome Description nach Spalte',
        width=600,
        height=400
    ).configure_axis(
        labelAngle=-45
    )

    dist_outcome_desc.save(path_cluster_analysis / 'distribution_outcome_desc.html')

    # Verteilung Component
    df_melted_component = pd.melt(df_clustered, id_vars=['VIN', 'Cluster'], value_vars=component_columns,
                                  var_name='Component', value_name='Count')

    df_melted_component = df_melted_component[df_melted_component['Count'] > 0]
    df_melted_component = df_melted_component.groupby(by=['Cluster', 'Component'], as_index=False)[
        'Count'].sum()

    dist_component = alt.Chart(df_melted_component).mark_bar().encode(
        x=alt.X('Component', sort='-y'),
        y='Count',
        color='Cluster:N'
    ).properties(
        title='Verteilung der Component nach Spalte',
        width=600,
        height=400
    ).configure_axis(
        labelAngle=-45
    )

    dist_component.save(path_cluster_analysis / 'distribution_component.html')

    # Cluster DataFrames erstellen
    clusters = df_clustered['Cluster'].unique()
    cluster_dfs = {cluster: df_clustered[df_clustered['Cluster'] == cluster] for cluster in clusters}

    results = {}

    for cluster, df in cluster_dfs.items():
        numerical_means = df.select_dtypes(include=[np.number]).mean()
        numerical_median = df.select_dtypes(include=[np.number]).median()
        nominal_distributions = {}

        for col in df.select_dtypes(include=['object', 'bool']).columns:
            nominal_distributions[col] = df[col].value_counts(normalize=True)

        results[cluster] = {
            'Numerical Means': numerical_means,
            'Numerical Medians': numerical_median,
            'Nominal Distributions': nominal_distributions
        }

    result_df = pd.DataFrame()

    mean_results = []
    for c in results:
        mean_results.append(results[c].get('Numerical Means'))

    mean_result_df = pd.concat(mean_results, axis=1).reset_index(names='Feature')
    mean_result_df = mean_result_df.melt(value_vars=[0, 1, 2], var_name='Cluster', value_name='Mean',
                                         id_vars=['Feature'])

    median_results = []
    for c in results:
        median_results.append(results[c].get('Numerical Medians'))

    median_result_df = pd.concat(median_results, axis=1).reset_index(names='Feature')
    median_result_df = median_result_df.melt(value_vars=[0, 1, 2], var_name='Cluster', value_name='Median',
                                             id_vars=['Feature'])

    result_df = mean_result_df.merge(median_result_df, on=['Feature', 'Cluster'], how='left')

    result_df = result_df.melt(value_vars=['Mean', 'Median'], value_name='Value', var_name='Typ', id_vars=['Feature', 'Cluster'])

    chart = alt.Chart(result_df).mark_point().encode(
        x='Cluster:N',
        y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),
        color='Cluster:N',
        shape='Typ:N',
        tooltip=['Feature', 'Cluster', 'Typ', 'Value']
    ).properties(
        width=100,
        height=400
    ).facet(
        column='Feature:N'
    ).resolve_scale(
        y='independent'
    )

    # chart.save('test.html')

    result_df.to_excel(path_cluster_analysis / 'Mittelwerte.xlsx')

    numerical_columns = df_clustered.select_dtypes(include=['number']).columns.tolist()


def interpret_pca_loadings(loadings, output_path: Path, top_n: str =5):
    """
    Interpret the PCA loadings by identifying the top N features contributing to each principal component.
    """
    n_components = loadings.shape[1]
    visuals = []
    for i in range(n_components):
        component_name = f'PC{i}'

        sorted_loadings = loadings[component_name].abs().sort_values(ascending=False)
        top_features = sorted_loadings.head(top_n).index.tolist()

        # for feature in top_features:
        #     loading_value = loadings.loc[feature, component_name]
        #     print(f'Feature: {feature}, Loading: {loading_value:.4f}')

        # Create a DataFrame for Altair
        top_loadings_df = loadings.loc[top_features, component_name].reset_index()
        top_loadings_df.columns = ['Feature', 'Loading']

        # Visualization with Altair
        bar_chart = alt.Chart(top_loadings_df).mark_bar().encode(
            x=alt.X('Loading:Q'),
            y=alt.Y('Feature:N', sort='-x'),
            color=alt.Color('Loading:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=[
            alt.Tooltip('Feature:N', title='Feature'),
            alt.Tooltip('Loading:Q')
        ]
        ).properties(
            title=f'Top {top_n} Features Contributing to {component_name}',
            width=600,
            height=400
        )

        visuals.append(bar_chart)

    chart = alt.vconcat(*visuals)

    path = output_path / 'loadings'
    path.mkdir(exist_ok=True, parents=True)

    chart.save(path / f'top_{top_n}_features.html')
