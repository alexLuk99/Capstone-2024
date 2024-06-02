from pathlib import Path

import pandas as pd
import altair as alt
from joblib import load

from sklearn.decomposition import PCA

from config.paths import input_path, output_path, interim_path, models_path


def analyze_clusters():
    df_clustered = pd.read_csv(interim_path / 'clustered.csv')
    label_encoder_path = Path(models_path / 'label_encoders.joblib')

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

    # Histogramm für 'Durschnittliche_Zeit_zwischen_Towings'
    tmp_towings = df_clustered[['VIN', 'Cluster', 'Durschnittliche_Zeit_zwischen_Towings']]
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
    hist_towings.save(output_path / 'histogram_towings.html')

    # Scatter plot für 'Repairs' vs. 'Aufenthalte'
    tmp_repairs_vs_aufenthalte = df_clustered[['VIN', 'Cluster', 'Repairs', 'Aufenthalte']]
    scatter_repairs_vs_aufenthalte = alt.Chart(tmp_repairs_vs_aufenthalte).mark_circle(size=60).encode(
        x='Repairs:Q',
        y='Aufenthalte:Q',
        color='Cluster:N',
        tooltip=['VIN', 'Cluster']
    ).properties(
        title='Scatter Plot of Repairs vs. Aufenthalte by Cluster'
    ).properties(
        width=600,
        height=600
    )

    # Speichern des Scatterplots
    scatter_repairs_vs_aufenthalte.save(output_path / 'scatter_repairs_vs_aufenthalte.html')

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
    hist_rental_car_days.save(output_path / 'histogram_rental_car_days.html')

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
    modellreihe_distribution.save(output_path / 'distribution_modellreihe.html')

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

    dist_outcome_desc.save(output_path / 'distribution_outcome_desc.html')

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

    dist_component.save(output_path / 'distribution_component.html')


# Visualisierungen um Cluster zu analysieren
