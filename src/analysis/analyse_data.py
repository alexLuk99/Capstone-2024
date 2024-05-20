from pathlib import Path
import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import norm
from collections import Counter
import re

from src.analysis.statistics.chi_square import perform_chi_square_test
from src.analysis.visuals.choropleth import create_country_choropleth
from src.analysis.visuals.timeline import get_timeline


def analyse_data() -> None:
    # CSV-Datei einlesen
    df_assistance = pd.read_csv('data/interim/assistance.csv', low_memory=False)
    df_assistance = df_assistance.convert_dtypes()

    # Erstellen des Ausgabeverzeichnisses
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)

    # Example 1
    # country_of_orgigin = alt.Chart(df_assistance).mark_bar().encode(
    #     alt.X('Country Of Origin:N', title='Unfallsland'),
    #     alt.Y('count(Country Of Origin):Q', title='Anzahl'),
    #     tooltip=[
    #         alt.Tooltip('Country Of Origin:N', title='Unfallsland'),
    #         alt.Tooltip('count(Country Of Origin):Q', title='Anzahl', format=',.0f'),
    #     ]
    # )
    # country_of_orgigin.save('output/country_of_orgigin_chart.html')
    #
    # # Example 2
    # component = alt.Chart(df_assistance).mark_bar().encode(
    #     alt.X('Component:N', title='Komponente'),
    #     alt.Y('count(Component):Q', title='Anzahl'),
    #     tooltip=[
    #         alt.Tooltip('Component:N', title='Komponente'),
    #         alt.Tooltip('count(Component):Q', title='Anzahl', format=',.0f'),
    #     ]
    # )
    #
    # component.save('output/component_chart.html')
    #
    # # Jan
    # create_country_choropleth(df=df_assistance, column='Country Of Origin', title='Number of permits')
    # create_country_choropleth(df=df_assistance, column='Country Of Incident', title='Number of incidents')


    # Zähle die Häufigkeit jedes Outcome Description
    outcome_counts = df_assistance['Outcome Description'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome Description', 'Frequency']

    # Erstelle ein Balkendiagramm mit Altair
    chart2 = alt.Chart(outcome_counts).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Frequency'),
        y=alt.Y('Outcome Description:N', sort='-x', title='Outcome Description'),
        tooltip=['Outcome Description', 'Frequency']
    ).properties(
        title='Frequency of Outcome Descriptions',
        width=800,
        height=400
    )

    # Speichere das Diagramm
    chart2.save(output_path / 'outcome_description_frequency.html')


'''
#%% Comonent frquency

    # Count the frequency of each component
    component_counts = df_assistance['Component'].value_counts().reset_index()
    component_counts.columns = ['Component', 'Frequency']

    # Create a bar chart using Altair
    chart = alt.Chart(component_counts).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Frequency'),
        y=alt.Y('Component:N', sort='-x', title='Component'),
        tooltip=['Component', 'Frequency']
    ).properties(
        title='Frequency of Components',
        width=800,
        height=400
    )

    # Save the chart
    chart.save(output_path / 'component_frequency.html')


#%% Fault description Wörter


    # Einlesen der Stopwords aus der CSV-Datei
    stopwords_df = pd.read_csv('data/interim/stopwords.csv')
    print("Spaltennamen der Stopwords CSV-Datei:", stopwords_df.columns.tolist())

    # Angenommen, die Stopwords sind in einer Spalte mit dem Namen 'stopwords' enthalten
    stopwords_column_name = 'stopwords'  # Passe dies an, falls die Spalte anders heißt
    stopwords = set(stopwords_df[stopwords_column_name].str.lower())

    # Häufigste Wörter in der Spalte 'Fault Description Customer'
    text = ' '.join(df_assistance['Fault Description Customer'].dropna().astype(str))
    words = re.findall(r'\b\w+\b', text.lower())

    # Filterung der Wörter, um Stopwords und Zahlen zu entfernen
    filtered_words = [word for word in words if word not in stopwords and not word.isdigit()]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(200)

    # Ausgabe der häufigsten Wörter
    print("Die XX am häufigsten verwendeten Wörter in der Spalte 'Fault Description Customer':")
    for word, count in most_common_words:
        print(f'{word}: {count}')

    # Speichern der häufigsten Wörter als CSV
    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])
    common_words_df.to_csv(output_path / 'most_common_words.csv', index=False)

    # CSV-Datei einlesen
    common_words_df = pd.read_csv('output/most_common_words.csv')

    # Daten visualisieren
    chart = alt.Chart(common_words_df).mark_bar().encode(
        x=alt.X('Word', sort='-y', title='Wort'),
        y=alt.Y('Count', title='Anzahl'),
        tooltip=['Word', 'Count']
    ).properties(
        title='Häufigste Wörter in der Spalte "Fault Description Customer"',
        width=800,
        height=400
    ).interactive()
    


#%% Häufigkeit der VINs zählen

    vin_counts = df_assistance['VIN'].value_counts()

    # Häufigkeiten in ein DataFrame umwandeln
    vin_counts_df = vin_counts.reset_index()
    vin_counts_df.columns = ['VIN', 'Frequency']

    # Histogramm der Häufigkeiten erstellen
    hist = alt.Chart(vin_counts_df).mark_bar().encode(
        alt.X('Frequency:Q', bin=alt.Bin(maxbins=30), title='Frequency'),
        alt.Y('count():Q', scale=alt.Scale(type='log'), title='Count (log scale)'),
        tooltip=[alt.Tooltip('Frequency:Q', title='Frequency'), alt.Tooltip('count():Q', title='Count')]
    ).properties(
        title='Histogram of VIN Frequencies (mit Log Scale)',
        width=800,
        height=400
    )

    # Normalverteilung der Häufigkeiten überprüfen
    mean = vin_counts_df['Frequency'].mean()
    std_dev = vin_counts_df['Frequency'].std()
    x = np.linspace(vin_counts_df['Frequency'].min(), vin_counts_df['Frequency'].max(), 100)
    y = norm.pdf(x, mean, std_dev)
    normal_dist_df = pd.DataFrame({'Frequency': x, 'Density': y})

    line = alt.Chart(normal_dist_df).mark_line(color='red').encode(
        x='Frequency:Q',
        y=alt.Y('Density:Q'),
        tooltip=[alt.Tooltip('Frequency:Q', title='Frequency'), alt.Tooltip('Density:Q', title='Density')]
    )

    # Kombiniertes Diagramm
    combined_chart = alt.layer(hist, line).resolve_scale(
        y='independent'
    ).properties(
        title='VIN Frequency Distribution with Normal Curve (Log Scale)'
    )

    # Speichern der Visualisierung als HTML
    combined_chart.save(output_path / 'vin_frequency_normal_distribution_log.html')

    # Ausgabe der Häufigkeiten als CSV
    vin_counts_df.to_csv(output_path / 'vin_frequency_counts.csv', index=False)

    # Untersuchung der Spalte 'VIN' mit describe
    print(vin_counts_df['Frequency'].describe())


#%% Kreuztabelle Component vs Outcome


        # Häufigkeit der Werte in der Spalte 'Component'
        component_value_counts = df_assistance['Component'].value_counts()
        print("Häufigkeit der Werte in der Spalte 'Component':")
        print(component_value_counts)

        # Häufigkeit der Werte in der Spalte 'Outcome Description'
        outcome_value_counts = df_assistance['Outcome Description'].value_counts()
        print("Häufigkeit der Werte in der Spalte 'Outcome Description':")
        print(outcome_value_counts)
    

    # Kreuztabelle erstellen und sortieren
    crosstab = pd.crosstab(df_assistance['Component'], df_assistance['Outcome Description'])
    crosstab['Total'] = crosstab.sum(axis=1)
    crosstab = crosstab.sort_values(by='Total', ascending=False).drop(columns='Total')

    # Kreuztabelle in ein DataFrame umwandeln für die Visualisierung
    crosstab_df = crosstab.reset_index().melt(id_vars='Component', var_name='Outcome Description', value_name='Count')

    # Visualisierung der Kreuztabelle
    heatmap = alt.Chart(crosstab_df).mark_rect().encode(
        alt.X('Outcome Description:O', title='Outcome Description',
              sort=alt.EncodingSortField(field='Count', op='sum', order='descending')),
        alt.Y('Component:O', title='Component',
              sort=alt.EncodingSortField(field='Count', op='sum', order='descending')),
        alt.Color('Count:Q',
                  scale=alt.Scale(
                      scheme='inferno',  # Farbskala
                      type='log',  # Logarithmische Skala
                      domain=[1, 10000]  # Wertebereich anpassen
                  ),
                  title='Count'),
        tooltip=[alt.Tooltip('Component:O', title='Component'),
                 alt.Tooltip('Outcome Description:O', title='Outcome Description'),
                 alt.Tooltip('Count:Q', title='Count')]
    ).properties(
        title='Heatmap of Component vs Outcome Description',
        width=800,
        height=600
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    ).configure_title(
        fontSize=16
    )

    # Speichern der Visualisierung als HTML
    heatmap.save(output_path / 'component_outcome_heatmap_log.html')

#%% Aufbereitung Odometer
    


    # Sicherstellen, dass 'Odometer' Werte nur numerisch sind, nicht numerische Werte werden zu NaN
    df_assistance['Odometer'] = pd.to_numeric(df_assistance['Odometer'], errors='coerce')

    # Berechnen der Quantile für die obersten und untersten x%
    lower_quantile = df_assistance['Odometer'].quantile(0.1)
    upper_quantile = df_assistance['Odometer'].quantile(0.9983)

    # perform_chi_square_test(data=df_assistance, col1='Component', col2='Outcome Description')
    # perform_chi_square_test(data=df_assistance, col1='Component', col2='Reason Of Call')
    # perform_chi_square_test(data=df_assistance, col1='Outcome Description', col2='Reason Of Call')
    # perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Component')
    # perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Outcome Description')
    # perform_chi_square_test(data=df_assistance, col1='Report Type', col2='Reason Of Call')
    # Ersetzen der Werte, die außerhalb der Quantile liegen, durch NaN
    df_assistance.loc[(df_assistance['Odometer'] <= lower_quantile) | (df_assistance['Odometer'] >= upper_quantile), 'Odometer'] = pd.NA

    # Filtern der Daten für die Visualisierung (nur gültige Werte)
    filtered_df = df_assistance.dropna(subset=['Odometer'])
    #%% Visualisirung

    # Erstellen des Histogramms mit logarithmisch skalierten y-Achse
    histogram = alt.Chart(filtered_df).mark_bar().encode(
        alt.X('Odometer:Q', bin=alt.Bin(maxbins=40), title='Kilometerstand'),
        alt.Y('count()', title='Anzahl der Fahrzeuge', scale=alt.Scale(type='log')),
        tooltip=[alt.Tooltip('Odometer:Q', title='Kilometerstand', bin=alt.Bin(maxbins=40)),
                 alt.Tooltip('count()', title='Anzahl der Fahrzeuge')]
    ).properties(
        title='Verteilung der Kilometerstände (logarithmische Skala)'
    )

    get_timeline(data=df_assistance, col='Incident Date', aggregate='Monat')
    # Speichern der Visualisierung als HTML
    histogram.save(output_path / 'kilometerstand_verteilung_log_skala.html')


    ### WORKSHOP ANALYSYS ###
    df_workshop = pd.read_csv('data/interim/workshop.csv')
    print(filtered_df["Odometer"].describe())
'''

    pass
