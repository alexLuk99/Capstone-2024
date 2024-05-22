from pathlib import Path
import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import norm
from collections import Counter
import re

def analyse_data() -> None:
    # CSV-Datei einlesen
    df_assistance = pd.read_csv('data/interim/assistance.csv', low_memory=False)
    df_assistance = df_assistance.convert_dtypes()

    # Erstellen des Ausgabeverzeichnisses
    output_path = Path('output')
    output_path.mkdir(exist_ok=True, parents=True)


    #%%

    # Remove rows where 'Reason Of Call' is empty
    df_assistance = df_assistance.dropna(subset=['Reason Of Call'])

    # Replace numeric values 7 and 8 in 'Reason Of Call' column
    df_assistance['Reason Of Call'] = df_assistance['Reason Of Call'].astype(str).replace({
        '7': 'Tyre Breakdown',
        '8': 'Tyre Accident'
    })

    # Count frequency of each category in 'Reason Of Call'
    value_counts = df_assistance['Reason Of Call'].value_counts().reset_index()
    value_counts.columns = ['Reason Of Call', 'Frequency']

    # Create a logarithmic bar chart with Altair
    bar_chart = alt.Chart(value_counts).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Frequency', scale=alt.Scale(type='log')),
        y=alt.Y('Reason Of Call:N', sort='-x', title='Reason Of Call'),
        tooltip=['Reason Of Call', 'Frequency']
    ).properties(
        title='Frequency of Categories in "Reason Of Call"',
        width=800,
        height=400
    )

    # Save the diagram
    bar_chart.save(output_path / 'reason_of_call_frequency_logarithmic.html')


    '''
    
    #%% Heatmep Bolean 

    # List of columns of interest
    columns_of_interest = [
        'Hotel Service', 'Alternative Transport', 'Taxi Service', 'Vehicle Transport',
        'Car Key Service', 'Parts Service', 'Personal Services', 'Damage During Assistance',
        'Additional Services Not Covered'
    ]

    # Ensure all columns are present in the dataframe
    df_selected = df_assistance[columns_of_interest]

    # Count and drop rows with missing values
    initial_row_count = df_selected.shape[0]
    df_selected = df_selected.dropna()
    deleted_row_count = initial_row_count - df_selected.shape[0]
    print(f"Deleted {deleted_row_count} rows with missing values")

    # Convert boolean values to binary
    df_binary = df_selected.applymap(lambda x: 1 if x == 'YES' else 0)

    # Calculate the correlation matrix
    correlation_matrix = df_binary.corr()

    # Melt the correlation matrix for visualization
    correlation_melt = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_melt.columns = ['Variable1', 'Variable2', 'Correlation']

    # Create a heatmap using Altair
    heatmap = alt.Chart(correlation_melt).mark_rect().encode(
        x='Variable1:O',
        y='Variable2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    ).properties(
        title='Correlation Heatmap of Boolean Variables',
        width=600,
        height=600
    )

    # Save the heatmap
    heatmap.save(output_path / 'correlation_heatmap.html')

    
#%%replacement cars distribution

    # Ensure the "Replacement Car Days" column is numeric and drop NaN values
    df_assistance['Replacement Car Days'] = pd.to_numeric(df_assistance['Replacement Car Days'], errors='coerce')
    df_assistance = df_assistance.dropna(subset=['Replacement Car Days'])

    # Calculate statistics
    stats = {
        'Mean': df_assistance['Replacement Car Days'].mean(),
        'Median': df_assistance['Replacement Car Days'].median(),
        'Q0.1': df_assistance['Replacement Car Days'].quantile(0.1),
        'Q0.9': df_assistance['Replacement Car Days'].quantile(0.9)
    }

    # Create histogram with hover details and logarithmic scale
    hist = alt.Chart(df_assistance).mark_bar().encode(
        alt.X('Replacement Car Days:Q', bin=alt.Bin(maxbins=20), title='Replacement Car Days'),
        alt.Y('count()', title='Frequency', scale=alt.Scale(type='log')),
        tooltip=[alt.Tooltip('Replacement Car Days:Q', bin=True, title='Replacement Car Days'),
                 alt.Tooltip('count()', title='Frequency')]
    ).properties(
        title='Histogram of Replacement Car Days with Statistical Indicators',
        width=800,
        height=400
    ).interactive()

    # Create rules for mean, median, and quartiles
    stats_df = pd.DataFrame(stats.items(), columns=['Statistic', 'Value'])
    stats_df['Color'] = stats_df['Statistic'].apply(
        lambda x: 'red' if x == 'Mean' else 'blue' if x == 'Median' else 'green')

    rules = alt.Chart(stats_df).mark_rule().encode(
        x='Value:Q',
        color=alt.Color('Color:N', scale=None),
        size=alt.value(2),
        tooltip=[alt.Tooltip('Value:Q', title='Value'), alt.Tooltip('Statistic:N', title='Statistic')]
    )

    # Combine the histogram with the statistical indicators
    chart = hist + rules

    # Save the chart
    chart.save(output_path / 'replacement_car_days_distribution.html')

#%%

   
#%% Zähle die Häufigkeit jedes Outcome Description

  
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

    # Speichern der Visualisierung als HTML
    histogram.save(output_path / 'kilometerstand_verteilung_log_skala.html')

    print(filtered_df["Odometer"].describe())
'''



analyse_data()
print("Ende")
